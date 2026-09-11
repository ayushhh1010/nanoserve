"""Continuous batching: sequences join and leave the batch on every forward pass.

The naive engine runs one request to completion before starting the next, so a
4-token request behind a 500-token one waits for all 500. Static batching is
barely better: a batch runs until its *longest* member finishes, and every other
slot idles at the end.

Iteration-level scheduling fixes both. Each step is one forward pass, and
between steps the scheduler:

    1. reaps finished sequences and returns their blocks
    2. admits waiting sequences if the pool has room
    3. runs one decode step over whatever is currently running

A sequence that finishes at step 40 frees its slot at step 40, and a request
that arrived at step 39 starts at step 41 -- not after the longest running
generation drains.

**Why this is where the paged cache pays.** Batch-1 decode on this hardware is
launch-bound: ~700 kernels at ~13us each, and the arithmetic for one token
barely registers. A batch of 32 launches the *same* ~700 kernels and does 32
tokens of work inside them, so per-token cost falls by nearly the batch factor.
The only thing standing between you and a large batch is KV memory, which is
what step 3 bought: at 3.9% waste instead of 72.4%, the same VRAM holds far more
concurrent sequences.

**Prefill is not batched with decode.** Prefills have different lengths and
would need padding to the longest, wasting most of the batch; mixing them with
decode also inflates inter-token latency for every running sequence, since one
long prefill stalls the step. Prefill runs per sequence, decode runs batched.
That is the original vLLM design, and chunked prefill (which does mix them, in
bounded pieces) is a later refinement.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from engine.admission import AdmissionConfig, AdmissionController
from engine.block_manager import OutOfBlocks, PagedCachePool, PagedKVCache
from engine.prefix_cache import PrefixCache
from engine.types import EngineStats, FinishReason, Request
from model.transformer import NanoForCausalLM


#: Reusable pinned staging buffers, keyed by size. Allocating pinned memory is
#: expensive, so the buffers are cached and grown rather than made per step.
_PINNED: dict[int, Tensor] = {}


def _pinned(size: int, device: torch.device) -> Tensor:
    """A pinned host buffer of at least `size` int64 elements.

    Pinned only matters on CUDA; on CPU the copy is a no-op and `pin_memory`
    would raise, so the buffer is plain there.
    """
    key = 1 << max(4, (size - 1).bit_length())  # round up to a power of two
    buf = _PINNED.get(key)
    if buf is None:
        # Allocated with inference mode explicitly off. A tensor created inside
        # inference_mode is an "inference tensor" and can never be mutated
        # outside it again -- which for a buffer cached across calls means the
        # first step poisons every later one.
        with torch.inference_mode(False):
            buf = torch.empty(key, dtype=torch.long)
            if device.type == "cuda":
                buf = buf.pin_memory()
        _PINNED[key] = buf
    return buf[:size]


@dataclass
class Sequence:
    """A request that has been admitted and has KV allocated."""

    request: Request
    cache: PagedKVCache
    #: The token fed to the next forward pass.
    next_token: int = 0
    finished: bool = False
    finish_reason: FinishReason | None = None

    @property
    def position(self) -> int:
        """Absolute position of the next token, for RoPE."""
        return len(self.cache)


class BatchedPagedCache:
    """One forward pass over N sequences, each with its own block table.

    Implements the model's `KVCache` protocol so the transformer is unchanged.
    The whole point is doing the scatter and the gather *once per layer* rather
    than once per sequence per layer: at batch 32 the per-sequence version
    would launch 512 extra kernels per step, which on a launch-bound machine
    costs more than the batching saves.

    Decode only -- every sequence contributes exactly one token.
    """

    def __init__(self, pool: PagedCachePool, seqs: list[Sequence]) -> None:
        self.pool = pool
        self.seqs = seqs
        self._prepared = False
        self.write_slots: Tensor | None = None
        self.read_index: Tensor | None = None
        self.mask: Tensor | None = None

    def _prepare(self) -> None:
        """Grow every sequence by one token and build the index tensors.

        Runs once per forward pass, on layer 0. Slot arithmetic is done on the
        host from the block tables -- which are plain Python lists -- so no GPU
        value is ever read back, which would drain the CUDA queue every step.
        """
        bs = self.pool.block_size

        # Two phases. Growing sequences one at a time and discovering
        # exhaustion halfway leaves the earlier ones already advanced, with
        # num_tokens past what their block table can address -- a corrupted
        # batch rather than a clean failure. So count first, commit second.
        need = 0
        for seq in self.seqs:
            cache = seq.cache
            if cache.num_tokens % bs == 0:
                need += 1  # crosses into a fresh block
            elif cache.block_table and self.pool.allocator.ref_count(cache.block_table[-1]) > 1:
                need += 1  # copy-on-write of a shared tail
        if need > self.pool.allocator.num_free:
            raise OutOfBlocks(
                f"batch of {len(self.seqs)} needs {need} blocks, "
                f"{self.pool.allocator.num_free} free"
            )

        write = []
        for seq in self.seqs:
            cache = seq.cache
            if cache.num_tokens % bs != 0:
                cache._unshare_tail()
            cache._ensure_capacity(1)
            pos = cache.num_tokens
            cache.num_tokens += 1
            write.append(cache.block_table[pos // bs] * bs + (pos % bs))

        device = self.pool.device
        lengths = [len(s.cache) for s in self.seqs]
        max_len = max(lengths)

        # One host-to-device copy, not two. A transfer out of ordinary pageable
        # memory is synchronous, so each one drains the CUDA queue -- measured
        # at ~45ms per step on this machine, which at small batch sizes dwarfs
        # the forward pass itself. Pinned staging makes the copy async, and
        # packing both vectors into a single buffer halves the number of them.
        n = len(self.seqs)
        staging = _pinned(2 * n, device)
        staging[:n] = torch.tensor(write, dtype=torch.long)
        staging[n:] = torch.tensor(lengths, dtype=torch.long)
        packed = staging.to(device, non_blocking=True)
        self.write_slots = packed[:n]
        valid = packed[n:]

        # Right-pad each sequence's slot list to the batch maximum. The inputs
        # are already device tensors, so this stays on device. Padding points at
        # slot 0 and is removed by the mask rather than by omission -- SDPA
        # needs a rectangular tensor.
        self.read_index = pad_sequence(
            [s.cache._slots[: len(s.cache)] for s in self.seqs],
            batch_first=True,
            padding_value=0,
        )
        # (B, 1, 1, max_len): broadcasts over heads and the single query.
        self.mask = (
            torch.arange(max_len, device=device)[None, :] < valid[:, None]
        )[:, None, None, :]
        self._prepared = True

    def __len__(self) -> int:
        """Longest sequence in the batch. Only used for RoPE capacity."""
        return max((len(s.cache) for s in self.seqs), default=0)

    def update(self, layer_idx: int, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        if layer_idx == 0 and not self._prepared:
            self._prepare()

        # k, v arrive as (B, H, 1, D). One scatter for the whole batch.
        self.pool.k[layer_idx][self.write_slots] = k[:, :, 0]
        self.pool.v[layer_idx][self.write_slots] = v[:, :, 0]

        # (B, max_len, H, D) -> (B, H, max_len, D). One gather for the batch.
        kk = self.pool.k[layer_idx][self.read_index].permute(0, 2, 1, 3)
        vv = self.pool.v[layer_idx][self.read_index].permute(0, 2, 1, 3)
        return kk, vv


class ContinuousBatchingEngine:
    """Iteration-level scheduling over a paged KV pool."""

    name = "continuous"

    def __init__(
        self,
        model: NanoForCausalLM,
        pool: PagedCachePool,
        eos_token_id: int,
        device: str = "cuda",
        max_batch_size: int = 64,
        #: Floor for the headroom kept back so running sequences can grow.
        #: The actual reserve also scales with the batch: in the worst case
        #: every running sequence crosses a block boundary on the same step, so
        #: headroom below `len(running)` lets admission starve the batch it
        #: just joined. A fixed reserve looks fine until the batch outgrows it.
        reserve_blocks: int = 8,
        #: Reuse KV blocks across requests sharing a prefix. Off gives the
        #: same engine without the cache, which is the honest comparison.
        enable_prefix_cache: bool = True,
        #: None disables admission control entirely, which is the naive
        #: unbounded queue the goodput comparison is made against.
        admission: AdmissionConfig | None = None,
    ) -> None:
        self.model = model.eval()
        self.pool = pool
        self.eos_token_id = eos_token_id
        self.device = torch.device(device)
        self.max_batch_size = max_batch_size
        self.reserve_blocks = reserve_blocks

        self.prefix_cache = PrefixCache(pool, enabled=enable_prefix_cache)
        self.admission = AdmissionController(admission) if admission else None

        self.waiting: deque[Request] = deque()
        self.running: list[Sequence] = []
        self.finished: list[Request] = []
        self._rejected: list[Request] = []
        self._cancelled: list[Request] = []
        self._cancel_requested: set[str] = set()
        self._emitted: list[tuple[Request, int]] = []
        #: How often memory pressure forced an eviction. A healthy server runs
        #: at zero; a non-zero count means the pool is undersized for the load.
        self.preemptions = 0

    # -- queue --------------------------------------------------------------

    def add_request(self, request: Request) -> None:
        """Admit or reject on arrival.

        The decision happens here rather than at scheduling time because the
        entire value of a rejection is that it is immediate. A 429 delivered
        after thirty seconds of queueing has already cost the client the thing
        it was trying to avoid.
        """
        if self.admission is not None:
            ok, reason = self.admission.admit(
                request,
                queue_depth=len(self.waiting),
                queued_tokens=sum(r.params.max_tokens for r in self.waiting),
                running_tokens=sum(
                    s.request.params.max_tokens - s.request.output_len for s in self.running
                ),
            )
            if not ok:
                request.finish(FinishReason.REJECTED)
                request.rejection_reason = reason
                self.finished.append(request)
                self._rejected.append(request)
                return
        self.waiting.append(request)

    def request_cancel(self, request_id: str) -> None:
        """Ask for a request to be cancelled at the next safe point.

        Deliberately does NOT touch `running` or `waiting`. The scheduler step
        may be executing on another thread -- the server drives it through
        `asyncio.to_thread` so the event loop stays free to notice disconnects
        -- and mutating the batch underneath a step in flight corrupts it. A
        cancellation is therefore recorded and applied by the owning thread.
        """
        self._cancel_requested.add(request_id)

    def apply_cancellations(self) -> list[Request]:
        """Apply pending cancellations. Called by whoever owns the step.

        Runs at the top of `step`, and from the idle path when nothing is
        running -- otherwise a cancellation arriving on an idle engine would
        wait for the next request before taking effect.
        """
        if not self._cancel_requested:
            return []

        wanted, self._cancel_requested = self._cancel_requested, set()
        done: list[Request] = []

        for seq in [s for s in self.running if s.request.request_id in wanted]:
            # Freeing here is the whole point. A cancelled generation that
            # keeps its blocks is the classic serving leak: the pool drains one
            # abandoned request at a time, throughput decays, and nothing ever
            # raises. It is invisible to every test that only exercises
            # requests which finish.
            seq.cache.free()
            seq.request.finish(FinishReason.CANCELLED)
            self.finished.append(seq.request)
            done.append(seq.request)
            self.running.remove(seq)

        for req in [r for r in self.waiting if r.request_id in wanted]:
            # A client can hang up before its request ever starts.
            self.waiting.remove(req)
            req.finish(FinishReason.CANCELLED)
            self.finished.append(req)
            done.append(req)

        self._cancelled.extend(done)
        return done

    def cancel(self, request_id: str) -> bool:
        """Cancel synchronously. Safe only when no step is in flight.

        Used by tests and single-threaded callers. The server uses
        `request_cancel` plus `apply_cancellations` instead.
        """
        self.request_cancel(request_id)
        return bool(self.apply_cancellations())

    def drain_cancelled(self) -> list[Request]:
        """Requests cancelled since the last call."""
        out, self._cancelled = self._cancelled, []
        return out

    def drain_emitted(self) -> list[tuple[Request, int]]:
        """(request, token) pairs produced since the last call."""
        out, self._emitted = self._emitted, []
        return out

    def drain_rejected(self) -> list[Request]:
        """Requests turned away since the last call.

        Surfaced separately so the harness can count them without treating a
        rejection as a completion -- they are the numerator of the rejection
        rate and must never reach the goodput numerator.
        """
        out, self._rejected = self._rejected, []
        return out

    def has_work(self) -> bool:
        """Is there anything left to *execute*?

        Deliberately excludes pending rejections. They are outcomes to be
        collected, not work -- and including them made `run()` spin forever,
        since `step()` has no reason to touch them.
        """
        return bool(self.waiting or self.running)

    def stats(self) -> EngineStats:
        a = self.pool.allocator
        return EngineStats(
            timestamp=time.perf_counter(),
            num_running=len(self.running),
            num_waiting=len(self.waiting),
            num_finished=len(self.finished),
            kv_blocks_used=a.num_allocated,
            kv_blocks_total=a.num_blocks,
            prefix_cache_hits=self.prefix_cache.stats.hit_blocks,
            prefix_cache_queries=self.prefix_cache.stats.query_blocks,
        )

    # -- scheduling ---------------------------------------------------------

    def _blocks_needed(self, req: Request) -> int:
        return -(-req.prompt_len // self.pool.block_size)

    def _admit(self) -> tuple[list[Sequence], list[Request]]:
        """Take waiting requests while there is room, and prefill them.

        Admission is checked against the prompt plus a reserve, not against the
        whole possible generation. Reserving the worst case would put us back
        at the contiguous allocator's concurrency limit and undo step 3.
        """
        admitted: list[Sequence] = []
        # A request can finish during prefill -- max_tokens=1 needs no decode
        # step at all. Retiring it here without returning it means the engine
        # completes work the caller never learns about.
        done: list[Request] = []
        while self.waiting and len(self.running) + len(admitted) < self.max_batch_size:
            req = self.waiting[0]
            need = self._blocks_needed(req) + 1
            # One spare block per already-running sequence, plus the floor.
            reserve = max(self.reserve_blocks, len(self.running) + len(admitted) + 1)
            if self.pool.allocator.num_free < need + reserve:
                # Cached prefixes are the first thing to give up: they are an
                # optimisation, and a request that cannot start is not.
                if not self.prefix_cache.ensure_free(need + reserve):
                    break
            if (
                not self.running
                and not admitted
                and need + reserve > self.pool.allocator.num_blocks
            ):
                # Nothing running to evict, and the pool could never hold this
                # request even when empty. Rejecting beats looping forever.
                self.waiting.popleft()
                req.finish(FinishReason.REJECTED)
                self.finished.append(req)
                done.append(req)
                continue

            self.waiting.popleft()
            req.scheduled_time = time.perf_counter()
            cache = self.pool.allocate()
            if cache is None:
                self.waiting.appendleft(req)
                break

            try:
                logits = self._prefill(req, cache)
                token = self._sample(logits[:, -1], req)
            except OutOfBlocks:
                cache.free()
                req.finish(FinishReason.PREEMPTED)
                self.finished.append(req)
                done.append(req)
                continue

            seq = Sequence(request=req, cache=cache, next_token=int(token.item()))
            before = req.output_len
            self._accept_token(seq, seq.next_token)
            if req.output_len > before:
                self._emitted.append((req, seq.next_token))
            if not seq.finished:
                admitted.append(seq)
            else:
                done.append(self._retire(seq))
        return admitted, done

    def _prefill(self, req: Request, cache: PagedKVCache) -> Tensor:
        """Run the prompt, skipping any prefix already in the block cache.

        The cached blocks are attached to the sequence and only the remaining
        suffix goes through the model. Positions continue from where the cache
        left off, so the suffix sees exactly the context it would have seen in
        a full prefill -- which is why the reused blocks have to have been
        produced by an identical prefix, and why the hash is chained.

        The last token of the prompt is never served from cache. The model has
        to produce logits for it to sample the first output token, and a fully
        cached prompt would leave nothing to run.
        """
        ids = req.prompt_token_ids
        reusable = ids[:-1]  # always leave one token to feed the model

        blocks = self.prefix_cache.match(reusable)
        n_cached = self.prefix_cache.attach(cache, blocks)

        suffix = ids[n_cached:]
        tokens = torch.tensor([suffix], device=self.device)
        self.model.model.rotary.ensure_capacity(len(ids) + 1)
        positions = torch.arange(
            n_cached, n_cached + len(suffix), device=self.device
        ).unsqueeze(0)

        logits, _ = self.model(tokens, position_ids=positions, cache=cache)

        # Register whatever full blocks this prompt produced, including the
        # ones just reused -- re-inserting an existing key is a no-op that
        # refreshes its LRU position.
        self.prefix_cache.insert(ids, cache.block_table)
        return logits

    def _accept_token(self, seq: Sequence, token_id: int) -> None:
        """Record a produced token and apply stop conditions."""
        req = seq.request
        if token_id == self.eos_token_id and not req.params.ignore_eos:
            seq.finished, seq.finish_reason = True, FinishReason.EOS
            return
        req.record_token(token_id)
        if req.output_len >= req.params.max_tokens:
            seq.finished, seq.finish_reason = True, FinishReason.LENGTH

    def _preempt_newest(self) -> None:
        """Evict the most recently admitted sequence back onto the queue.

        Its generated tokens are discarded and its timing reset, because it
        will be re-run from the prompt. Keeping partial output would report a
        request as having produced its early tokens twice.
        """
        if not self.running:
            raise OutOfBlocks("pool exhausted with nothing running to evict")

        victim = self.running.pop()
        victim.cache.free()
        req = victim.request
        req.output_token_ids.clear()
        req.token_times.clear()
        req.first_token_time = None
        req.scheduled_time = None
        self.preemptions += 1
        self.waiting.appendleft(req)

    def _retire(self, seq: Sequence) -> Request:
        seq.cache.free()
        seq.request.finish(seq.finish_reason or FinishReason.LENGTH)
        self.finished.append(seq.request)
        return seq.request

    # -- execution ----------------------------------------------------------

    @torch.inference_mode()
    def step(self) -> list[Request]:
        """One scheduling iteration: reap, admit, then a single decode pass."""
        cancelled = self.apply_cancellations()

        new_seqs, done_in_admit = self._admit()
        done_in_admit = cancelled + done_in_admit
        self.running.extend(new_seqs)
        if not self.running:
            return done_in_admit

        batch = self.running
        tokens = torch.tensor(
            [[s.next_token] for s in batch], dtype=torch.long, device=self.device
        )
        positions = torch.tensor(
            [[s.position] for s in batch], dtype=torch.long, device=self.device
        )

        # RoPE tables only auto-grow when the model derives positions itself.
        # This engine always supplies them, so honouring that contract is the
        # caller's job -- and getting it wrong is a device-side assert deep in
        # an unrelated kernel rather than a readable error.
        self.model.model.rotary.ensure_capacity(max(s.position for s in batch) + 1)

        cache = BatchedPagedCache(self.pool, batch)
        step_started = time.perf_counter()
        try:
            logits, _ = self.model(
                tokens, position_ids=positions, cache=cache, attn_mask=cache_mask(cache)
            )
        except OutOfBlocks:
            # The pool ran dry. Evict the newest sequence and put it back on
            # the queue: it has made the least progress, so recomputing it
            # wastes the least work, and evicting the oldest would starve
            # whoever has already waited longest.
            #
            # This is preemption by recompute -- the victim's blocks are freed
            # and its prompt is re-prefilled when memory allows. The
            # alternative, swapping its KV out to host memory, preserves the
            # work but spends PCIe bandwidth; which wins depends on sequence
            # length.
            self._preempt_newest()
            return done_in_admit

        # One host sync per step, not per sequence: stop conditions depend on
        # the token values, so they have to come back to the host once.
        next_ids = self._sample_batch(logits[:, -1], batch).tolist()

        # That sync makes this a real elapsed time rather than a queue-depth
        # measurement, so it is safe to feed the throughput estimate.
        if self.admission is not None:
            self.admission.record_step(time.perf_counter() - step_started, len(batch))

        done: list[Request] = list(done_in_admit)
        still_running: list[Sequence] = []
        for seq, token_id in zip(batch, next_ids):
            seq.next_token = token_id
            before = seq.request.output_len
            self._accept_token(seq, token_id)
            if seq.request.output_len > before:
                # Recorded per step so a streaming transport can forward tokens
                # as they are produced rather than polling request state, which
                # would race with the next step overwriting it.
                self._emitted.append((seq.request, token_id))
            if seq.finished:
                done.append(self._retire(seq))
            else:
                still_running.append(seq)

        self.running = still_running
        return done

    # -- sampling -----------------------------------------------------------

    def _sample(self, logits: Tensor, req: Request) -> Tensor:
        if req.params.greedy:
            return logits.argmax(-1)
        return _sample_with_params(logits, req).squeeze(-1)

    def _sample_batch(self, logits: Tensor, batch: list[Sequence]) -> Tensor:
        """Sample the whole batch at once when every request is greedy.

        The common case by far, and worth special-casing: a per-sequence loop
        would launch B small kernels per step for no benefit.
        """
        if all(s.request.params.greedy for s in batch):
            return logits.argmax(-1)
        out = [self._sample(logits[i : i + 1], s.request) for i, s in enumerate(batch)]
        return torch.cat(out)

    # -- convenience --------------------------------------------------------

    def run(self, requests: list[Request]) -> list[Request]:
        """Drain a list of requests. Rejections are returned alongside them."""
        out: list[Request] = []
        for r in requests:
            self.add_request(r)
            out.extend(self.drain_rejected())
        while self.has_work():
            out.extend(self.step())
            out.extend(self.drain_rejected())
        return out


def cache_mask(cache: BatchedPagedCache) -> Tensor | None:
    """The padding mask, materialised on layer 0 by the cache itself.

    The model needs the mask *before* the first `update` call, so this forces
    preparation up front. Without a mask a decode step attends to the padding
    of shorter sequences -- which does not crash and quietly corrupts every
    sequence shorter than the batch maximum.
    """
    if not cache._prepared:
        cache._prepare()
    return cache.mask


def _sample_with_params(logits: Tensor, req: Request) -> Tensor:
    p = req.params
    scaled = logits / p.temperature
    if p.top_k:
        k = min(p.top_k, scaled.size(-1))
        threshold = torch.topk(scaled, k, dim=-1).values[..., -1, None]
        scaled = scaled.masked_fill(scaled < threshold, float("-inf"))
    if p.top_p is not None and p.top_p < 1.0:
        ordered, index = torch.sort(scaled, descending=True, dim=-1)
        probs = torch.softmax(ordered, dim=-1)
        drop = probs.cumsum(dim=-1) - probs > p.top_p
        scaled = scaled.masked_fill(drop.scatter(-1, index, drop), float("-inf"))
    return torch.multinomial(torch.softmax(scaled, dim=-1), num_samples=1)
