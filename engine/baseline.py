"""The naive engine. This is the denominator.

One request at a time, run to completion before the next one starts. That is
not a strawman -- it is what a serving loop looks like when nobody has thought
about it yet, and it is the honest thing to measure against, provided it is
*competently* naive: no accidental host syncs, no cold-clock benchmarking, no
missing `inference_mode`. A baseline that is slow for uninteresting reasons
inflates every later speedup and the claim evaporates the moment anyone asks
"faster than what?".

Two failure modes are visible here and are the whole point of the phase:

**Head-of-line blocking.** A request arriving one millisecond after a 500-token
generation starts waits for all 500 tokens. Its queue time is somebody else's
generation length, which has nothing to do with its own work. Continuous
batching in week 6 exists to kill this.

**Quadratic recompute without a cache.** With `use_cache=False` the model
re-reads the entire prefix on every single token, so generating n tokens is
O(n^2) attention. `use_cache=True` fixes that but reallocates and copies the
whole cache each step, which is what the paged allocator replaces in week 5.
Both are measurable here, which makes the improvement in each later step
attributable to that step.
"""

from __future__ import annotations

import time
from collections import deque

import torch
import torch.nn.functional as F

from engine.block_manager import OutOfBlocks, PagedCachePool
from engine.kv_cache import StaticCachePool
from engine.types import EngineStats, FinishReason, Request
from model.transformer import DynamicCache, NanoForCausalLM


class BaselineEngine:
    """Sequential, one request at a time."""

    name = "baseline"

    def __init__(
        self,
        model: NanoForCausalLM,
        eos_token_id: int,
        device: str = "cuda",
        use_cache: bool = True,
        pool: StaticCachePool | PagedCachePool | None = None,
    ) -> None:
        self.model = model.eval()
        self.eos_token_id = eos_token_id
        self.device = torch.device(device)
        self.use_cache = use_cache
        # A StaticCachePool hands out contiguous per-sequence slots; a
        # PagedCachePool hands out block tables. The loop below does not know
        # or care which -- both satisfy the model's KVCache protocol, which is
        # the seam this was designed around in week 1.
        self.pool = pool

        self.waiting: deque[Request] = deque()
        self.finished: list[Request] = []
        self._tokens_generated = 0

    # -- queue --------------------------------------------------------------

    def add_request(self, request: Request) -> None:
        self.waiting.append(request)

    def has_work(self) -> bool:
        return bool(self.waiting)

    def stats(self) -> EngineStats:
        stats = EngineStats(
            timestamp=time.perf_counter(),
            num_running=0,
            num_waiting=len(self.waiting),
            num_finished=len(self.finished),
        )
        if isinstance(self.pool, PagedCachePool):
            a = self.pool.allocator
            stats.kv_blocks_used = a.num_allocated
            stats.kv_blocks_total = a.num_blocks
        elif self.pool is not None:
            p = self.pool.stats()
            stats.kv_blocks_used = p.tokens_used
            stats.kv_blocks_total = p.tokens_capacity
        return stats

    # -- execution ----------------------------------------------------------

    @torch.inference_mode()
    def step(self) -> list[Request]:
        """Run the next queued request to completion.

        A "step" for this engine is an entire request, which is precisely the
        problem. For the continuous-batching engine a step will be one forward
        pass over the whole batch; the harness drives both through this same
        method, so the difference shows up in the measurements rather than in
        the benchmark code.
        """
        if not self.waiting:
            return []

        req = self.waiting.popleft()
        req.scheduled_time = time.perf_counter()

        cache = None
        if self.use_cache:
            if self.pool is not None:
                cache = self.pool.allocate(expected_len=req.total_len_estimate)
                if cache is None:  # contiguous pool full
                    # A full pool is a scheduling condition. This engine has
                    # nowhere to put the request, so it goes back on the queue;
                    # week 7 turns this into a deliberate admission decision.
                    self.waiting.appendleft(req)
                    return []
            else:
                cache = DynamicCache()

        if self.use_cache:
            prompt = torch.tensor([req.prompt_token_ids], device=self.device)
            logits, _ = self.model(prompt, cache=cache)
            buffer = None
        else:
            # Preallocate the whole sequence on device and grow a view into it.
            # The obvious version -- rebuilding `torch.tensor(prompt + output)`
            # each step -- costs a Python list concat and a fresh host-to-device
            # copy per token, and measurement showed *that*, not the quadratic
            # attention, was most of the no-cache penalty. A baseline has to be
            # slow for the reason it is supposed to illustrate, or the number it
            # anchors is not about what it claims to be about.
            n_prompt = req.prompt_len
            buffer = torch.empty(
                (1, n_prompt + req.params.max_tokens), dtype=torch.long, device=self.device
            )
            buffer[0, :n_prompt] = torch.tensor(req.prompt_token_ids, device=self.device)
            logits, _ = self.model(buffer[:, :n_prompt])

        try:
            self._generate(req, logits, cache, buffer)
        except OutOfBlocks:
            # The pool cannot hold this sequence. For a sequential engine no
            # amount of waiting helps -- nothing else is running to free
            # anything -- so the request fails with whatever it produced.
            # Step 5 (preemption) turns this into a recoverable event by
            # evicting a victim instead.
            req.finish(FinishReason.PREEMPTED)
        finally:
            # Freeing must happen on every path, including the failing one.
            # Without this the pool drains one request at a time, throughput
            # decays to zero, and nothing raises to say why.
            if cache is not None and self.pool is not None:
                if isinstance(self.pool, PagedCachePool):
                    cache.free()
                else:
                    self.pool.free(cache)

        if req.finish_reason is None:
            req.finish(FinishReason.LENGTH)

        self.finished.append(req)
        return [req]

    def _generate(self, req: Request, logits, cache, buffer) -> None:
        """The decode loop. Raises OutOfBlocks if the pool runs dry."""
        for _ in range(req.params.max_tokens):
            token = self._sample(logits[:, -1], req)

            # `.item()` synchronises, and that is inherent rather than
            # sloppy: the stop condition depends on the token's value, so the
            # host has to see it. Real engines sync once per step for the same
            # reason.
            token_id = int(token.item())

            if token_id == self.eos_token_id and not req.params.ignore_eos:
                req.finish(FinishReason.EOS)
                return

            req.record_token(token_id)
            self._tokens_generated += 1

            if self.use_cache:
                logits, _ = self.model(token.view(1, 1), cache=cache)
            else:
                # No cache: re-attend over the whole sequence every token, so
                # the generation is O(n^2) in attention work. The token is
                # written into the preallocated buffer and the model sees a
                # slice -- no reallocation, no host round-trip.
                end = req.prompt_len + req.output_len
                buffer[0, end - 1] = token
                logits, _ = self.model(buffer[:, :end])

    def _sample(self, logits: torch.Tensor, req: Request) -> torch.Tensor:
        p = req.params
        if p.greedy:
            return logits.argmax(-1)

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
        return torch.multinomial(torch.softmax(scaled, dim=-1), num_samples=1).squeeze(-1)

    # -- convenience --------------------------------------------------------

    def run(self, requests: list[Request]) -> list[Request]:
        """Drain a list of requests. Ignores arrival times.

        Useful for correctness checks and throughput-only measurements. The
        latency benchmark drives `step()` against a real arrival schedule
        instead, because a closed loop cannot show queueing behaviour.
        """
        for r in requests:
            self.add_request(r)
        out: list[Request] = []
        while self.has_work():
            out.extend(self.step())
        return out
