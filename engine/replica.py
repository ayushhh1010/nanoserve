"""gRPC replica server: one engine behind the router's wire contract.

This is the data plane. The router terminates client connections and decides
*which* replica; this process holds a model and actually generates.

Three things here are load-bearing for Phase 3 and easy to leave out:

**Idempotency.** A retry after a replica dies carries the same `request_id`. A
replica that already has that id attaches to the existing generation rather
than starting a second one. Without it every retry doubles the cluster's work
at exactly the moment it is least able to afford it -- retries arrive when
things are already failing.

**Resume tokens.** On migration the router replays the prompt plus whatever the
client already received. The replica continues from there instead of
restarting, so a mid-generation replica death costs the remaining tokens rather
than all of them, and the client never sees a token twice.

**Draining.** On SIGTERM the replica reports `draining` in its health response,
stops accepting, and finishes what it has. The router removes it from rotation
within one health interval. A deploy then costs no dropped requests, which is
the difference between a graceful rollout and an outage nobody attributes to
the deploy.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field

import grpc
import torch

from engine.admission import AdmissionConfig
from engine.block_manager import PagedCachePool, blocks_for_budget
from engine.pb import inference_pb2 as pb
from engine.pb import inference_pb2_grpc as rpc
from engine.scheduler import ContinuousBatchingEngine
from engine.server import AsyncEngine
from engine.types import FinishReason, Request, SamplingParams
from model.generate import load_model
from model.tokenizer import BPETokenizer

#: engine FinishReason -> proto enum. Explicit rather than by name so a rename
#: on either side is a failure here instead of a silently wrong wire value.
_FINISH = {
    FinishReason.EOS: pb.Finish.EOS,
    FinishReason.LENGTH: pb.Finish.LENGTH,
    FinishReason.CANCELLED: pb.Finish.CANCELLED,
    FinishReason.REJECTED: pb.Finish.REJECTED,
    FinishReason.PREEMPTED: pb.Finish.PREEMPTED,
}


@dataclass
class InFlight:
    """A generation the replica is running, keyed by idempotency id."""

    request: Request
    started: float = field(default_factory=time.perf_counter)
    #: Tokens already handed to a client, so a duplicate call can replay them.
    emitted: list[tuple[int, str]] = field(default_factory=list)


class InferenceServicer(rpc.InferenceServicer):
    def __init__(
        self,
        async_engine: AsyncEngine,
        tokenizer: BPETokenizer,
        replica_id: str,
        started_at: float,
    ) -> None:
        self.ae = async_engine
        self.tok = tokenizer
        self.replica_id = replica_id
        self.started_at = started_at
        self.draining = False
        self._inflight: dict[str, InFlight] = {}

    # -- generation ---------------------------------------------------------

    async def Generate(self, request: pb.GenerateRequest, context):  # noqa: N802
        if self.draining:
            # Refuse rather than accept-and-die. A draining replica that takes
            # one more request turns a graceful rollout into a dropped one.
            yield _finish(pb.Finish.REJECTED, 0, "replica is draining")
            return

        req_id = request.request_id or f"req-{uuid.uuid4().hex[:12]}"

        existing = self._inflight.get(req_id)
        if existing is not None:
            # Same idempotency key: this is a retry of something already
            # running. Replay what it has produced and then follow it, rather
            # than generating the same tokens a second time.
            async for chunk in self._follow(existing, context):
                yield chunk
            return

        prompt_ids = self.tok.encode(request.prompt, allowed_special=False)
        if not prompt_ids:
            yield _finish(pb.Finish.ERROR, 0, "prompt encodes to zero tokens")
            return

        # Migration: the client already holds these, so they become part of the
        # context rather than something to generate again.
        resume = list(request.resume_tokens)
        p = request.params
        remaining = max(1, int(p.max_tokens) - len(resume))

        engine_req = Request(
            prompt_token_ids=prompt_ids + resume,
            params=SamplingParams(
                max_tokens=remaining,
                temperature=float(p.temperature),
                top_k=int(p.top_k) or None,
                top_p=float(p.top_p) or None,
                ignore_eos=bool(p.ignore_eos),
            ),
            arrival_time=time.perf_counter(),
            deadline=_deadline(request.deadline_unix),
        )
        engine_req.request_id = req_id

        entry = InFlight(request=engine_req)
        self._inflight[req_id] = entry
        stream = await self.ae.submit(engine_req)

        if engine_req.finish_reason is FinishReason.REJECTED:
            self._inflight.pop(req_id, None)
            self.ae.release(req_id)
            yield _finish(pb.Finish.REJECTED, 0, engine_req.rejection_reason)
            return

        try:
            produced: list[int] = []
            emitted_chars = 0
            # Index continues past the resumed tokens so the client sees one
            # unbroken sequence across a migration.
            base = len(resume)

            while True:
                if context.cancelled():
                    break
                try:
                    token_id = await asyncio.wait_for(stream.queue.get(), timeout=0.25)
                except asyncio.TimeoutError:
                    if stream.done.is_set() and stream.queue.empty():
                        break
                    continue

                produced.append(token_id)
                text = self.tok.decode(produced)
                if text.endswith("�"):
                    # Half of a multi-byte character; wait for the rest rather
                    # than emit a replacement character mid-word.
                    continue
                if len(text) > emitted_chars:
                    piece = text[emitted_chars:]
                    emitted_chars = len(text)
                    index = base + len(entry.emitted)
                    entry.emitted.append((token_id, piece))
                    yield pb.GenerateChunk(
                        token=pb.Token(text=piece, token_id=token_id, index=index)
                    )

            reason = stream.finish_reason or FinishReason.CANCELLED
            yield _finish(
                _FINISH.get(reason, pb.Finish.REASON_UNSPECIFIED),
                base + len(produced),
            )
        finally:
            # Every exit path: normal finish, client cancel, and the
            # CancelledError a killed RPC raises. Releasing KV here is what
            # keeps an abandoned stream from leaking a sequence.
            await self.ae.cancel(req_id)
            self.ae.release(req_id)
            self._inflight.pop(req_id, None)

    async def _follow(self, entry: InFlight, context):
        """Replay an in-flight generation to a duplicate caller."""
        for index, (token_id, piece) in enumerate(list(entry.emitted)):
            yield pb.GenerateChunk(
                token=pb.Token(text=piece, token_id=token_id, index=index)
            )
        # The original call owns the stream; this one reports what it saw.
        yield _finish(pb.Finish.CANCELLED, len(entry.emitted), "duplicate request_id")

    # -- control ------------------------------------------------------------

    async def Health(self, request: pb.HealthRequest, context):  # noqa: N802
        s = self.ae.engine.stats()
        return pb.HealthResponse(
            ready=not self.draining,
            draining=self.draining,
            num_running=s.num_running,
            num_waiting=s.num_waiting,
            kv_utilization=float(s.kv_utilization),
            prefix_hit_rate=float(s.prefix_hit_rate),
            replica_id=self.replica_id,
            uptime_seconds=int(time.perf_counter() - self.started_at),
        )

    async def Drain(self, request: pb.DrainRequest, context):  # noqa: N802
        self.draining = True
        return pb.DrainResponse(in_flight=len(self._inflight))


def _finish(reason, tokens: int, message: str = "") -> pb.GenerateChunk:
    return pb.GenerateChunk(
        finish=pb.Finish(reason=reason, output_tokens=tokens, message=message)
    )


def _deadline(unix_seconds: float) -> float | None:
    """Convert an absolute wall-clock deadline to this process's monotonic clock.

    The wire carries epoch seconds because two machines share that and nothing
    else; `perf_counter` is meaningless across processes. Converting on arrival
    keeps the engine on one clock.
    """
    if not unix_seconds:
        return None
    return time.perf_counter() + (unix_seconds - time.time())


async def serve(
    # 9101, not the conventional 50051. Windows reserves 50000-50559 for
    # Hyper-V/WSL/Docker, so binding there fails with WinError 10013
    # ("permission denied") from a plain socket -- and gRPC reports it only as
    # a bare "failed to bind", which looks like the server never started.
    # `netsh interface ipv4 show excludedportrange protocol=tcp` lists them.
    port: int = 9101,
    checkpoint: str = "checkpoints/run1/best.pt",
    tokenizer_path: str = "model/tokenizer.json",
    device: str = "cuda",
    dtype: str = "bfloat16",
    kv_budget_mb: int = 1024,
    block_size: int = 16,
    max_batch_size: int = 64,
    max_queue_depth: int | None = 64,
    replica_id: str | None = None,
    #: IPv4 wildcard rather than "[::]". The IPv6 wildcard fails to bind on
    #: Windows here, and a bind failure surfaces at the client as a bare
    #: "connection refused" with nothing pointing at the server.
    host: str = "0.0.0.0",
) -> None:
    """Run one replica until cancelled."""
    started = time.perf_counter()
    replica_id = replica_id or f"replica-{uuid.uuid4().hex[:8]}"

    tok = BPETokenizer.load(tokenizer_path)
    model, _ = load_model(checkpoint, device=device, dtype=getattr(torch, dtype))
    pool = PagedCachePool(
        model.cfg,
        num_blocks=blocks_for_budget(model.cfg, kv_budget_mb * 1024 * 1024, block_size),
        block_size=block_size, device=device, dtype=getattr(torch, dtype),
    )
    engine = ContinuousBatchingEngine(
        model, pool, eos_token_id=tok.eos_id, device=device,
        max_batch_size=max_batch_size,
        admission=AdmissionConfig(max_queue_depth=max_queue_depth),
    )
    ae = AsyncEngine(engine)
    await ae.start()

    servicer = InferenceServicer(ae, tok, replica_id, started)
    server = grpc.aio.server()
    rpc.add_InferenceServicer_to_server(servicer, server)
    server.add_insecure_port(f"{host}:{port}")

    await server.start()
    print(f"{replica_id} listening on {host}:{port} "
          f"({pool.allocator.num_blocks:,} KV blocks)", flush=True)
    try:
        await server.wait_for_termination()
    finally:
        servicer.draining = True
        await server.stop(grace=10)
        await ae.stop()
