"""HTTP server: SSE token streaming, and cancellation that actually frees memory.

Two things here are easy to get wrong and expensive to get wrong quietly.

**The engine loop must not run on the request's task.** Each connection is a
coroutine; the engine is one shared, synchronous scheduler. Stepping it from
whichever request happens to be awake would serialise the server on that
request and give every other connection its latency. So one background task
drives `engine.step()` and connections read from per-request queues.

**A disconnect has to reach the scheduler.** FastAPI's `StreamingResponse`
does not tell a generator that its client has gone; the generator simply stops
being consumed. If nothing notices, the sequence keeps generating, keeps its KV
blocks, and the pool drains one abandoned request at a time. Throughput decays
and nothing ever raises. So the generator polls `request.is_disconnected()` and
cancels explicitly, and `finally` cancels again on any exit path -- including
`asyncio.CancelledError`, which is what a cancelled task actually sees.

The engine step is synchronous and holds the GIL for its duration, so it runs
in a thread via `asyncio.to_thread`. That keeps the event loop free to accept
connections and notice disconnects while a forward pass is in flight.
"""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import AsyncIterator

import torch
from fastapi import FastAPI, HTTPException, Request as HTTPRequest
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from engine.admission import AdmissionConfig
from engine.block_manager import PagedCachePool, blocks_for_budget
from engine.scheduler import ContinuousBatchingEngine
from engine.types import FinishReason, Request, SamplingParams
from model.generate import load_model
from model.tokenizer import BPETokenizer


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = Field(default=128, ge=1, le=4096)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_k: int | None = Field(default=None, ge=1)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    ignore_eos: bool = False
    #: Seconds from arrival. Drives admission control's deadline projection.
    slo_seconds: float | None = None


@dataclass
class Stream:
    """A connection's view of one in-flight request."""

    request: Request
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    done: asyncio.Event = field(default_factory=asyncio.Event)
    finish_reason: FinishReason | None = None


class AsyncEngine:
    """Drives the scheduler on a background task and fans tokens out to streams."""

    def __init__(self, engine: ContinuousBatchingEngine, idle_sleep: float = 0.001) -> None:
        self.engine = engine
        self.idle_sleep = idle_sleep
        self.streams: dict[str, Stream] = {}
        self._task: asyncio.Task | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop = asyncio.Event()

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._stop.clear()
        self._task = asyncio.create_task(self._run(), name="engine-loop")

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            await asyncio.wait_for(self._task, timeout=10)
            self._task = None

    async def submit(self, req: Request) -> Stream:
        stream = Stream(request=req)
        self.streams[req.request_id] = stream
        self.engine.add_request(req)
        # Admission rejects synchronously, so the outcome may already exist.
        self._fanout()
        return stream

    async def cancel(self, request_id: str, timeout: float = 2.0) -> bool:
        """Request cancellation and wait for the engine thread to apply it.

        The wait is what makes the KV release observable: without it a caller
        cannot tell whether the blocks came back, and `finally` handlers would
        return before the sequence was actually gone.
        """
        stream = self.streams.get(request_id)
        known = stream is not None and not stream.done.is_set()
        self.engine.request_cancel(request_id)

        deadline = asyncio.get_running_loop().time() + timeout
        while asyncio.get_running_loop().time() < deadline:
            if not self.engine._cancel_requested:
                break
            await asyncio.sleep(0.001)
        self._fanout()
        return known

    # -- the loop -----------------------------------------------------------

    async def _run(self) -> None:
        while not self._stop.is_set():
            if not self.engine.has_work():
                # Apply cancellations even when idle -- otherwise one arriving
                # on an empty engine would wait for the next request.
                self.engine.apply_cancellations()
                self._fanout()
                await asyncio.sleep(self.idle_sleep)
                continue
            # The step is synchronous and CPU/GPU bound. Running it in a thread
            # keeps the event loop free to accept connections and notice
            # disconnects while a forward pass is in flight.
            await asyncio.to_thread(self.engine.step)
            self._fanout()

    def _fanout(self) -> None:
        """Move engine output into per-request queues.

        Drained through the engine's explicit accessors rather than by
        inspecting request state, which would race with the next step.
        """
        for req, token_id in self.engine.drain_emitted():
            stream = self.streams.get(req.request_id)
            if stream is not None:
                stream.queue.put_nowait(token_id)

        for req in (
            self.engine.drain_rejected()
            + self.engine.drain_cancelled()
            + [r for r in self.engine.finished if r.request_id in self.streams]
        ):
            stream = self.streams.get(req.request_id)
            if stream is not None and req.finish_reason is not None and not stream.done.is_set():
                stream.finish_reason = req.finish_reason
                stream.done.set()

    def release(self, request_id: str) -> None:
        self.streams.pop(request_id, None)


def build_app(
    checkpoint: str = "checkpoints/run1/best.pt",
    tokenizer_path: str = "model/tokenizer.json",
    device: str = "cuda",
    dtype: str = "bfloat16",
    kv_budget_mb: int = 1024,
    block_size: int = 16,
    max_batch_size: int = 64,
    max_queue_depth: int | None = 64,
    enable_prefix_cache: bool = True,
    poll_seconds: float = 0.05,
) -> FastAPI:
    state: dict = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        tok = BPETokenizer.load(tokenizer_path)
        model, meta = load_model(checkpoint, device=device, dtype=getattr(torch, dtype))
        pool = PagedCachePool(
            model.cfg,
            num_blocks=blocks_for_budget(model.cfg, kv_budget_mb * 1024 * 1024, block_size),
            block_size=block_size, device=device, dtype=getattr(torch, dtype),
        )
        engine = ContinuousBatchingEngine(
            model, pool, eos_token_id=tok.eos_id, device=device,
            max_batch_size=max_batch_size, enable_prefix_cache=enable_prefix_cache,
            admission=AdmissionConfig(max_queue_depth=max_queue_depth),
        )
        state["tok"] = tok
        state["meta"] = meta
        state["async_engine"] = AsyncEngine(engine)
        await state["async_engine"].start()
        try:
            yield
        finally:
            await state["async_engine"].stop()

    app = FastAPI(title="Nanoserve", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict:
        eng = state["async_engine"].engine
        s = eng.stats()
        return {
            "status": "ok",
            "running": s.num_running,
            "waiting": s.num_waiting,
            "kv_blocks_used": s.kv_blocks_used,
            "kv_blocks_total": s.kv_blocks_total,
            "kv_utilization": round(s.kv_utilization, 4),
            "prefix_hit_rate": round(s.prefix_hit_rate, 4),
            "preemptions": eng.preemptions,
        }

    @app.get("/metrics")
    async def metrics() -> dict:
        eng = state["async_engine"].engine
        s = eng.stats()
        out = {
            "engine": {
                "running": s.num_running,
                "waiting": s.num_waiting,
                "finished": s.num_finished,
                "preemptions": eng.preemptions,
            },
            "kv": {
                "blocks_used": s.kv_blocks_used,
                "blocks_total": s.kv_blocks_total,
                "utilization": round(s.kv_utilization, 4),
            },
            "prefix_cache": eng.prefix_cache.stats.to_dict(eng.pool.block_size),
        }
        if eng.admission is not None:
            out["admission"] = eng.admission.stats.to_dict()
            out["admission"]["estimated_tokens_per_s"] = round(
                eng.admission.tokens_per_second, 1
            )
        return out

    @app.post("/generate")
    async def generate(body: GenerateRequest, http: HTTPRequest) -> EventSourceResponse:
        tok: BPETokenizer = state["tok"]
        ae: AsyncEngine = state["async_engine"]

        ids = tok.encode(body.prompt, allowed_special=False)
        if not ids:
            raise HTTPException(status_code=400, detail="prompt encodes to zero tokens")

        now = time.perf_counter()
        req = Request(
            prompt_token_ids=ids,
            params=SamplingParams(
                max_tokens=body.max_tokens, temperature=body.temperature,
                top_k=body.top_k, top_p=body.top_p, ignore_eos=body.ignore_eos,
            ),
            arrival_time=now,
            deadline=None if body.slo_seconds is None else now + body.slo_seconds,
        )
        stream = await ae.submit(req)

        if req.finish_reason is FinishReason.REJECTED:
            ae.release(req.request_id)
            # 429 with a reason, immediately. That is the entire point of
            # admission control: the client can retry, shed or degrade, and
            # none of those are options once it is already waiting.
            raise HTTPException(
                status_code=429,
                detail={"error": "rejected", "reason": req.rejection_reason},
            )

        return EventSourceResponse(
            _token_stream(ae, stream, tok, http),
            ping=15,
        )

    @app.post("/cancel/{request_id}")
    async def cancel(request_id: str) -> dict:
        cancelled = await state["async_engine"].cancel(request_id)
        if not cancelled:
            raise HTTPException(status_code=404, detail="unknown or finished request")
        return {"cancelled": request_id}

    return app


async def _token_stream(
    ae: AsyncEngine, stream: Stream, tok: BPETokenizer, http: HTTPRequest
) -> AsyncIterator[dict]:
    """Yield SSE events, and guarantee the sequence is released on every exit.

    Decoding runs on the accumulated byte buffer rather than per token: a
    multi-byte character is routinely split across two tokens, and per-token
    decoding would emit replacement characters mid-word.
    """
    req = stream.request
    produced: list[int] = []
    emitted_chars = 0

    try:
        yield {"event": "start", "data": json.dumps({"request_id": req.request_id})}

        while True:
            # FastAPI does not tell a generator its client has gone; it just
            # stops consuming. Polling is the documented way to find out, and
            # without it an abandoned generation keeps its KV forever.
            if await http.is_disconnected():
                break

            try:
                token_id = await asyncio.wait_for(stream.queue.get(), timeout=0.25)
            except asyncio.TimeoutError:
                if stream.done.is_set() and stream.queue.empty():
                    break
                continue

            produced.append(token_id)
            text = tok.decode(produced)
            if text.endswith("�"):
                continue  # half of a multi-byte character; wait for the rest
            if len(text) > emitted_chars:
                yield {"event": "token", "data": json.dumps({"text": text[emitted_chars:]})}
                emitted_chars = len(text)

        reason = (stream.finish_reason or FinishReason.CANCELLED).value
        yield {
            "event": "done",
            "data": json.dumps(
                {"request_id": req.request_id, "finish_reason": reason,
                 "output_tokens": len(produced)}
            ),
        }
    finally:
        # Runs on every exit: normal completion, client disconnect, and
        # asyncio.CancelledError, which is what a cancelled task actually
        # sees. Cancelling an already-finished request is a no-op, so this is
        # safe to call unconditionally -- and calling it unconditionally is
        # what makes the leak impossible rather than unlikely.
        await ae.cancel(req.request_id)
        ae.release(req.request_id)
