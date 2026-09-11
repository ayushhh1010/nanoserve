"""The HTTP layer: SSE streaming, cancellation, and the leak it prevents.

`test_disconnect_releases_kv_blocks` is the reason this file exists. A client
that hangs up mid-generation is the single most common way a serving system
leaks memory, because nothing about it looks like an error: the request simply
stops being read, the sequence keeps generating, and its KV blocks are never
returned. The pool drains one abandoned request at a time until throughput
reaches zero, and no exception is ever raised.

It is also invisible to every test that only exercises requests which finish,
which is why it gets an explicit one: record the free-block count, start a
generation, hang up, and require the count to come back.

The engine runs on CPU with a tiny model throughout -- these tests are about
the transport and the resource accounting, not the arithmetic.
"""

from __future__ import annotations

import asyncio

import pytest
import torch

from engine.admission import AdmissionConfig
from engine.block_manager import PagedCachePool
from engine.scheduler import ContinuousBatchingEngine
from engine.server import AsyncEngine
from engine.types import FinishReason, Request, SamplingParams
from model.config import NanoConfig
from model.transformer import NanoForCausalLM

CFG = NanoConfig(
    vocab_size=256, hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
    num_key_value_heads=2, intermediate_size=176, max_position_embeddings=512,
)
EOS = 255

pytestmark = pytest.mark.asyncio


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return NanoForCausalLM(CFG).eval()


def make_engine(model, num_blocks=128, **kw):
    pool = PagedCachePool(CFG, num_blocks=num_blocks, block_size=16,
                          device="cpu", dtype=torch.float32)
    return ContinuousBatchingEngine(
        model, pool, eos_token_id=EOS, device="cpu",
        max_batch_size=kw.pop("max_batch_size", 16),
        reserve_blocks=kw.pop("reserve_blocks", 4), **kw,
    )


def req(max_tokens=32, prompt_len=16, rid=None):
    r = Request(
        prompt_token_ids=[7] * prompt_len,
        params=SamplingParams(max_tokens=max_tokens, ignore_eos=True),
    )
    if rid:
        r.request_id = rid
    return r


@pytest.fixture
async def running(model):
    engine = make_engine(model)
    ae = AsyncEngine(engine, idle_sleep=0.001)
    await ae.start()
    try:
        yield ae
    finally:
        await ae.stop()


# ---------------------------------------------------------------------------
# The leak
# ---------------------------------------------------------------------------


async def test_cancel_releases_kv_blocks(running):
    """Cancelling mid-generation must return every block.

    The free-block count is recorded before and required to come back after.
    A leak here does not raise, does not corrupt output, and does not show up
    until the pool is empty.
    """
    engine = running.engine
    free_before = engine.pool.allocator.num_free

    r = req(max_tokens=500, rid="long")
    await running.submit(r)
    for _ in range(200):  # let it actually start generating
        await asyncio.sleep(0.005)
        if r.output_len > 3:
            break
    assert r.output_len > 0, "request never started"
    assert engine.pool.allocator.num_free < free_before, "no blocks were taken"

    assert await running.cancel("long") is True
    assert r.finish_reason is FinishReason.CANCELLED

    # The prefix cache legitimately keeps the prompt's full blocks -- that is
    # the feature, not a leak. Everything the *generation* held must be back.
    held_by_cache = len(engine.prefix_cache._entries)
    assert engine.pool.allocator.num_free == free_before - held_by_cache, (
        "generation blocks leaked on cancel"
    )
    engine.prefix_cache.clear()
    assert engine.pool.allocator.num_free == free_before, "blocks leaked on cancel"
    assert all(rc == 0 for rc in engine.pool.allocator._refs)


async def test_cancelling_a_queued_request_also_works(running):
    """A client can hang up before its request ever starts."""
    engine = running.engine
    engine.max_batch_size = 1

    first = req(max_tokens=400, rid="first")
    queued = req(max_tokens=10, rid="queued")
    await running.submit(first)
    await asyncio.sleep(0.05)
    await running.submit(queued)

    assert await running.cancel("queued") is True
    assert queued.finish_reason is FinishReason.CANCELLED
    assert all(s.request.request_id != "queued" for s in engine.running)
    assert await running.cancel("first") is True


async def test_repeated_cancellation_is_safe(running):
    """`finally` blocks cancel unconditionally, so this happens constantly."""
    r = req(max_tokens=200, rid="x")
    await running.submit(r)
    await asyncio.sleep(0.03)

    first = await running.cancel("x")
    assert first is True
    assert await running.cancel("x") is False  # already gone, not an error
    assert await running.cancel("never-existed") is False


async def test_many_cancellations_leave_the_pool_intact(running):
    """Twenty abandoned requests in a row must not accumulate."""
    engine = running.engine
    free_before = engine.pool.allocator.num_free

    for i in range(20):
        r = req(max_tokens=300, rid=f"abandon-{i}")
        await running.submit(r)
        await asyncio.sleep(0.01)
        await running.cancel(f"abandon-{i}")

    # Cached prefixes legitimately hold blocks; clear them and require the
    # pool to be exactly as it started.
    engine.prefix_cache.clear()
    assert engine.pool.allocator.num_free == free_before
    assert all(r == 0 for r in engine.pool.allocator._refs)


async def test_cancelled_request_stops_producing(running):
    r = req(max_tokens=500, rid="stop")
    await running.submit(r)
    await asyncio.sleep(0.05)

    await running.cancel("stop")
    after_cancel = r.output_len
    await asyncio.sleep(0.1)
    assert r.output_len == after_cancel, "cancelled request kept generating"


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


async def test_tokens_are_delivered_as_they_are_produced(running):
    r = req(max_tokens=12, rid="stream")
    stream = await running.submit(r)

    got = []
    for _ in range(400):
        try:
            got.append(await asyncio.wait_for(stream.queue.get(), timeout=0.02))
        except asyncio.TimeoutError:
            if stream.done.is_set() and stream.queue.empty():
                break
    assert len(got) == 12
    assert got == r.output_token_ids


async def test_stream_signals_completion(running):
    r = req(max_tokens=6, rid="finish")
    stream = await running.submit(r)
    for _ in range(400):
        if stream.done.is_set():
            break
        await asyncio.sleep(0.005)
    assert stream.done.is_set()
    assert stream.finish_reason is FinishReason.LENGTH


async def test_concurrent_streams_do_not_cross(running):
    """Each connection must see only its own tokens."""
    reqs = [req(max_tokens=10, prompt_len=8 + i, rid=f"c{i}") for i in range(5)]
    streams = [await running.submit(r) for r in reqs]

    for _ in range(1500):
        if all(s.done.is_set() for s in streams):
            break
        await asyncio.sleep(0.005)

    for r, s in zip(reqs, streams):
        drained = []
        while not s.queue.empty():
            drained.append(s.queue.get_nowait())
        assert drained == r.output_token_ids, f"{r.request_id} received wrong tokens"


async def test_streamed_tokens_match_a_synchronous_run(model, running):
    """The transport must not change the result."""
    sync_engine = make_engine(model)
    want = sync_engine.run([req(max_tokens=14, rid="ref")])[0].output_token_ids

    r = req(max_tokens=14, rid="via-server")
    stream = await running.submit(r)
    for _ in range(1500):
        if stream.done.is_set():
            break
        await asyncio.sleep(0.005)
    assert r.output_token_ids == want


# ---------------------------------------------------------------------------
# Admission through the async layer
# ---------------------------------------------------------------------------


async def test_rejection_is_visible_immediately(model):
    engine = make_engine(
        model, admission=AdmissionConfig(max_queue_depth=1, enforce_deadlines=False)
    )
    engine.max_batch_size = 1
    ae = AsyncEngine(engine, idle_sleep=0.001)
    await ae.start()
    try:
        outcomes = [ (await ae.submit(req(max_tokens=200, rid=f"r{i}"))).request
                     for i in range(6) ]
        rejected = [r for r in outcomes if r.finish_reason is FinishReason.REJECTED]
        assert rejected, "queue cap never engaged"
        for r in rejected:
            assert r.rejection_reason == "queue_full"
            assert r.output_len == 0
    finally:
        await ae.stop()


async def test_engine_loop_survives_an_idle_period(running):
    """The background task must keep running when there is nothing to do."""
    await asyncio.sleep(0.05)
    assert running._task is not None and not running._task.done()

    r = req(max_tokens=4, rid="after-idle")
    stream = await running.submit(r)
    for _ in range(600):
        if stream.done.is_set():
            break
        await asyncio.sleep(0.005)
    assert r.output_len == 4


async def test_stop_is_clean(model):
    engine = make_engine(model)
    ae = AsyncEngine(engine, idle_sleep=0.001)
    await ae.start()
    await ae.submit(req(max_tokens=8, rid="inflight"))
    await asyncio.sleep(0.02)
    await ae.stop()
    assert ae._task is None
