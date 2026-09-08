"""Resource safety and reporting honesty.

Every test here exists because an audit found the bug it now pins. None of them
were caught by the correctness suite, because none of these failures produce a
wrong token -- they produce a server that quietly degrades, or a results file
that disagrees with the numbers someone read off the screen.

The KV leak is the most serious. A pool that is not released on the failure
path drains one request at a time: throughput decays, then everything stops,
and nothing ever raises to say why. It is the canonical serving bug and the
correctness tests could not see it, because every one of them takes the happy
path.
"""

from __future__ import annotations

import pytest
import torch

from bench.metrics import aggregate
from engine.baseline import BaselineEngine
from engine.block_manager import PagedCachePool
from engine.kv_cache import StaticCachePool
from engine.types import FinishReason, Request, SamplingParams
from model.config import NanoConfig
from model.transformer import NanoForCausalLM

CFG = NanoConfig(
    vocab_size=256, hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
    num_key_value_heads=2, intermediate_size=176, max_position_embeddings=512,
)


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return NanoForCausalLM(CFG).eval()


def req(prompt_len: int, out_len: int) -> Request:
    return Request(
        prompt_token_ids=list(range(5, 5 + prompt_len)),
        params=SamplingParams(max_tokens=out_len, ignore_eos=True),
    )


def paged(num_blocks=8):
    return PagedCachePool(CFG, num_blocks=num_blocks, block_size=16,
                          device="cpu", dtype=torch.float32)


def static(num_slots=2, max_seq_len=64):
    return StaticCachePool(CFG, num_slots=num_slots, max_seq_len=max_seq_len,
                           device="cpu", dtype=torch.float32)


# ---------------------------------------------------------------------------
# KV must be released on every path
# ---------------------------------------------------------------------------


def test_paged_blocks_are_released_when_generation_fails(model):
    """A request too large for the pool must not strand its blocks."""
    pool = paged(num_blocks=8)  # 128 tokens
    engine = BaselineEngine(model, eos_token_id=255, device="cpu", pool=pool)
    engine.add_request(req(20, 200))  # needs ~14 blocks

    (done,) = engine.step()

    assert done.finish_reason is FinishReason.PREEMPTED
    assert done.output_len > 0, "should keep the tokens it managed to produce"
    assert pool.allocator.num_allocated == 0, "blocks leaked on the failure path"
    assert all(r == 0 for r in pool.allocator._refs)


def test_static_slot_is_released_when_generation_fails(model):
    pool = static(num_slots=2, max_seq_len=64)
    engine = BaselineEngine(model, eos_token_id=255, device="cpu", pool=pool)
    engine.add_request(req(20, 200))

    with pytest.raises(RuntimeError, match="exceeded slot capacity"):
        engine.step()

    assert pool.stats().slots_allocated == 0, "slot leaked on the failure path"


def test_pool_still_serves_after_a_failure(model):
    """The real symptom of a leak: the *next* request suffers.

    A single stranded allocation is invisible. What matters is that the pool
    is undamaged, so this asserts a later request completes normally.
    """
    pool = paged(num_blocks=8)
    engine = BaselineEngine(model, eos_token_id=255, device="cpu", pool=pool)

    engine.add_request(req(20, 200))  # will fail
    engine.step()
    engine.add_request(req(10, 20))  # must succeed
    (done,) = engine.step()

    assert done.finish_reason is FinishReason.LENGTH
    assert done.output_len == 20
    assert pool.allocator.num_allocated == 0


def test_repeated_failures_do_not_accumulate(model):
    """Ten failures in a row must leave the pool exactly as it started."""
    pool = paged(num_blocks=8)
    engine = BaselineEngine(model, eos_token_id=255, device="cpu", pool=pool)
    free_before = pool.allocator.num_free

    for _ in range(10):
        engine.add_request(req(20, 200))
        engine.step()

    assert pool.allocator.num_free == free_before


def test_normal_completion_also_releases(model):
    pool = paged(num_blocks=64)
    engine = BaselineEngine(model, eos_token_id=255, device="cpu", pool=pool)
    for _ in range(5):
        engine.add_request(req(30, 40))
        engine.step()
    assert pool.allocator.num_allocated == 0


# ---------------------------------------------------------------------------
# The allocator contract
# ---------------------------------------------------------------------------


def test_paged_allocate_reports_a_full_pool(model):
    """None, not a doomed cache.

    A pool with no free blocks cannot make progress. Returning a cache anyway
    defers the failure to an OutOfBlocks several layers into a forward pass,
    and gives the scheduler two different ways to learn the same fact.
    """
    pool = paged(num_blocks=1)
    first = pool.allocate()
    assert first is not None
    first._ensure_capacity(16)  # consumes the only block

    assert pool.allocate() is None
    first.free()
    assert pool.allocate() is not None


def test_can_fit_answers_before_committing():
    """Admission control needs to decide up front, not discover mid-flight."""
    pool = paged(num_blocks=4)  # 64 tokens
    assert pool.can_fit(64) is True
    assert pool.can_fit(65) is False
    assert pool.can_fit(1) is True

    c = pool.allocate()
    c._ensure_capacity(48)  # takes 3 blocks
    assert pool.can_fit(16) is True
    assert pool.can_fit(17) is False


def test_both_pools_share_the_full_pool_contract(model):
    """Static and paged must answer 'is there room?' the same way."""
    for pool in (paged(num_blocks=1), static(num_slots=1, max_seq_len=32)):
        first = pool.allocate(expected_len=16)
        assert first is not None
        if isinstance(pool, PagedCachePool):
            first._ensure_capacity(16)
        assert pool.allocate(expected_len=16) is None


# ---------------------------------------------------------------------------
# Reporting honesty
# ---------------------------------------------------------------------------


def _fake_summary(throughput: float, tag: str) -> dict:
    return {
        "engine": "fake",
        "wall_seconds": 1.0,
        "tag": tag,
        "throughput": {"output_tokens_per_s": throughput, "requests_per_s": throughput / 10},
        "goodput": {"requests_per_s": throughput / 10},
        "latency_s": {
            "ttft": {"p50": 1.0, "p99": 2.0},
            "inter_token": {"p50": 0.01, "p99": 0.02},
        },
    }


def test_representative_run_is_chosen_by_value_not_by_order():
    """The middle repeat is not the median repeat.

    Reporting `summaries[len//2]` picked whichever run happened to execute
    second, which for [100, 300, 200] is the 300 -- the best run presented as
    typical. Selection is now by throughput.
    """
    runs = [_fake_summary(v, f"run{i}") for i, v in enumerate([100.0, 300.0, 200.0])]
    agg = aggregate(runs)

    assert agg["representative"]["tag"] == "run2"
    assert agg["representative"]["throughput"]["output_tokens_per_s"] == 200.0
    assert agg["across_repeats"]["output_tokens_per_s"]["median"] == 200.0
    # The headline and the spread must describe the same run.
    assert (
        agg["representative"]["throughput"]["output_tokens_per_s"]
        == agg["across_repeats"]["output_tokens_per_s"]["median"]
    )


def test_single_run_is_its_own_representative():
    agg = aggregate([_fake_summary(42.0, "only")])
    assert agg["repeats"] == 1
    assert agg["representative"]["tag"] == "only"
    assert agg["stable"] is True


def test_instability_is_flagged():
    """A run whose throughput swings must not be silently compared."""
    steady = aggregate([_fake_summary(v, f"r{i}") for i, v in enumerate([100.0, 101.0, 99.5])])
    swinging = aggregate([_fake_summary(v, f"r{i}") for i, v in enumerate([100.0, 180.0, 60.0])])

    assert steady["stable"] is True
    assert swinging["stable"] is False
    assert swinging["worst_cv"] > 0.05
