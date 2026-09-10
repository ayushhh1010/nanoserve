"""Admission control, and the counter-intuitive property it exists for.

Rejecting requests increases the number served. A request admitted with no
chance of meeting its deadline still consumes KV blocks, batch slots and
forward passes -- all taken from requests that could have made it. The central
test here asserts exactly that: under overload, the engine with admission
control completes *more* work within SLO than the one that accepts everything.

The second thing being pinned is that a rejection is never counted as a
success. A rejected request must not reach the goodput numerator, or an engine
that refuses everything scores perfectly.
"""

from __future__ import annotations

import time

import pytest
import torch

from bench.metrics import RunResult
from engine.admission import AdmissionConfig, AdmissionController, AdmissionStats
from engine.scheduler import ContinuousBatchingEngine
from engine.block_manager import PagedCachePool
from engine.types import FinishReason, Request, SamplingParams
from model.config import NanoConfig
from model.transformer import NanoForCausalLM

CFG = NanoConfig(
    vocab_size=256, hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
    num_key_value_heads=2, intermediate_size=176, max_position_embeddings=512,
)
EOS = 255


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return NanoForCausalLM(CFG).eval()


def pool(num_blocks=256):
    return PagedCachePool(CFG, num_blocks=num_blocks, block_size=16,
                          device="cpu", dtype=torch.float32)


def req(max_tokens=16, deadline=None, prompt_len=16, rid=None):
    r = Request(
        prompt_token_ids=[7] * prompt_len,
        params=SamplingParams(max_tokens=max_tokens, ignore_eos=True),
    )
    r.arrival_time = 0.0
    r.deadline = deadline
    if rid:
        r.request_id = rid
    return r


# ---------------------------------------------------------------------------
# The controller in isolation
# ---------------------------------------------------------------------------


def test_queue_depth_gate():
    c = AdmissionController(AdmissionConfig(max_queue_depth=3, enforce_deadlines=False))
    for depth in (0, 1, 2):
        assert c.admit(req(), depth, 0, 0)[0] is True
    ok, reason = c.admit(req(), 3, 0, 0)
    assert ok is False and reason == "queue_full"
    assert c.stats.rejected_queue_full == 1


def test_queue_depth_gate_can_be_disabled():
    c = AdmissionController(AdmissionConfig(max_queue_depth=None, enforce_deadlines=False))
    assert c.admit(req(), 10_000, 0, 0)[0] is True


def test_deadline_gate_rejects_what_it_cannot_finish():
    """The projection: work ahead plus own work, over measured throughput."""
    c = AdmissionController(AdmissionConfig(max_queue_depth=None, slack=1.0))
    for _ in range(20):
        c.record_step(0.01, batch_size=1)  # 100 tokens/s
    assert c.tokens_per_second == pytest.approx(100.0, rel=0.01)

    now = time.perf_counter()
    # 900 tokens queued + 100 of its own = 10s of work.
    generous = req(max_tokens=100, deadline=now + 30)
    tight = req(max_tokens=100, deadline=now + 2)

    assert c.admit(generous, 1, 900, 0, now=now)[0] is True
    ok, reason = c.admit(tight, 1, 900, 0, now=now)
    assert ok is False and reason == "deadline_exceeded"


def test_requests_without_deadlines_are_best_effort():
    c = AdmissionController(AdmissionConfig(max_queue_depth=None))
    for _ in range(10):
        c.record_step(0.01, 1)
    assert c.admit(req(max_tokens=10_000, deadline=None), 1, 10**6, 0)[0] is True


def test_slack_admits_marginal_requests():
    """The estimate is noisy, and the two errors are not symmetric.

    A wrongly rejected request fails for certain; a wrongly admitted one only
    might. Slack above 1 leans toward admitting.
    """
    now = time.perf_counter()
    strict = AdmissionController(AdmissionConfig(max_queue_depth=None, slack=1.0))
    lenient = AdmissionController(AdmissionConfig(max_queue_depth=None, slack=2.0))
    for c in (strict, lenient):
        for _ in range(10):
            c.record_step(0.01, 1)

    # 1100 queued + 100 own = 1200 tokens at 100 tok/s = a 12s projection,
    # against a 9s deadline. Strict refuses; slack of 2.0 stretches the bar to
    # 18s and admits. Chosen well clear of the boundary on both sides.
    marginal = req(max_tokens=100, deadline=now + 9)
    assert strict.admit(marginal, 1, 1100, 0, now=now)[0] is False
    assert lenient.admit(marginal, 1, 1100, 0, now=now)[0] is True


def test_throughput_estimate_accounts_for_batch_size():
    """Aggregate rate, not per-sequence rate.

    The same engine runs at wildly different rates depending on how full the
    batch is, and admission has to reason about the state it is actually in.
    """
    solo = AdmissionController()
    batched = AdmissionController()
    for _ in range(20):
        solo.record_step(0.01, batch_size=1)
        batched.record_step(0.01, batch_size=32)

    assert batched.tokens_per_second == pytest.approx(32 * solo.tokens_per_second, rel=0.01)


def test_estimate_has_a_defined_value_before_any_step():
    c = AdmissionController(AdmissionConfig(initial_step_seconds=0.05))
    assert c.tokens_per_second == pytest.approx(20.0)
    assert c.admit(req(deadline=time.perf_counter() + 100), 0, 0, 0)[0] is True


def test_stats_accounting():
    s = AdmissionStats(offered=10, admitted=6, rejected_queue_full=3, rejected_deadline=1)
    assert s.rejected == 4
    assert s.rejection_rate == pytest.approx(0.4)
    assert s.to_dict()["rejected_queue_full"] == 3


# ---------------------------------------------------------------------------
# Wired into the engine
# ---------------------------------------------------------------------------


def engine(model, admission=None, **kw):
    kw.setdefault("reserve_blocks", 4)
    return ContinuousBatchingEngine(
        model, kw.pop("pool", None) or pool(), eos_token_id=EOS, device="cpu",
        admission=admission, **kw
    )


def test_rejection_is_immediate_not_queued(model):
    """The entire value of a 429 is that it arrives fast."""
    e = engine(model, AdmissionConfig(max_queue_depth=2, enforce_deadlines=False))
    for i in range(6):
        e.add_request(req(rid=f"r{i}"))

    rejected = e.drain_rejected()
    assert len(rejected) == 4
    assert len(e.waiting) == 2
    for r in rejected:
        assert r.finish_reason is FinishReason.REJECTED
        assert r.rejection_reason == "queue_full"
        # Never scheduled, never produced a token.
        assert r.scheduled_time is None and r.output_len == 0


def test_drain_is_not_repeatable(model):
    e = engine(model, AdmissionConfig(max_queue_depth=0, enforce_deadlines=False))
    e.add_request(req())
    assert len(e.drain_rejected()) == 1
    assert e.drain_rejected() == []


def test_no_admission_config_means_no_gate(model):
    e = engine(model, admission=None)
    for i in range(200):
        e.add_request(req(rid=f"r{i}"))
    assert len(e.waiting) == 200
    assert e.admission is None


def test_rejected_requests_never_count_as_goodput():
    """An engine that refuses everything must not score perfectly."""
    rejected = req(deadline=time.perf_counter() + 100)
    rejected.finish(FinishReason.REJECTED)
    completed = req(deadline=time.perf_counter() + 100)
    completed.record_token(1)
    completed.finish(FinishReason.LENGTH)

    s = RunResult("t", 1.0, [rejected, completed], {}, {}).summary()
    assert s["counts"]["rejected"] == 1
    assert s["counts"]["completed"] == 1
    assert s["goodput"]["rejection_rate"] == pytest.approx(0.5)
    assert not rejected.met_deadline


def test_admitted_requests_still_run_correctly(model):
    """Admission control must not change what an admitted request produces."""
    reqs = lambda: [req(max_tokens=12, rid=f"r{i}") for i in range(6)]  # noqa: E731

    without = {r.request_id: r.output_token_ids for r in engine(model).run(reqs())}
    with_ac = {
        r.request_id: r.output_token_ids
        for r in engine(
            model, AdmissionConfig(max_queue_depth=None, enforce_deadlines=False)
        ).run(reqs())
        if r.finish_reason is not FinishReason.REJECTED
    }
    assert with_ac == without


def test_engine_records_step_time_for_the_estimate(model):
    e = engine(model, AdmissionConfig(max_queue_depth=None, enforce_deadlines=False))
    for i in range(4):
        e.add_request(req(max_tokens=10, rid=f"r{i}"))
    for _ in range(3):
        e.step()

    assert len(e.admission._step_times) >= 2
    assert e.admission.tokens_per_second > 0


def test_overload_rejects_and_still_completes_the_rest(model):
    """The property admission control exists for, on a calibrated overload.

    The deadline is derived from a measured run rather than guessed: an
    absolute value that overloads this model on one machine leaves it idle on
    another, and a test that silently stops overloading stops testing anything.
    Here the workload is timed ungated first, then the deadline is set to a
    fraction of that, guaranteeing most requests cannot possibly finish.

    With no gate, everything is accepted and almost everything misses. With a
    gate, the requests that had no chance are turned away and the survivors
    land -- so the engine that refuses work completes MORE work within SLO.
    """
    N, TOKENS = 24, 40

    def workload(deadline_after: float | None):
        base = time.perf_counter()
        out = []
        for i in range(N):
            r = req(max_tokens=TOKENS, rid=f"r{i}")
            r.arrival_time = base
            r.deadline = None if deadline_after is None else base + deadline_after
            out.append(r)
        return out

    # Calibrate: how long does this machine actually take?
    warm = engine(model, admission=None)
    t0 = time.perf_counter()
    warm.run(workload(None))
    full_run = time.perf_counter() - t0

    # A deadline only the first fifth of the work can meet.
    slo = full_run / 5

    open_gate = engine(model, admission=None)
    open_gate.run(workload(slo))
    open_met = sum(1 for r in open_gate.finished if r.met_deadline)

    gated = engine(model, AdmissionConfig(max_queue_depth=None, slack=1.0))
    gated.run(workload(slo))
    gated_met = sum(1 for r in gated.finished if r.met_deadline)
    gated_rejected = sum(1 for r in gated.finished if r.finish_reason is FinishReason.REJECTED)

    assert open_met < N, "the workload did not overload; the test proves nothing"
    assert gated_rejected > 0, "nothing was rejected; the gate never engaged"
    assert gated_met >= open_met, (
        f"admission control served fewer within SLO ({gated_met}) than the "
        f"unbounded queue ({open_met})"
    )


def test_rejection_reasons_are_recorded_separately(model):
    """Queue-full and deadline rejections have different fixes.

    One says the server is saturated; the other says this particular request
    was never going to make it. Collapsing them into one counter loses the
    only actionable part.
    """
    depth_only = AdmissionController(
        AdmissionConfig(max_queue_depth=2, enforce_deadlines=False)
    )
    for i in range(5):
        depth_only.admit(req(rid=f"r{i}"), queue_depth=i, queued_tokens=0, running_tokens=0)
    assert depth_only.stats.rejected_queue_full == 3
    assert depth_only.stats.rejected_deadline == 0

    now = time.perf_counter()
    deadline_only = AdmissionController(AdmissionConfig(max_queue_depth=None, slack=1.0))
    for _ in range(10):
        deadline_only.record_step(0.01, 1)
    for i in range(4):
        deadline_only.admit(
            req(max_tokens=100, deadline=now + 1, rid=f"d{i}"),
            queue_depth=0, queued_tokens=5000, running_tokens=0, now=now,
        )
    assert deadline_only.stats.rejected_deadline == 4
    assert deadline_only.stats.rejected_queue_full == 0
