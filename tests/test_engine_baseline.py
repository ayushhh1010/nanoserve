"""The baseline engine and the benchmark harness.

Two things are being pinned here.

**The cache must not change the answer.** `use_cache=True` and
`use_cache=False` are mathematically identical and must produce byte-identical
token sequences. This is the same invariant every later engine has to satisfy --
the paged allocator, the continuous-batching scheduler and preemption all have
to reproduce this output token for token -- so it is worth having the assertion
written once, here, against the simplest possible implementation.

**The harness must measure what it claims.** A benchmark that reports a number
nobody checked is worse than no benchmark, because it carries authority. TTFT,
inter-token latency and queue time are asserted against hand-constructed
requests with known timestamps rather than trusted.
"""

from __future__ import annotations

import time

import pytest
import torch

from bench.metrics import RunResult, percentiles
from bench.workload import WorkloadConfig, build_workload, summarise
from engine.baseline import BaselineEngine
from engine.types import FinishReason, Request, SamplingParams
from model.config import NanoConfig
from model.transformer import NanoForCausalLM

CFG = NanoConfig(
    vocab_size=256,
    hidden_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    intermediate_size=176,
    max_position_embeddings=256,
)
EOS = 255


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return NanoForCausalLM(CFG).eval()


def engine(model, **kw):
    return BaselineEngine(model, eos_token_id=EOS, device="cpu", **kw)


def req(prompt, max_tokens=8, **kw):
    return Request(
        prompt_token_ids=list(prompt),
        params=SamplingParams(max_tokens=max_tokens, ignore_eos=True, **kw),
    )


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------


def test_cache_does_not_change_the_output(model):
    """The invariant every later engine inherits.

    A KV cache is an optimisation, not a modelling decision. If caching changes
    a single token, something is wrong with the mask, the positions, or the
    cache itself -- and it will be wrong in the paged version too.
    """
    prompt = list(range(10, 34))

    cached = engine(model, use_cache=True).run([req(prompt, 24)])[0]
    uncached = engine(model, use_cache=False).run([req(prompt, 24)])[0]

    assert cached.output_token_ids == uncached.output_token_ids
    assert len(cached.output_token_ids) == 24


def test_engine_matches_the_reference_generator(model):
    """The baseline must reproduce model/generate.py exactly.

    model/generate.py is the correctness reference for the whole phase. If the
    engine and the reference ever disagree, every downstream "matches the
    baseline" claim is measuring agreement with the wrong thing.
    """
    from model.generate import SamplingParams as GenParams
    from model.transformer import DynamicCache

    prompt = list(range(5, 29))
    got = engine(model).run([req(prompt, 20)])[0].output_token_ids

    # The reference, inlined: greedy, one token at a time, contiguous cache.
    with torch.inference_mode():
        cache = DynamicCache()
        logits, _ = model(torch.tensor([prompt]), cache=cache)
        want = []
        for _ in range(20):
            nxt = logits[:, -1].argmax(-1, keepdim=True)
            want.append(int(nxt))
            logits, _ = model(nxt, cache=cache)

    assert got == want
    assert GenParams().greedy is False  # reference defaults to sampling


def test_max_tokens_is_respected(model):
    for n in (1, 5, 17):
        out = engine(model).run([req(range(4, 20), n)])[0]
        assert out.output_len == n
        assert out.finish_reason is FinishReason.LENGTH


def _force_token(e, token_id, after=0):
    """Make the engine emit `token_id` from call `after` onward.

    Patching the sampler rather than the weights. Boosting `lm_head.weight[EOS]`
    does not reliably make EOS the argmax here: embeddings are tied, so that row
    is also token 255's input embedding, and the resulting logit shift depends
    on the sign of the hidden state.
    """
    calls = {"n": 0}
    real = e._sample

    def patched(logits, req):
        calls["n"] += 1
        if calls["n"] > after:
            return torch.full_like(real(logits, req), token_id)
        return real(logits, req)

    e._sample = patched
    return calls


def test_eos_is_a_stop_signal_not_an_output_token(model):
    e = engine(model)
    _force_token(e, EOS, after=3)
    r = Request(
        prompt_token_ids=list(range(4, 20)),
        params=SamplingParams(max_tokens=50, ignore_eos=False),
    )
    out = e.run([r])[0]

    assert out.finish_reason is FinishReason.EOS
    assert out.output_len == 3, "should stop on the 4th token, having emitted 3"
    assert EOS not in out.output_token_ids


def test_ignore_eos_runs_to_length(model):
    """Fixed-length generation: EOS is emitted as an ordinary token.

    Benchmarks need every request to produce exactly the tokens the workload
    asked for, or engine-vs-engine comparisons pick up noise from wherever the
    weights happen to place EOS.
    """
    e = engine(model)
    _force_token(e, EOS, after=0)
    out = e.run([req(range(4, 20), 12)])[0]

    assert out.output_len == 12
    assert out.finish_reason is FinishReason.LENGTH
    assert all(t == EOS for t in out.output_token_ids)


def test_sampling_is_reproducible_and_differs_from_greedy(model):
    torch.manual_seed(7)
    a = engine(model).run([req(range(4, 20), 16, temperature=0.9, top_k=20)])[0]
    torch.manual_seed(7)
    b = engine(model).run([req(range(4, 20), 16, temperature=0.9, top_k=20)])[0]
    greedy = engine(model).run([req(range(4, 20), 16)])[0]

    assert a.output_token_ids == b.output_token_ids
    assert a.output_token_ids != greedy.output_token_ids


# ---------------------------------------------------------------------------
# Head-of-line blocking -- the thing this engine exists to demonstrate
# ---------------------------------------------------------------------------


def test_short_request_waits_for_the_long_one_ahead_of_it(model):
    """The failure continuous batching exists to fix, measured.

    A 4-token request queued behind a 120-token one waits for all 120. Its
    queue time is somebody else's generation length -- work it has nothing to
    do with.
    """
    e = engine(model)
    long_req = req(range(4, 20), 120)
    short_req = req(range(4, 20), 4)

    now = time.perf_counter()
    for r in (long_req, short_req):
        r.arrival_time = now

    e.add_request(long_req)
    e.add_request(short_req)
    e.step()  # drains the long one entirely
    e.step()

    assert short_req.queue_time > long_req.e2e_latency * 0.9, (
        "short request did not wait behind the long one -- is the engine "
        "interleaving? it should not be"
    )
    # And its own work was trivial by comparison.
    assert short_req.queue_time > sum(short_req.inter_token_latencies) * 3


def test_engine_reports_queue_depth(model):
    e = engine(model)
    assert e.stats().num_waiting == 0
    for _ in range(5):
        e.add_request(req(range(4, 20), 2))
    assert e.stats().num_waiting == 5
    e.step()
    assert e.stats().num_waiting == 4
    assert e.stats().num_finished == 1


# ---------------------------------------------------------------------------
# Request accounting
# ---------------------------------------------------------------------------


def test_latency_fields_are_computed_from_the_right_endpoints():
    """TTFT is measured from arrival, not from scheduling.

    Queue delay is latency the user experiences. An engine that hides a growing
    queue behind a fast prefill has not made anything faster, and measuring
    TTFT from the scheduling instant would report exactly that illusion.
    """
    r = Request(prompt_token_ids=[1, 2, 3])
    r.arrival_time = 100.0
    r.scheduled_time = 105.0
    r.record_token(7, now=106.0)
    r.record_token(8, now=106.5)
    r.record_token(9, now=107.25)
    r.finish(FinishReason.LENGTH, now=107.25)

    assert r.ttft == pytest.approx(6.0)  # arrival -> first token
    assert r.queue_time == pytest.approx(5.0)
    assert r.e2e_latency == pytest.approx(7.25)
    # Gaps between output tokens only; the pre-first-token gap is TTFT.
    assert r.inter_token_latencies == pytest.approx([0.5, 0.75])


def test_deadline_predicate():
    def make(finish_at, deadline, reason=FinishReason.LENGTH):
        r = Request(prompt_token_ids=[1])
        r.arrival_time, r.deadline = 0.0, deadline
        r.finish(reason, now=finish_at)
        return r

    assert make(5.0, 10.0).met_deadline
    assert not make(15.0, 10.0).met_deadline
    assert make(15.0, None).met_deadline  # best-effort
    # A rejected request never counts toward goodput, deadline or not.
    assert not make(1.0, 10.0, FinishReason.REJECTED).met_deadline


# ---------------------------------------------------------------------------
# Workload generation
# ---------------------------------------------------------------------------


def test_workload_is_reproducible():
    cfg = WorkloadConfig(num_requests=60, seed=42)
    a = build_workload(cfg, 256, EOS)
    b = build_workload(cfg, 256, EOS)
    assert [r.prompt_token_ids for r in a] == [r.prompt_token_ids for r in b]
    assert [r.params.max_tokens for r in a] == [r.params.max_tokens for r in b]
    assert [r.arrival_time for r in a] == [r.arrival_time for r in b]


def test_arrival_rate_matches_the_request():
    """Poisson arrivals: inter-arrival gaps are exponential with mean 1/rate."""
    cfg = WorkloadConfig(num_requests=4000, request_rate=8.0, seed=1)
    reqs = build_workload(cfg, 256, EOS)
    gaps = [b.arrival_time - a.arrival_time for a, b in zip(reqs, reqs[1:])]

    assert sum(gaps) / len(gaps) == pytest.approx(1 / 8.0, rel=0.08)
    # Exponential, not constant: the standard deviation of an exponential
    # equals its mean. A constant-rate generator would give ~0.
    import numpy as np

    assert np.std(gaps) == pytest.approx(1 / 8.0, rel=0.15)


def test_burst_mode_has_every_request_arrive_at_zero():
    reqs = build_workload(WorkloadConfig(num_requests=50, request_rate=None), 256, EOS)
    assert all(r.arrival_time == 0.0 for r in reqs)


def test_output_lengths_vary():
    """Without length variance, head-of-line blocking cannot be observed."""
    reqs = build_workload(WorkloadConfig(num_requests=500, seed=3), 256, EOS)
    lens = [r.params.max_tokens for r in reqs]
    assert len(set(lens)) > 50
    assert max(lens) > 3 * min(lens)


def test_prefixes_are_shared_and_zipfian():
    """A handful of prefixes should dominate, or prefix caching is untestable."""
    cfg = WorkloadConfig(num_requests=1200, num_prefixes=16, zipf_alpha=1.0,
                         prefix_fraction=1.0, prefix_len=32, seed=5)
    reqs = build_workload(cfg, 256, EOS)

    from collections import Counter

    heads = Counter(tuple(r.prompt_token_ids[:32]) for r in reqs)
    assert len(heads) <= cfg.num_prefixes
    # Zipf with alpha=1 over 16 items puts ~1/3 of mass on the first.
    most_common = heads.most_common(1)[0][1] / len(reqs)
    assert most_common > 0.20, f"top prefix only {most_common:.1%} -- not heavy-tailed"


def test_deadlines_track_arrivals():
    cfg = WorkloadConfig(num_requests=40, request_rate=5.0, slo_seconds=2.5, seed=0)
    for r in build_workload(cfg, 256, EOS):
        assert r.deadline == pytest.approx(r.arrival_time + 2.5)


def test_summarise_reports_what_ran():
    reqs = build_workload(WorkloadConfig(num_requests=100, seed=0), 256, EOS)
    s = summarise(reqs)
    assert s["num_requests"] == 100
    assert s["output_tokens"] == sum(r.params.max_tokens for r in reqs)
    assert s["prompt_len"]["p99"] >= s["prompt_len"]["p50"]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_percentiles_are_ordered_and_complete():
    p = percentiles([float(i) for i in range(1, 101)])
    assert p["p50"] == pytest.approx(50.5)
    assert p["p50"] < p["p90"] < p["p99"] <= p["max"]
    assert p["n"] == 100


def test_percentiles_handle_empty_input():
    p = percentiles([])
    assert p["n"] == 0 and p["p50"] is None and p["mean"] is None


def test_summary_separates_throughput_from_goodput(model):
    """Goodput counts only work that was still useful when it finished."""
    done, late = req([1, 2, 3], 10), req([1, 2, 3], 10)
    for r, finish in ((done, 1.0), (late, 99.0)):
        r.arrival_time, r.deadline = 0.0, 5.0
        for i in range(10):
            r.record_token(i, now=finish)
        r.finish(FinishReason.LENGTH, now=finish)

    s = RunResult("test", 100.0, [done, late], {}, {}).summary()
    assert s["counts"]["completed"] == 2
    assert s["throughput"]["output_tokens_per_s"] == pytest.approx(0.2)
    assert s["goodput"]["fraction_met_slo"] == pytest.approx(0.5)
