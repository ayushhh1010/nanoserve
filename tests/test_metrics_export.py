"""Prometheus export.

Two properties are worth pinning, because both break silently.

**Counters must only ever go up.** The engine reports totals; the exporter has
to turn them into increments. A Counter that moves backwards makes every
`rate()` query built on it produce garbage, and Prometheus will not complain.

**Rejections must not pollute latency histograms.** A request refused in a
millisecond is not a fast request. Letting it into the TTFT histogram would
make an overloaded server -- which rejects nearly everything -- look like the
fastest one on the dashboard.
"""

from __future__ import annotations

import pytest
import torch

from engine.admission import AdmissionConfig
from engine.block_manager import PagedCachePool
from engine.metrics import SECONDS, SECONDS_FAST, EngineMetrics
from engine.scheduler import ContinuousBatchingEngine
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


def engine(model, **kw):
    pool = PagedCachePool(CFG, num_blocks=256, block_size=16,
                          device="cpu", dtype=torch.float32)
    return ContinuousBatchingEngine(
        model, pool, eos_token_id=EOS, device="cpu", max_batch_size=8,
        reserve_blocks=4, **kw
    )


def finished_request(
    ttft=0.05, itl=0.01, n_tokens=5, reason=FinishReason.LENGTH, rejection=""
) -> Request:
    r = Request(prompt_token_ids=[1, 2, 3],
                params=SamplingParams(max_tokens=n_tokens, ignore_eos=True))
    r.arrival_time = 100.0
    r.scheduled_time = 100.0 + ttft / 2
    for i in range(n_tokens):
        r.record_token(i, now=100.0 + ttft + i * itl)
    r.finish(reason, now=100.0 + ttft + n_tokens * itl)
    r.rejection_reason = rejection
    return r


def sample(text: str, name: str) -> float | None:
    for line in text.splitlines():
        if line.startswith(name + " ") or line.startswith(name + "{"):
            return float(line.rsplit(" ", 1)[1])
    return None


def render(m: EngineMetrics) -> str:
    return m.render()[0].decode()


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------


def test_completed_request_populates_every_family():
    m = EngineMetrics()
    m.observe_finished(finished_request(n_tokens=5))
    text = render(m)

    assert sample(text, 'nanoserve:request_success_total{finish_reason="length"}') == 1.0
    assert sample(text, "nanoserve:generation_tokens_total") == 5.0
    assert sample(text, "nanoserve:prompt_tokens_total") == 3.0
    assert sample(text, "nanoserve:time_to_first_token_seconds_count") == 1.0
    assert sample(text, "nanoserve:e2e_request_latency_seconds_count") == 1.0
    # Four gaps between five tokens.
    assert sample(text, "nanoserve:inter_token_latency_seconds_count") == 4.0


def test_rejections_are_counted_but_not_timed():
    """A request refused in a millisecond is not a fast request."""
    m = EngineMetrics()
    m.observe_finished(finished_request(reason=FinishReason.REJECTED, rejection="queue_full"))
    text = render(m)

    assert sample(text, 'nanoserve:request_rejected_total{reason="queue_full"}') == 1.0
    assert sample(text, "nanoserve:time_to_first_token_seconds_count") == 0.0
    assert sample(text, "nanoserve:e2e_request_latency_seconds_count") == 0.0
    assert sample(text, "nanoserve:generation_tokens_total") == 0.0


def test_finish_reasons_are_separate_series():
    m = EngineMetrics()
    m.observe_finished(finished_request(reason=FinishReason.LENGTH))
    m.observe_finished(finished_request(reason=FinishReason.EOS))
    m.observe_finished(finished_request(reason=FinishReason.CANCELLED))
    text = render(m)

    for reason in ("length", "eos", "cancelled"):
        assert sample(text, f'nanoserve:request_success_total{{finish_reason="{reason}"}}') == 1.0


def test_latency_lands_in_the_right_bucket():
    """A 50ms TTFT must be counted at le=0.06 and not at le=0.04."""
    m = EngineMetrics()
    m.observe_finished(finished_request(ttft=0.05, n_tokens=1))
    text = render(m)

    assert sample(text, 'nanoserve:time_to_first_token_seconds_bucket{le="0.04"}') == 0.0
    assert sample(text, 'nanoserve:time_to_first_token_seconds_bucket{le="0.06"}') == 1.0


def test_bucket_families_are_distinct_and_ordered():
    """Fine buckets for inter-token latency, coarse for end-to-end.

    One family for both wastes most buckets: a 13ms decode gap lands in the
    slow family's first bucket and carries no information.
    """
    assert list(SECONDS) == sorted(SECONDS)
    assert list(SECONDS_FAST) == sorted(SECONDS_FAST)
    assert SECONDS_FAST[0] < SECONDS[0]
    assert SECONDS[-1] > SECONDS_FAST[-1]
    # A typical decode step must land somewhere informative.
    assert sum(1 for b in SECONDS_FAST if b < 0.013) >= 4


# ---------------------------------------------------------------------------
# Counters must never go backwards
# ---------------------------------------------------------------------------


def test_engine_totals_become_increments(model):
    """The exporter converts totals to deltas.

    Setting a Counter to a total would work once and break the moment the
    engine is restarted or the total is recomputed -- and a Counter that
    decreases makes every rate() query on it meaningless.
    """
    m = EngineMetrics()
    e = engine(model)

    e.preemptions = 3
    m.observe_engine(e)
    assert sample(render(m), "nanoserve:num_preemptions_total") == 3.0

    # Sampling again with no change must not double-count.
    m.observe_engine(e)
    m.observe_engine(e)
    assert sample(render(m), "nanoserve:num_preemptions_total") == 3.0

    e.preemptions = 5
    m.observe_engine(e)
    assert sample(render(m), "nanoserve:num_preemptions_total") == 5.0


def test_a_total_that_drops_does_not_move_the_counter(model):
    """Defensive: a reset engine must not produce a negative increment."""
    m = EngineMetrics()
    e = engine(model)
    e.preemptions = 10
    m.observe_engine(e)
    e.preemptions = 2  # as if the engine were replaced
    m.observe_engine(e)
    assert sample(render(m), "nanoserve:num_preemptions_total") == 10.0


def test_gauges_track_current_state(model):
    m = EngineMetrics()
    e = engine(model)
    for i in range(3):
        e.add_request(Request(prompt_token_ids=[7] * 20,
                              params=SamplingParams(max_tokens=20, ignore_eos=True)))
    e.step()

    m.observe_engine(e)
    text = render(m)
    assert sample(text, "nanoserve:num_requests_running") == 3.0
    assert sample(text, "nanoserve:gpu_cache_blocks_total") == 256.0
    assert 0 < sample(text, "nanoserve:gpu_cache_usage_perc") <= 1.0


def test_prefix_cache_counters_advance(model):
    m = EngineMetrics()
    e = engine(model)
    prompt = list(range(1, 65))
    e.run([Request(prompt_token_ids=prompt + [i],
                   params=SamplingParams(max_tokens=3, ignore_eos=True)) for i in (1, 2)])
    m.observe_engine(e)
    text = render(m)

    assert sample(text, "nanoserve:prefix_cache_query_blocks_total") > 0
    assert sample(text, "nanoserve:prefix_cache_hit_blocks_total") > 0


# ---------------------------------------------------------------------------
# Export shape
# ---------------------------------------------------------------------------


def test_registry_is_private_so_two_engines_can_coexist():
    """The default global registry would raise on the second instance."""
    a, b = EngineMetrics(), EngineMetrics()
    a.observe_finished(finished_request(n_tokens=2))
    b.observe_finished(finished_request(n_tokens=7))

    assert sample(render(a), "nanoserve:generation_tokens_total") == 2.0
    assert sample(render(b), "nanoserve:generation_tokens_total") == 7.0


def test_exposition_format_is_parseable():
    m = EngineMetrics()
    m.observe_finished(finished_request())
    body, content_type = m.render()

    assert content_type.startswith("text/plain")
    text = body.decode()
    assert "# HELP nanoserve:time_to_first_token_seconds" in text
    assert "# TYPE nanoserve:time_to_first_token_seconds histogram" in text
    # Every metric carries the project prefix, so a shared Prometheus can
    # select them with one matcher.
    names = {
        line.split()[2] for line in text.splitlines() if line.startswith("# TYPE")
    }
    assert names and all(n.startswith("nanoserve:") for n in names)


def test_otel_names_are_documented_in_help():
    """The OpenTelemetry mapping is written down, not folklore."""
    text = render(EngineMetrics())
    assert "gen_ai.server.time_to_first_token" in text
    assert "gen_ai.server.time_per_output_token" in text
