"""Prometheus metrics for the serving engine.

Naming follows vLLM's convention -- a `nanoserve:` prefix, `_seconds` on every
time-bearing metric -- so anyone who has built a dashboard for a real inference
server can read this one without a translation table. The OpenTelemetry GenAI
semantic conventions name the same two quantities `gen_ai.server.time_to_first_token`
and `gen_ai.server.time_per_output_token`; those names are recorded here as
metric documentation so the mapping is explicit rather than folklore.

**Histograms, not gauges, for latency.** A gauge holding "current p99" is
meaningless: it cannot be aggregated across replicas, cannot be re-windowed,
and silently lies whenever the scrape interval and the traffic pattern
disagree. Buckets aggregate correctly, which matters the moment there is more
than one server -- which is exactly where Phase 3 goes.

Two bucket families, following the same source:

  SECONDS      0.05s .. 300s   end-to-end and queueing, where seconds matter
  SECONDS_FAST 0.001s .. 60s   inter-token latency, which lives in milliseconds

Using one family for both would waste most buckets: an inter-token latency of
13ms lands in the first bucket of the slow family and carries no information.
"""

from __future__ import annotations

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from engine.types import FinishReason, Request

#: End-to-end and queueing latencies, which range from tens of milliseconds to
#: minutes under overload.
SECONDS = (
    0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 7.5,
    10.0, 15.0, 20.0, 30.0, 45.0, 60.0, 120.0, 300.0,
)

#: Inter-token latency, which lives in milliseconds. A decode step here is
#: ~13ms, so the slow family's first bucket would swallow the entire
#: distribution.
SECONDS_FAST = (
    0.001, 0.0025, 0.005, 0.0075, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1,
    0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0,
)

TOKENS = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 4096)


class EngineMetrics:
    """Prometheus collectors for one engine instance.

    Owns a private registry rather than using the global default, so two
    engines in one process -- which every test module does -- do not collide
    on duplicate metric names.
    """

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry or CollectorRegistry()
        r = self.registry

        # -- counters -------------------------------------------------------
        self.requests_total = Counter(
            "nanoserve:request_success_total",
            "Requests that finished, by reason.",
            ["finish_reason"], registry=r,
        )
        self.requests_rejected = Counter(
            "nanoserve:request_rejected_total",
            "Requests refused by admission control, by gate.",
            ["reason"], registry=r,
        )
        self.prompt_tokens = Counter(
            "nanoserve:prompt_tokens_total", "Prompt tokens accepted.", registry=r
        )
        self.generation_tokens = Counter(
            "nanoserve:generation_tokens_total", "Tokens generated.", registry=r
        )
        self.preemptions = Counter(
            "nanoserve:num_preemptions_total",
            "Sequences evicted under memory pressure.", registry=r,
        )
        self.prefix_hit_blocks = Counter(
            "nanoserve:prefix_cache_hit_blocks_total",
            "KV blocks served from the prefix cache.", registry=r,
        )
        self.prefix_query_blocks = Counter(
            "nanoserve:prefix_cache_query_blocks_total",
            "KV blocks looked up in the prefix cache.", registry=r,
        )

        # -- gauges ---------------------------------------------------------
        self.num_running = Gauge(
            "nanoserve:num_requests_running", "Sequences in the current batch.", registry=r
        )
        self.num_waiting = Gauge(
            "nanoserve:num_requests_waiting", "Requests queued for admission.", registry=r
        )
        self.kv_usage = Gauge(
            "nanoserve:gpu_cache_usage_perc",
            "Fraction of KV blocks allocated, 0-1.", registry=r,
        )
        self.kv_blocks_total = Gauge(
            "nanoserve:gpu_cache_blocks_total", "KV blocks in the pool.", registry=r
        )

        # -- histograms -----------------------------------------------------
        self.ttft = Histogram(
            "nanoserve:time_to_first_token_seconds",
            "Arrival to first token. OpenTelemetry: gen_ai.server.time_to_first_token.",
            buckets=SECONDS_FAST, registry=r,
        )
        self.inter_token = Histogram(
            "nanoserve:inter_token_latency_seconds",
            "Gap between consecutive output tokens. "
            "OpenTelemetry: gen_ai.server.time_per_output_token.",
            buckets=SECONDS_FAST, registry=r,
        )
        self.e2e = Histogram(
            "nanoserve:e2e_request_latency_seconds",
            "Arrival to completion.", buckets=SECONDS, registry=r,
        )
        self.queue_time = Histogram(
            "nanoserve:request_queue_time_seconds",
            "Arrival to first scheduling.", buckets=SECONDS, registry=r,
        )
        self.output_tokens = Histogram(
            "nanoserve:request_generation_tokens",
            "Tokens generated per request.", buckets=TOKENS, registry=r,
        )

    # -- recording ----------------------------------------------------------

    def observe_finished(self, req: Request) -> None:
        """Record one completed or rejected request.

        Rejections are counted separately and contribute no latency samples:
        a request refused in a millisecond would otherwise drag every latency
        histogram toward zero and make an overloaded server look fast.
        """
        if req.finish_reason is FinishReason.REJECTED:
            self.requests_rejected.labels(reason=req.rejection_reason or "unknown").inc()
            return

        reason = req.finish_reason.value if req.finish_reason else "unknown"
        self.requests_total.labels(finish_reason=reason).inc()
        self.prompt_tokens.inc(req.prompt_len)
        self.generation_tokens.inc(req.output_len)
        self.output_tokens.observe(req.output_len)

        if req.ttft is not None:
            self.ttft.observe(req.ttft)
        if req.queue_time is not None:
            self.queue_time.observe(max(0.0, req.queue_time))
        if req.e2e_latency is not None:
            self.e2e.observe(req.e2e_latency)
        for gap in req.inter_token_latencies:
            self.inter_token.observe(gap)

    def observe_engine(self, engine) -> None:
        """Sample the gauges. Cheap enough to call every step."""
        s = engine.stats()
        self.num_running.set(s.num_running)
        self.num_waiting.set(s.num_waiting)
        self.kv_blocks_total.set(s.kv_blocks_total)
        self.kv_usage.set(s.kv_utilization)

        # Counters must only ever move forward, so engine totals are converted
        # to deltas rather than set. A Counter that goes backwards breaks every
        # rate() query built on it.
        self._advance(self.preemptions, "_preemptions", engine.preemptions)
        ps = engine.prefix_cache.stats
        self._advance(self.prefix_hit_blocks, "_prefix_hits", ps.hit_blocks)
        self._advance(self.prefix_query_blocks, "_prefix_queries", ps.query_blocks)

    def _advance(self, counter: Counter, attr: str, total: int) -> None:
        seen = getattr(self, attr, 0)
        if total > seen:
            counter.inc(total - seen)
            setattr(self, attr, total)

    # -- export -------------------------------------------------------------

    def render(self) -> tuple[bytes, str]:
        """(body, content-type) for a /metrics response."""
        return generate_latest(self.registry), CONTENT_TYPE_LATEST
