"""The vocabulary every engine in Phase 2 speaks.

These types are the seam between the load generator, the engines, and the
metrics. Getting them right now matters more than it looks: the paged engine,
the continuous-batching scheduler and the admission controller all have to be
measurable by the *same* harness as the naive baseline, or the comparison that
the whole phase rests on is not a comparison.

Two things are deliberately in here that a first version usually omits:

**Timestamps are recorded per event, not per request.** Time to first token and
inter-token latency are different numbers with different causes -- TTFT is
dominated by queueing and prefill, inter-token latency by the decode loop. A
scheduler can improve one while destroying the other, and a single end-to-end
latency figure hides exactly that.

**Every request carries a deadline.** Nothing uses it until admission control
in week 7, but goodput -- requests completed *within SLO* -- is the headline
metric of this phase, and it cannot be computed retroactively from runs that
never recorded what the SLO was.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from enum import Enum

_ids = itertools.count()


class FinishReason(str, Enum):
    EOS = "eos"  # model emitted the stop token
    LENGTH = "length"  # hit max_tokens
    CANCELLED = "cancelled"  # client disconnected mid-generation
    REJECTED = "rejected"  # admission control refused it
    PREEMPTED = "preempted"  # evicted and not restarted (should not happen)


@dataclass
class SamplingParams:
    """How to turn logits into a token.

    Greedy by default. Every engine in this phase must reproduce the baseline
    token for token under greedy decoding, which is only a meaningful test if
    greedy is the default rather than an option someone forgets to set.
    """

    max_tokens: int = 128
    temperature: float = 0.0
    top_k: int | None = None
    top_p: float | None = None
    ignore_eos: bool = False  # fixed-length generation, for clean benchmarks

    @property
    def greedy(self) -> bool:
        return self.temperature == 0.0


@dataclass
class Request:
    """One inference request, from arrival to completion."""

    prompt_token_ids: list[int]
    params: SamplingParams = field(default_factory=SamplingParams)
    request_id: str = field(default_factory=lambda: f"req-{next(_ids)}")

    #: When the client sent it. Set by the load generator, not the engine, so
    #: that time spent queued is visible rather than absorbed.
    arrival_time: float = field(default_factory=time.perf_counter)
    #: Absolute deadline. None means best-effort.
    deadline: float | None = None

    #: Set by the engine as it runs.
    scheduled_time: float | None = None
    first_token_time: float | None = None
    finish_time: float | None = None
    token_times: list[float] = field(default_factory=list)
    output_token_ids: list[int] = field(default_factory=list)
    finish_reason: FinishReason | None = None

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def output_len(self) -> int:
        return len(self.output_token_ids)

    @property
    def total_len(self) -> int:
        return self.prompt_len + self.output_len

    @property
    def total_len_estimate(self) -> int:
        """Longest this sequence can become: prompt plus its token budget.

        What a contiguous allocator must reserve, since it cannot know where
        the sequence will actually stop. The gap between this and `total_len`
        at completion is precisely the internal fragmentation being measured.
        """
        return self.prompt_len + self.params.max_tokens

    def record_token(self, token_id: int, now: float | None = None) -> None:
        now = time.perf_counter() if now is None else now
        if not self.output_token_ids:
            self.first_token_time = now
        self.output_token_ids.append(token_id)
        self.token_times.append(now)

    def finish(self, reason: FinishReason, now: float | None = None) -> None:
        self.finish_time = time.perf_counter() if now is None else now
        self.finish_reason = reason

    # -- derived latencies --------------------------------------------------

    @property
    def ttft(self) -> float | None:
        """Time to first token, from the client's point of view.

        Measured from arrival rather than from scheduling, because queueing
        delay is latency the user experiences. An engine that hides a growing
        queue behind a fast prefill has not made anything faster.
        """
        if self.first_token_time is None:
            return None
        return self.first_token_time - self.arrival_time

    @property
    def queue_time(self) -> float | None:
        if self.scheduled_time is None:
            return None
        return self.scheduled_time - self.arrival_time

    @property
    def e2e_latency(self) -> float | None:
        if self.finish_time is None:
            return None
        return self.finish_time - self.arrival_time

    @property
    def inter_token_latencies(self) -> list[float]:
        """Gaps between consecutive output tokens.

        Excludes the first token: the gap before it is TTFT, which is a
        different quantity with a different cause. Averaging them together
        produces a number that describes neither.
        """
        return [b - a for a, b in zip(self.token_times, self.token_times[1:])]

    @property
    def met_deadline(self) -> bool:
        """Did this request finish within its SLO? The goodput predicate."""
        if self.finish_reason not in (FinishReason.EOS, FinishReason.LENGTH):
            return False
        if self.deadline is None:
            return True
        return self.finish_time is not None and self.finish_time <= self.deadline

    def __repr__(self) -> str:
        return (
            f"Request({self.request_id}, prompt={self.prompt_len}, "
            f"out={self.output_len}, {self.finish_reason.value if self.finish_reason else 'running'})"
        )


@dataclass
class EngineStats:
    """A snapshot of engine state, sampled during a run.

    The naive baseline can only ever report 1 or 0 running. The fields exist
    anyway so that the same harness can chart the scheduler in week 6 without
    the metrics code needing to know which engine it is talking to.
    """

    timestamp: float
    num_running: int = 0
    num_waiting: int = 0
    num_finished: int = 0
    kv_blocks_used: int = 0
    kv_blocks_total: int = 0
    prefix_cache_hits: int = 0
    prefix_cache_queries: int = 0

    @property
    def kv_utilization(self) -> float:
        return self.kv_blocks_used / self.kv_blocks_total if self.kv_blocks_total else 0.0

    @property
    def prefix_hit_rate(self) -> float:
        return self.prefix_cache_hits / self.prefix_cache_queries if self.prefix_cache_queries else 0.0
