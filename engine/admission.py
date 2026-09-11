"""Admission control: refuse work you cannot finish, so the rest still lands.

An unbounded queue is the default, and it is the worst option. Under overload
the queue grows without bound, every request's wait grows with it, and
eventually *everything* misses its deadline while the server is at 100%
utilisation. Throughput looks fine. Goodput -- work still useful when it
finished -- goes to zero.

The fix is counter-intuitive and is the whole point of this module: **rejecting
requests increases the number served**. A request admitted with no chance of
meeting its deadline consumes KV blocks, batch slots and forward passes that
would otherwise have gone to a request that could have made it. Turning it away
immediately costs that one request and saves several.

Two gates, in order of cost to evaluate:

**Queue depth.** A hard cap on waiting requests. Crude, cheap, and enough to
stop the queue itself becoming the latency.

**Deadline projection.** Estimate when this request would finish given the
current queue and batch, and reject it now if that is past its deadline. A 429
that arrives in a millisecond is a far better answer than a response that
arrives after the client gave up -- the client can retry, shed, or degrade,
and none of those are options once it is already waiting.

The projection is deliberately simple: measured decode throughput, the work
already queued, and this request's own token budget. A sophisticated estimator
would be more accurate and much harder to reason about when it is wrong, and
being wrong is the normal case under load.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

from engine.types import Request


@dataclass
class AdmissionConfig:
    #: Hard cap on waiting requests. None disables the queue-depth gate.
    max_queue_depth: int | None = 64
    #: Reject when the projected finish time is past the deadline.
    enforce_deadlines: bool = True
    #: Multiplier on the projection before comparing to the deadline. Above 1
    #: admits optimistically -- the estimate is noisy and a request rejected in
    #: error is a certain failure, where an optimistic admission is only a
    #: possible one.
    slack: float = 1.25
    #: Steps of history used for the throughput estimate. Long enough to ride
    #: out a single slow step, short enough to track a real change in load.
    window: int = 50
    #: Assumed step rate before any step has been observed.
    initial_step_seconds: float = 0.02
    #: Steps that must be observed before the deadline gate is allowed to
    #: reject anything. Projecting from an assumed rate is guessing, and the
    #: guess is pessimistic: this engine's cold estimate is 50 tok/s against a
    #: real ~900, so an unguarded gate rejects a burst of perfectly servable
    #: traffic at start-up and then measures a rate from the handful it let
    #: through. The queue-depth gate still applies throughout.
    min_observations: int = 10


@dataclass
class AdmissionStats:
    offered: int = 0
    admitted: int = 0
    rejected_queue_full: int = 0
    rejected_deadline: int = 0

    @property
    def rejected(self) -> int:
        return self.rejected_queue_full + self.rejected_deadline

    @property
    def rejection_rate(self) -> float:
        return self.rejected / self.offered if self.offered else 0.0

    def to_dict(self) -> dict:
        return {
            "offered": self.offered,
            "admitted": self.admitted,
            "rejected": self.rejected,
            "rejected_queue_full": self.rejected_queue_full,
            "rejected_deadline": self.rejected_deadline,
            "rejection_rate": round(self.rejection_rate, 4),
        }


class AdmissionController:
    """Decides whether a request is worth starting."""

    def __init__(self, cfg: AdmissionConfig | None = None) -> None:
        self.cfg = cfg or AdmissionConfig()
        self.stats = AdmissionStats()
        self._step_times: deque[float] = deque(maxlen=self.cfg.window)
        self._batch_sizes: deque[int] = deque(maxlen=self.cfg.window)

    # -- observation --------------------------------------------------------

    def record_step(self, seconds: float, batch_size: int) -> None:
        """Feed one completed decode step into the throughput estimate."""
        if seconds > 0:
            self._step_times.append(seconds)
            self._batch_sizes.append(max(1, batch_size))

    @property
    def seconds_per_step(self) -> float:
        if not self._step_times:
            return self.cfg.initial_step_seconds
        return sum(self._step_times) / len(self._step_times)

    @property
    def is_warm(self) -> bool:
        """Has enough been measured to project from?"""
        return len(self._step_times) >= self.cfg.min_observations

    @property
    def tokens_per_second(self) -> float:
        """Measured aggregate decode rate.

        Derived from observed step time and batch size rather than assumed,
        because the same engine runs at wildly different rates depending on how
        full the batch is -- which is exactly the state this has to reason
        about.
        """
        if not self._step_times:
            return 1.0 / self.cfg.initial_step_seconds
        mean_batch = sum(self._batch_sizes) / len(self._batch_sizes)
        return mean_batch / self.seconds_per_step

    # -- the decision -------------------------------------------------------

    def project_finish(
        self, req: Request, queued_tokens: int, running_tokens: int, now: float | None = None
    ) -> float:
        """Estimate when `req` would finish if admitted now.

        Work ahead of it is everything already running plus everything already
        queued; its own work is its token budget. Dividing by the measured
        aggregate rate gives a wall-clock estimate that accounts for batching:
        a fuller batch finishes the same backlog faster.
        """
        now = time.perf_counter() if now is None else now
        ahead = queued_tokens + running_tokens
        own = req.params.max_tokens
        rate = max(self.tokens_per_second, 1e-6)
        return now + (ahead + own) / rate

    def admit(
        self,
        req: Request,
        queue_depth: int,
        queued_tokens: int,
        running_tokens: int,
        now: float | None = None,
    ) -> tuple[bool, str]:
        """(admit?, reason). Reason is empty when admitted."""
        self.stats.offered += 1
        now = time.perf_counter() if now is None else now

        if self.cfg.max_queue_depth is not None and queue_depth >= self.cfg.max_queue_depth:
            self.stats.rejected_queue_full += 1
            return False, "queue_full"

        warm = len(self._step_times) >= self.cfg.min_observations
        if self.cfg.enforce_deadlines and req.deadline is not None and warm:
            finish = self.project_finish(req, queued_tokens, running_tokens, now)
            # slack > 1 lets a marginal request through: the estimate is noisy,
            # and a wrongly rejected request fails for certain where a wrongly
            # admitted one only might.
            if (finish - now) > (req.deadline - now) * self.cfg.slack:
                self.stats.rejected_deadline += 1
                return False, "deadline_exceeded"

        self.stats.admitted += 1
        return True, ""

    def __repr__(self) -> str:
        s = self.stats
        return (
            f"AdmissionController(offered={s.offered}, admitted={s.admitted}, "
            f"rejected={s.rejected}, rate={self.tokens_per_second:.0f} tok/s)"
        )
