"""Driving an engine against an arrival schedule.

**Open loop, not closed loop.** Requests are submitted when their arrival time
says so, whether or not the engine has kept up. A closed-loop harness -- N
workers that each submit the next request only after the previous one returns --
is self-limiting: it cannot generate more load than the system can absorb, so
it can never show a queue growing, latency collapsing, or goodput falling off a
cliff. Those are precisely the phenomena this phase is about, so the load
generator must be willing to overwhelm the engine.

**Warm-up is not optional.** The first measurement in this project was 33
tokens/sec against a roofline of 3,100, and a large part of that was a GPU
sitting at 210 MHz because the benchmark ran three warm-up iterations on an
idle card. Every run here warms the model until the clocks are up and the
allocator has settled.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

import torch

from bench.metrics import RunResult
from bench.workload import rebase
from engine.types import Request, SamplingParams


@dataclass
class RunnerConfig:
    #: Warm-up generations before the clock starts. Enough to clock the GPU up
    #: and let the caching allocator reach steady state.
    warmup_requests: int = 12
    warmup_prompt_len: int = 128
    warmup_output_len: int = 32
    #: Idle poll interval while waiting for the next scheduled arrival.
    poll_seconds: float = 0.0005
    #: Abort a run that exceeds this, so a pathological configuration fails
    #: fast instead of hanging a benchmark sweep.
    timeout_seconds: float = 900.0
    verbose: bool = True


def warmup(engine, cfg: RunnerConfig, vocab_size: int) -> None:
    """Run throwaway requests until the hardware is at steady state."""
    import numpy as np

    rng = np.random.default_rng(1234)
    reqs = [
        Request(
            prompt_token_ids=[int(x) for x in rng.integers(1, vocab_size - 1, cfg.warmup_prompt_len)],
            params=SamplingParams(max_tokens=cfg.warmup_output_len, ignore_eos=True),
        )
        for _ in range(cfg.warmup_requests)
    ]
    # Submit everything before stepping. A batching engine warmed one request
    # at a time never runs its batched path, so the first measured steps pay
    # the allocator and kernel-selection costs the warm-up was supposed to
    # absorb -- which showed up as a 2.5x spread across repeats.
    for r in reqs:
        engine.add_request(r)
    while engine.has_work():
        engine.step()

    engine.finished.clear()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run(
    engine,
    requests: list[Request],
    workload_summary: dict,
    cfg: RunnerConfig | None = None,
    vocab_size: int = 8192,
) -> RunResult:
    """Submit `requests` on schedule and drive the engine until all finish."""
    cfg = cfg or RunnerConfig()

    if cfg.verbose:
        print(f"  warming up ({cfg.warmup_requests} requests)...", flush=True)
    warmup(engine, cfg, vocab_size)

    pending = deque(requests)
    t0 = time.perf_counter()
    # Workload times are offsets from zero. Rebasing is shared with any other
    # caller rather than inlined here, because an un-rebased deadline makes
    # admission control reject 100% of traffic.
    rebase(requests, t0)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    completed: list[Request] = []
    last_report = t0

    while pending or engine.has_work():
        now = time.perf_counter()

        if now - t0 > cfg.timeout_seconds:
            raise TimeoutError(
                f"run exceeded {cfg.timeout_seconds}s with "
                f"{len(pending)} unsubmitted and {len(completed)} completed"
            )

        while pending and pending[0].arrival_time <= now:
            engine.add_request(pending.popleft())

        drains = hasattr(engine, "drain_finished")

        if engine.has_work():
            stepped = engine.step()
            if drains:
                # Engines with admission control report every outcome through
                # one drain, rejections included. Those belong in the results
                # as the rejection rate and never as goodput -- dropping them
                # would make an engine that refuses everything look like one
                # that served a small workload perfectly.
                completed.extend(engine.drain_finished())
                engine._emitted.clear()  # nothing is streaming these
            else:
                completed.extend(stepped)
        else:
            if drains:
                completed.extend(engine.drain_finished())
            if pending:
                # Nothing to do until the next arrival. Sleeping rather than
                # spinning keeps the benchmark process from competing with the
                # engine for CPU -- and draining is not a substitute for it.
                time.sleep(
                    min(cfg.poll_seconds, max(0.0, pending[0].arrival_time - now))
                )

        if cfg.verbose and now - last_report > 5.0:
            last_report = now
            print(
                f"    {len(completed):>5,}/{len(requests):,} done, "
                f"{len(pending):>5,} not yet arrived, {now - t0:6.1f}s",
                flush=True,
            )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0

    return RunResult(
        engine=engine.name,
        wall_seconds=wall,
        requests=completed,
        workload=workload_summary,
        extra={},
    )
