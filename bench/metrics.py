"""Turning a finished run into numbers, and recording what produced them.

Percentiles, never means alone. A mean latency hides the tail completely, and
the tail is the entire subject of this phase: an engine that halves the median
while tripling p99 has made the service worse, and a single average would call
that an improvement.

`environment()` records more than is conventional, because this project has
already been bitten twice by a number that was correct but not comparable. A
benchmark run on battery measured 18,800 tok/s; the identical code on AC power
measured 29,400. Nothing in a results file that records only "RTX 3050" would
let you tell those two runs apart six weeks later. So the enforced power limit,
the clocks and the driver version go in the file.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from engine.types import FinishReason, Request


def percentiles(values: list[float], ps=(50, 90, 95, 99)) -> dict:
    if not values:
        return {f"p{p}": None for p in ps} | {"mean": None, "min": None, "max": None, "n": 0}
    a = np.asarray(values, dtype=float)
    out = {f"p{p}": float(np.percentile(a, p)) for p in ps}
    out |= {"mean": float(a.mean()), "min": float(a.min()), "max": float(a.max()), "n": int(a.size)}
    return out


@dataclass
class RunResult:
    engine: str
    wall_seconds: float
    requests: list[Request]
    workload: dict
    extra: dict

    def summary(self) -> dict:
        done = [r for r in self.requests if r.finish_reason in (FinishReason.EOS, FinishReason.LENGTH)]
        rejected = [r for r in self.requests if r.finish_reason is FinishReason.REJECTED]

        out_tokens = sum(r.output_len for r in done)
        prompt_tokens = sum(r.prompt_len for r in done)

        ttfts = [r.ttft for r in done if r.ttft is not None]
        e2es = [r.e2e_latency for r in done if r.e2e_latency is not None]
        queues = [r.queue_time for r in done if r.queue_time is not None]
        itls = [x for r in done for x in r.inter_token_latencies]

        # Normalised per-request latency: end-to-end divided by tokens
        # produced. Lets a 500-token request and a 10-token one be compared,
        # which raw e2e latency cannot do.
        norm = [
            r.e2e_latency / r.output_len
            for r in done
            if r.e2e_latency is not None and r.output_len
        ]

        met = [r for r in done if r.met_deadline]

        return {
            "engine": self.engine,
            "wall_seconds": round(self.wall_seconds, 4),
            "counts": {
                "submitted": len(self.requests),
                "completed": len(done),
                "rejected": len(rejected),
                "prompt_tokens": prompt_tokens,
                "output_tokens": out_tokens,
            },
            "throughput": {
                "output_tokens_per_s": out_tokens / self.wall_seconds if self.wall_seconds else 0,
                "total_tokens_per_s": (prompt_tokens + out_tokens) / self.wall_seconds
                if self.wall_seconds
                else 0,
                "requests_per_s": len(done) / self.wall_seconds if self.wall_seconds else 0,
            },
            "goodput": {
                # Requests completed within SLO, per second. The headline
                # metric: throughput counts work done, goodput counts work
                # that was still useful when it finished.
                "requests_per_s": len(met) / self.wall_seconds if self.wall_seconds else 0,
                "fraction_met_slo": len(met) / len(done) if done else 0.0,
                "rejection_rate": len(rejected) / len(self.requests) if self.requests else 0.0,
            },
            "latency_s": {
                "ttft": percentiles(ttfts),
                "e2e": percentiles(e2es),
                "queue": percentiles(queues),
                "inter_token": percentiles(itls),
                "per_output_token": percentiles(norm),
            },
            "workload": self.workload,
            "extra": self.extra,
        }


def _nvidia_smi(fields: str) -> dict:
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=8,
        ).stdout.strip().splitlines()[0]
        return dict(zip(fields.split(","), [v.strip() for v in out.split(",")]))
    except Exception:
        return {}


def environment() -> dict:
    """Everything needed to know whether two results are comparable."""
    import torch

    env = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        env |= {
            "gpu": props.name,
            "gpu_memory_gb": round(props.total_memory / 1024**3, 2),
            "compute_capability": f"{props.major}.{props.minor}",
            "sm_count": props.multi_processor_count,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
        }
        # The fields that made two correct measurements look contradictory.
        env |= _nvidia_smi(
            "driver_version,enforced.power.limit,power.default_limit,clocks.max.sm,clocks.sm"
        )
    try:
        import triton

        env["triton"] = triton.__version__
    except ImportError:
        env["triton"] = None
    return env


def save_result(result: RunResult, path: Path, extra: dict | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "environment": environment(),
        "result": result.summary(),
        **(extra or {}),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def aggregate(summaries: list[dict]) -> dict:
    """Combine repeated runs of the same configuration.

    Standard practice is to repeat a benchmark and report the spread, not a
    single number, and this project has a specific reason to care: two correct
    measurements here once differed by 55% purely because the machine changed
    power state between them. A single run cannot tell you that happened; a
    coefficient of variation across repeats can.

    Reports the median as the headline (robust to one bad run) alongside the
    spread, and flags configurations whose variation is high enough that the
    comparison should not be trusted.
    """
    if len(summaries) == 1:
        return {"repeats": 1, "representative": summaries[0], "stable": True}

    def pluck(path: tuple[str, ...]) -> list[float]:
        out = []
        for s in summaries:
            node = s
            for key in path:
                node = node[key]
            if node is not None:
                out.append(float(node))
        return out

    tracked = {
        "output_tokens_per_s": ("throughput", "output_tokens_per_s"),
        "requests_per_s": ("throughput", "requests_per_s"),
        "goodput_per_s": ("goodput", "requests_per_s"),
        "ttft_p50": ("latency_s", "ttft", "p50"),
        "ttft_p99": ("latency_s", "ttft", "p99"),
        "itl_p50": ("latency_s", "inter_token", "p50"),
        "itl_p99": ("latency_s", "inter_token", "p99"),
    }

    stats, worst_cv = {}, 0.0
    for name, path in tracked.items():
        vals = pluck(path)
        if not vals:
            continue
        arr = np.asarray(vals)
        mean = float(arr.mean())
        # Coefficient of variation: spread relative to magnitude, so a latency
        # in milliseconds and a throughput in thousands are comparable.
        cv = float(arr.std() / mean) if mean else 0.0
        stats[name] = {
            "median": float(np.median(arr)),
            "mean": mean,
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "cv": cv,
            "values": vals,
        }
        # Tail percentiles are legitimately noisy; hold them to a looser bar
        # than throughput, which should be steady on a healthy machine.
        if not name.endswith("p99"):
            worst_cv = max(worst_cv, cv)

    # The representative run is the one whose throughput is the median, not
    # whichever happened to execute in the middle. Picking by order would
    # report a 300 tok/s run as the "median" of [100, 300, 200].
    by_throughput = sorted(
        summaries, key=lambda s: s["throughput"]["output_tokens_per_s"]
    )
    representative = by_throughput[len(by_throughput) // 2]

    return {
        "repeats": len(summaries),
        "representative": representative,
        "across_repeats": stats,
        "worst_cv": worst_cv,
        # 5% is tight enough to catch a thermal ramp or a power-state change,
        # loose enough not to fire on ordinary scheduling noise.
        "stable": worst_cv < 0.05,
    }


def format_repeats(agg: dict) -> str:
    if agg["repeats"] == 1:
        return ""
    lines = [
        "",
        f"  across {agg['repeats']} repeats:",
        f"  {'':22}{'median':>10}{'min':>10}{'max':>10}{'cv':>8}",
    ]
    for name, s in agg["across_repeats"].items():
        # Latencies are stored in seconds but read in milliseconds everywhere
        # else in this output. Scale them here rather than leaving two units
        # in one report.
        latency = name.startswith(("ttft", "itl", "e2e"))
        scale = 1000.0 if latency else 1.0
        label = f"{name} (ms)" if latency else name
        lines.append(
            f"  {label:<22}{s['median'] * scale:>10.2f}{s['min'] * scale:>10.2f}"
            f"{s['max'] * scale:>10.2f}{s['cv'] * 100:>7.1f}%"
        )
    if not agg["stable"]:
        lines += [
            "",
            f"  UNSTABLE: worst coefficient of variation {agg['worst_cv'] * 100:.1f}% "
            f"exceeds 5%.",
            "  The machine changed underneath the run -- thermal ramp, power state,",
            "  or another process competing. Do not compare these numbers to another",
            "  configuration until the run is stable.",
        ]
    return "\n".join(lines)


def format_summary(summary: dict) -> str:
    """A compact human-readable block, printed after every run."""
    c, t, g, lat = summary["counts"], summary["throughput"], summary["goodput"], summary["latency_s"]

    def ms(d, key):
        """Milliseconds, 9 wide. Queue waits under overload run to five
        figures, and a narrower field silently runs the columns together."""
        v = d.get(key)
        return f"{'-':>9}" if v is None else f"{v * 1000:9.1f}"

    lines = [
        f"  engine            {summary['engine']}",
        f"  wall              {summary['wall_seconds']:.2f} s",
        f"  completed         {c['completed']:,} / {c['submitted']:,}"
        + (f"  ({c['rejected']:,} rejected)" if c["rejected"] else ""),
        f"  output tokens     {c['output_tokens']:,}",
        "",
        f"  throughput        {t['output_tokens_per_s']:>9,.0f} output tok/s"
        f"   ({t['requests_per_s']:.2f} req/s)",
        f"  goodput           {g['requests_per_s']:>9.2f} req/s within SLO"
        f"   ({g['fraction_met_slo'] * 100:.1f}% met)",
        "",
        f"  {'':18}{'p50':>9}{'p90':>9}{'p99':>9}{'max':>9}   (ms)",
        f"  TTFT              {ms(lat['ttft'], 'p50')}{ms(lat['ttft'], 'p90')}"
        f"{ms(lat['ttft'], 'p99')}{ms(lat['ttft'], 'max')}",
        f"  inter-token       {ms(lat['inter_token'], 'p50')}{ms(lat['inter_token'], 'p90')}"
        f"{ms(lat['inter_token'], 'p99')}{ms(lat['inter_token'], 'max')}",
        f"  end-to-end        {ms(lat['e2e'], 'p50')}{ms(lat['e2e'], 'p90')}"
        f"{ms(lat['e2e'], 'p99')}{ms(lat['e2e'], 'max')}",
        f"  queue wait        {ms(lat['queue'], 'p50')}{ms(lat['queue'], 'p90')}"
        f"{ms(lat['queue'], 'p99')}{ms(lat['queue'], 'max')}",
    ]
    return "\n".join(lines)
