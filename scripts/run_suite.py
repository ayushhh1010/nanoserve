"""Run the standard benchmark matrix and write one comparable results set.

The point of a suite rather than ad-hoc invocations is that the configurations
stay fixed. An engine comparison is only meaningful if both engines saw the
same workloads, and "the same workloads" has to be written down somewhere that
is not a shell history.

Two regimes, because they answer different questions:

**Burst** -- every request arrives at once. Measures raw capacity: how fast can
this engine drain a backlog. No queueing information, because the queue is
simply the whole workload.

**Rate sweep** -- Poisson arrivals at increasing rates. Measures behaviour under
load, which is where goodput collapses and the curve that matters lives. The
naive engine falls over at a rate the later engines will absorb, and the
comparison is the phase's headline chart.

Every configuration runs `--repeats` times so the spread is recorded. A number
without its variance cannot be compared to anything.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: The fixed matrix. Adding an engine means adding it here, not changing flags
#: at the command line, so past results stay comparable to future ones.
SUITE = {
    "burst": {"rate": None, "requests": 60, "output_len": 96},
    "rate_1": {"rate": 1.0, "requests": 60, "output_len": 96},
    "rate_2": {"rate": 2.0, "requests": 80, "output_len": 96},
    "rate_4": {"rate": 4.0, "requests": 120, "output_len": 96},
    "rate_8": {"rate": 8.0, "requests": 160, "output_len": 96},
    "long_output": {"rate": None, "requests": 30, "output_len": 384},
    "short_output": {"rate": None, "requests": 120, "output_len": 24},
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engines", nargs="+", default=["baseline"])
    ap.add_argument("--configs", nargs="+", default=list(SUITE))
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "bench" / "results")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    unknown = set(args.configs) - set(SUITE)
    if unknown:
        print(f"unknown configs: {sorted(unknown)}", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "engines": args.engines,
        "configs": {k: SUITE[k] for k in args.configs},
        "repeats": args.repeats,
        "runs": [],
    }

    total = len(args.engines) * len(args.configs)
    n = 0
    for engine in args.engines:
        for name in args.configs:
            cfg = SUITE[name]
            n += 1
            tag = f"{engine}__{name}"
            cmd = [
                sys.executable, "-u", str(ROOT / "scripts" / "run_bench.py"),
                "--engine", engine,
                "--requests", str(cfg["requests"]),
                "--output-len", str(cfg["output_len"]),
                "--repeats", str(args.repeats),
                "--tag", tag,
            ]
            if cfg["rate"] is not None:
                cmd += ["--rate", str(cfg["rate"])]

            print(f"\n{'=' * 72}\n[{n}/{total}] {tag}\n{'=' * 72}", flush=True)
            if args.dry_run:
                print("  " + " ".join(cmd))
                continue

            proc = subprocess.run(cmd, cwd=ROOT)
            manifest["runs"].append(
                {"engine": engine, "config": name, "tag": tag, "returncode": proc.returncode}
            )
            if proc.returncode != 0:
                print(f"  FAILED (exit {proc.returncode})", file=sys.stderr)

    if not args.dry_run:
        manifest["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        path = args.out_dir / "suite_manifest.json"
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        failed = [r for r in manifest["runs"] if r["returncode"] != 0]
        print(f"\n{len(manifest['runs']) - len(failed)}/{len(manifest['runs'])} runs succeeded")
        print(f"manifest: {path.relative_to(ROOT)}")
        return 1 if failed else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
