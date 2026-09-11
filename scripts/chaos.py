"""Run the chaos suite.

    python scripts/chaos.py                    # every scenario
    python scripts/chaos.py --only etcd_down   # one of them
    python scripts/chaos.py --list

Needs etcd and Redis running, and the Go binaries built:

    docker run -d --name nanoserve-etcd  -p 2379:2379 registry.k8s.io/etcd:3.7.1-0 \
      etcd --name n1 --listen-client-urls http://0.0.0.0:2379 \
           --advertise-client-urls http://127.0.0.1:2379 --data-dir /tmp/etcd
    docker run -d --name nanoserve-redis -p 6380:6379 redis:8-alpine
    cd router && go build -o router.exe ./cmd/router && go build -o autoscaler.exe ./cmd/autoscaler

Exit code is non-zero if any scenario fails, so CI can gate on it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bench.chaos.scenarios import ALL, ScenarioResult  # noqa: E402


def cooldown(timeout: float = 90.0) -> None:
    """Wait for the previous scenario's replicas to actually let go of the GPU.

    Not politeness. Each scenario runs three model replicas on a 4 GB card, and
    a terminated process does not release its VRAM at the moment the parent
    stops waiting for it. Running scenarios back to back put the next cluster
    into a card that was still full: it started, technically, and then served
    13 requests in 74 seconds. That looks exactly like a routing failure in the
    report and is nothing of the kind -- the scenario before it was still
    shutting down. Measuring a starved cluster and calling the result 'chaos'
    is the fastest way to ship a fix for a bug that does not exist.
    """
    import subprocess as sp

    deadline = time.time() + timeout
    baseline = None
    while time.time() < deadline:
        try:
            out = sp.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            ).stdout.strip().splitlines()
            used = int(out[0])
        except (OSError, ValueError, IndexError, sp.SubprocessError):
            time.sleep(5)  # no nvidia-smi: fall back to a fixed pause
            return
        if baseline is None:
            baseline = used
        if used < 400:
            time.sleep(2)
            return
        time.sleep(2)
    print(f"  (cooldown timed out with {used} MiB still resident)", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", action="append", default=None,
                    help="run just these scenarios (repeatable)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "bench" / "results" / "chaos.json"))
    args = ap.parse_args()

    if args.list:
        for name in ALL:
            print(name)
        return 0

    names = args.only or list(ALL)
    unknown = [n for n in names if n not in ALL]
    if unknown:
        print(f"unknown scenario(s): {', '.join(unknown)}", file=sys.stderr)
        return 2

    results: list[ScenarioResult] = []
    for name in names:
        print(f"\n=== {name} ".ljust(72, "="), flush=True)
        started = time.time()
        try:
            result = ALL[name]()
        except Exception as exc:  # noqa: BLE001
            # A scenario that crashes is a failure, not an absence. Swallowing
            # it would let the suite report "6 passed" while one never ran.
            traceback.print_exc()
            result = ScenarioResult(
                name=name, assertion="scenario completed", passed=False,
                note=f"raised {type(exc).__name__}: {exc}",
            )
        elapsed = time.time() - started
        result.observed["elapsed_seconds"] = round(elapsed, 1)
        results.append(result)
        print(result.line(), flush=True)
        if result.observed:
            print(json.dumps(result.observed, indent=2, default=str), flush=True)
        if result.note:
            print(f"note: {result.note}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([asdict(r) for r in results], indent=2, default=str))

    passed = sum(1 for r in results if r.passed)
    print("\n" + "=" * 72)
    for r in results:
        print(f"  {r.line()}")
    print(f"\n{passed}/{len(results)} scenarios passed -> {out}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
