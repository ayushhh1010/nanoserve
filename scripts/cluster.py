"""Bring up a local cluster: N gRPC replicas behind the Go router.

    python scripts/cluster.py --replicas 3

Starts everything, waits for readiness, and keeps running until interrupted.
Used by the end-to-end and chaos tests, and by hand for a demo.

Replica ports start at 9101 because Windows reserves 50000-50559 for
Hyper-V/WSL/Docker, so the conventional gRPC 50051 cannot be bound here at all
-- and the failure surfaces only as "connection refused".
"""

from __future__ import annotations

import argparse
import atexit
import json
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ROUTER_BIN = ROOT / "router" / "router.exe"
BASE_PORT = 9101


@dataclass
class Cluster:
    replicas: list[subprocess.Popen] = field(default_factory=list)
    replica_ports: list[int] = field(default_factory=list)
    router: subprocess.Popen | None = None
    router_addr: str = "127.0.0.1:8080"

    def stop(self) -> None:
        if self.router is not None and self.router.poll() is None:
            self.router.terminate()
            try:
                self.router.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.router.kill()
        for p in self.replicas:
            if p.poll() is None:
                p.terminate()
        for p in self.replicas:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()

    def kill_replica(self, index: int) -> None:
        """SIGKILL one replica, the way a machine failure looks."""
        p = self.replicas[index]
        if p.poll() is None:
            p.kill()

    def stats(self) -> dict:
        with urllib.request.urlopen(f"http://{self.router_addr}/stats", timeout=5) as r:
            return json.loads(r.read())


def wait_for_router(addr: str, timeout: float = 90.0) -> bool:
    """Poll /healthz until at least one replica is ready."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://{addr}/healthz", timeout=2) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(1.0)
    return False


def start(
    n_replicas: int = 3,
    kv_budget_mb: int = 256,
    max_batch_size: int = 32,
    router_addr: str = "127.0.0.1:8080",
    epsilon: float = 0.25,
    prefix_len: int = 128,
    device: str = "cuda",
    quiet: bool = True,
) -> Cluster:
    if not ROUTER_BIN.exists():
        raise SystemExit(
            f"{ROUTER_BIN} not built. Run:\n"
            f'  cd router && go build -o router.exe ./cmd/router'
        )

    cluster = Cluster(router_addr=router_addr)
    atexit.register(cluster.stop)
    sink = subprocess.DEVNULL if quiet else None

    for i in range(n_replicas):
        port = BASE_PORT + i
        cluster.replica_ports.append(port)
        cluster.replicas.append(
            subprocess.Popen(
                [sys.executable, "-u", str(ROOT / "scripts" / "serve_replica.py"),
                 "--port", str(port), "--kv-budget-mb", str(kv_budget_mb),
                 "--max-batch-size", str(max_batch_size), "--device", device,
                 "--replica-id", f"replica-{i}"],
                cwd=ROOT, stdout=sink, stderr=sink,
            )
        )

    spec = ",".join(
        f"replica-{i}=127.0.0.1:{BASE_PORT + i}" for i in range(n_replicas)
    )
    cluster.router = subprocess.Popen(
        [str(ROUTER_BIN), "--addr", router_addr, "--replicas", spec,
         "--epsilon", str(epsilon), "--prefix-len", str(prefix_len)],
        cwd=ROOT, stdout=sink, stderr=sink,
    )
    return cluster


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--replicas", type=int, default=3)
    ap.add_argument("--kv-budget-mb", type=int, default=256)
    ap.add_argument("--max-batch-size", type=int, default=32)
    ap.add_argument("--addr", default="127.0.0.1:8080")
    ap.add_argument("--epsilon", type=float, default=0.25)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    cluster = start(
        n_replicas=args.replicas, kv_budget_mb=args.kv_budget_mb,
        max_batch_size=args.max_batch_size, router_addr=args.addr,
        epsilon=args.epsilon, device=args.device, quiet=not args.verbose,
    )
    print(f"starting {args.replicas} replicas + router on {args.addr} ...", flush=True)

    if not wait_for_router(args.addr):
        cluster.stop()
        print("cluster did not become ready", file=sys.stderr)
        return 1

    s = cluster.stats()
    print(f"ready: {s['ready']}/{len(s['replicas'])} replicas, "
          f"load cap {s['load_cap']}", flush=True)
    print(f"  curl -N http://{args.addr}/generate -H 'content-type: application/json' \\")
    print('       -d \'{"prompt":"Once upon a time","max_tokens":60}\'')
    print("ctrl-c to stop", flush=True)

    try:
        signal.pause() if hasattr(signal, "pause") else _sleep_forever()
    except KeyboardInterrupt:
        pass
    finally:
        cluster.stop()
    return 0


def _sleep_forever() -> None:
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    raise SystemExit(main())
