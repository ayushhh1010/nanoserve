"""Chaos harness: break the cluster on purpose, measure what the client saw.

Everything here measures from *outside* the router, and that is the whole
methodological point. The router retries, migrates in-flight requests onto
replacement replicas, and replays already-delivered tokens as context -- all of
which are supposed to be invisible. A harness that counted retries as failures
would report a disaster; one that inspected router internals would grade the
system on its own homework. The only honest question is what a client holding a
socket actually received.

"Zero dropped requests" therefore means: no client saw an error, a truncated
stream, or a duplicated token. Not "no replica died".
"""

from __future__ import annotations

import json
import os
import signal
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

#: Device the chaos replicas run on. An env var rather than a plain default
#: because CI has no GPU and the scenarios previously hardcoded "cuda" in two
#: places -- so the suite could only ever run on this one laptop, which is the
#: opposite of what a chaos suite is for. Defaults to cuda so a developer run
#: is unchanged.
DEVICE = os.environ.get("NANOSERVE_DEVICE", "cuda")


@dataclass
class RequestOutcome:
    """What one client observed. The unit of every assertion in this file."""

    ok: bool
    started: float
    #: Time to first token. The number a user feels; a mean latency that hides
    #: a 4-second TTFT behind fast tokens is not measuring the experience.
    ttft: float | None = None
    total: float | None = None
    tokens: int = 0
    text: str = ""
    error: str = ""
    status: int = 0


@dataclass
class LoadStats:
    total: int = 0
    ok: int = 0
    failed: int = 0
    errors: dict[str, int] = field(default_factory=dict)
    ttft_p50: float = 0.0
    ttft_p99: float = 0.0
    latency_p50: float = 0.0
    latency_p99: float = 0.0

    @property
    def drop_rate(self) -> float:
        return 0.0 if self.total == 0 else self.failed / self.total


def percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile.

    Not statistics.quantiles: that interpolates, and an interpolated p99 over a
    few hundred samples invents a value between two real observations. For tail
    latency the real observation is the point.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, int(round(q * len(ordered) + 0.5)) - 1))
    return ordered[k]


def summarise(outcomes: list[RequestOutcome]) -> LoadStats:
    stats = LoadStats(total=len(outcomes))
    errors: dict[str, int] = {}
    ttfts, totals = [], []
    for o in outcomes:
        if o.ok:
            stats.ok += 1
            if o.ttft is not None:
                ttfts.append(o.ttft)
            if o.total is not None:
                totals.append(o.total)
        else:
            stats.failed += 1
            key = (o.error or f"http {o.status}")[:80]
            errors[key] = errors.get(key, 0) + 1
    stats.errors = errors
    stats.ttft_p50 = percentile(ttfts, 0.50)
    stats.ttft_p99 = percentile(ttfts, 0.99)
    stats.latency_p50 = percentile(totals, 0.50)
    stats.latency_p99 = percentile(totals, 0.99)
    return stats


def one_request(
    addr: str, prompt: str, max_tokens: int = 48, timeout: float = 60.0
) -> RequestOutcome:
    """Issue one SSE generation and consume it to completion."""
    started = time.perf_counter()
    body = json.dumps({"prompt": prompt, "max_tokens": max_tokens}).encode()
    req = urllib.request.Request(
        f"http://{addr}/generate", data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    out = RequestOutcome(ok=False, started=started)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            out.status = resp.status
            event = None
            for raw in resp:
                line = raw.decode("utf-8", "replace").rstrip("\r\n")
                if line.startswith("event:"):
                    event = line[6:].strip()
                    continue
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if event == "error":
                    out.error = payload[:200]
                    return out
                if event == "rejected":
                    # An explicit rejection is the server working as designed
                    # under overload, not a drop. Counted separately: admission
                    # control refusing work is the opposite of losing it.
                    out.error = f"rejected: {payload[:120]}"
                    return out
                if event == "token":
                    if out.ttft is None:
                        out.ttft = time.perf_counter() - started
                    try:
                        out.text += json.loads(payload).get("text", "")
                    except json.JSONDecodeError:
                        pass
                    out.tokens += 1
                elif event == "done":
                    out.ok = True
                    out.total = time.perf_counter() - started
                    return out
            # The stream ended without a terminal event: the connection was
            # cut mid-generation. A client cannot tell that from truncation,
            # so neither will we.
            out.error = "stream ended without done event"
            return out
    except urllib.error.HTTPError as exc:
        out.status = exc.code
        out.error = f"http {exc.code}"
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        out.error = f"{type(exc).__name__}: {exc}"[:120]
    return out


class LoadDriver:
    """Constant-concurrency load, running until told to stop.

    Closed-loop (each worker waits for its response before sending the next)
    rather than open-loop. Under an injected fault an open-loop driver keeps
    firing into a cluster that cannot answer and measures its own backlog;
    closed-loop measures what a fixed population of clients experiences, which
    is what the assertions are about.
    """

    def __init__(
        self, addr: str, concurrency: int = 16,
        prompts: list[str] | None = None, max_tokens: int = 48,
    ) -> None:
        self.addr = addr
        self.concurrency = concurrency
        self.max_tokens = max_tokens
        self.prompts = prompts or default_prompts()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.outcomes: list[RequestOutcome] = []
        self._threads: list[threading.Thread] = []

    def _worker(self, worker_id: int) -> None:
        i = worker_id
        while not self._stop.is_set():
            prompt = self.prompts[i % len(self.prompts)]
            outcome = one_request(self.addr, prompt, self.max_tokens)
            with self._lock:
                self.outcomes.append(outcome)
            i += self.concurrency

    def start(self) -> None:
        self._stop.clear()
        for w in range(self.concurrency):
            t = threading.Thread(target=self._worker, args=(w,), daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self, timeout: float = 90.0) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=timeout / max(1, len(self._threads)))
        self._threads.clear()

    def mark(self) -> int:
        """Index into outcomes right now, to slice phases apart afterwards."""
        with self._lock:
            return len(self.outcomes)

    def slice(self, start: int, end: int | None = None) -> list[RequestOutcome]:
        with self._lock:
            return list(self.outcomes[start:end])


def default_prompts() -> list[str]:
    """A Zipfian-ish prompt mix: one hot system prompt, plus a tail.

    Uniform prompts would spread perfectly across replicas and make both prefix
    caching and hot-spotting invisible -- so a chaos run on uniform traffic
    would be exercising a routing problem the router never faces.
    """
    system = (
        "You are a careful assistant. Answer precisely and admit uncertainty "
        "rather than guessing. Keep responses grounded in what was asked. "
    )
    tail = [
        "Explain why the sky appears blue.",
        "Summarise how a hash table works.",
        "What is the capital of France?",
        "Describe gradient descent in one paragraph.",
        "Write a haiku about winter.",
    ]
    prompts = [system + "Tell me a short story about a lighthouse."] * 12
    prompts += [system + q for q in tail]
    prompts += tail
    return prompts


class Cluster:
    """A router plus replicas, discovered through etcd."""

    def __init__(
        self,
        replicas: int = 2,
        etcd: str = "127.0.0.1:2379",
        #: 6380, not 6379. This machine already has a Redis inside WSL that
        #: owns Windows localhost:6379 through wslrelay.exe, so the chaos
        #: container cannot bind it -- and, worse, `docker stop` on the
        #: container leaves the port answering PING from WSL's Redis. The
        #: redis_down scenario passed for a while against a Redis it had never
        #: started and could not stop, which is the failure mode that makes a
        #: chaos suite actively misleading: a green result for an outage that
        #: never happened.
        redis: str = "",
        router_addr: str = "127.0.0.1:8080",
        base_port: int = 9101,
        device: str = DEVICE,
        kv_budget_mb: int = 192,
        lease_ttl: int = 5,
        #: Small on purpose. A 27M model on this GPU will happily batch 64
        #: sequences, so at any concurrency this harness can drive, nothing
        #: ever waits and queue depth is identically zero -- which makes the
        #: queue-depth autoscaler look broken when it is reading the truth.
        #: A production-sized model has a batch ceiling in this range, so
        #: lowering it here reproduces the regime the signal was designed for
        #: rather than inventing load.
        max_batch_size: int = 8,
        #: Generous by default so the limiter is not the binding constraint in
        #: scenarios that are not about rate limiting.
        rpm: float = 1_000_000,
        tpm: float = 1_000_000_000,
        request_burst: float = 100_000,
        token_burst: float = 100_000_000,
        quiet: bool = True,
    ) -> None:
        self.n = replicas
        self.etcd = etcd
        self.redis = redis
        self.router_addr = router_addr
        self.base_port = base_port
        self.device = device
        self.kv_budget_mb = kv_budget_mb
        self.lease_ttl = lease_ttl
        self.max_batch_size = max_batch_size
        self.rpm = rpm
        self.tpm = tpm
        self.request_burst = request_burst
        self.token_burst = token_burst
        self.quiet = quiet
        self.replica_procs: list[subprocess.Popen | None] = []
        self.router: subprocess.Popen | None = None

    # -- lifecycle ----------------------------------------------------------

    def start(self, ready_timeout: float = 300.0) -> None:
        sink = subprocess.DEVNULL if self.quiet else None
        for i in range(self.n):
            self.replica_procs.append(self._spawn_replica(i, sink))

        router_bin = ROOT / "router" / "router.exe"
        if not router_bin.exists():
            router_bin = ROOT / "router" / "router"
        if not router_bin.exists():
            raise SystemExit("router binary not built: cd router && go build ./cmd/router")

        argv = [
            str(router_bin), "--addr", self.router_addr, "--etcd", self.etcd,
            "--health-interval", "1s", "--health-timeout", "1s",
        ]
        if self.redis:
            argv += [
                "--redis", self.redis,
                "--rpm", str(self.rpm), "--tpm", str(self.tpm),
                "--request-burst", str(self.request_burst),
                "--token-burst", str(self.token_burst),
            ]
        self.router = subprocess.Popen(argv, cwd=ROOT, stdout=sink, stderr=sink)

        if not self.wait_ready(self.n, timeout=ready_timeout):
            self.stop()
            raise SystemExit(
                f"cluster never reached {self.n} ready replicas; "
                f"check that etcd is up at {self.etcd}"
            )

    def _spawn_replica(self, index: int, sink) -> subprocess.Popen:
        port = self.base_port + index
        return subprocess.Popen(
            [sys.executable, "-u", str(ROOT / "scripts" / "serve_replica.py"),
             "--port", str(port), "--replica-id", f"replica-{index}",
             "--device", self.device, "--kv-budget-mb", str(self.kv_budget_mb),
             "--etcd", self.etcd, "--advertise", f"127.0.0.1:{port}",
             "--lease-ttl", str(self.lease_ttl),
             "--max-batch-size", str(self.max_batch_size)],
            cwd=ROOT, stdout=sink, stderr=sink, **_process_group(),
        )

    def stop(self) -> None:
        if self.router is not None and self.router.poll() is None:
            self.router.terminate()
            try:
                self.router.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.router.kill()
        for p in self.replica_procs:
            if p is not None and p.poll() is None:
                p.terminate()
        for p in self.replica_procs:
            if p is None:
                continue
            try:
                p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                p.kill()

    # -- faults -------------------------------------------------------------

    def sigkill_replica(self, index: int) -> None:
        """Hard kill: the machine-loses-power case, no cleanup, no deregister.

        Recovery here rests entirely on the lease expiring, because nothing
        ran to announce the death.
        """
        p = self.replica_procs[index]
        if p is not None and p.poll() is None:
            p.kill()

    def sigterm_replica(self, index: int) -> None:
        """Graceful stop: the deploy case. The replica revokes its lease and
        drains, so the router should see it leave before it stops answering.

        Not Popen.terminate() on Windows. There, terminate() calls
        TerminateProcess, which is a hard kill with no signal delivered and no
        cleanup handler run -- so a "graceful shutdown" test written that way
        silently measures SIGKILL instead, and reports the graceful path as
        broken. The first run of this suite did exactly that: 539 of 715
        requests failed a scenario that was never actually graceful.
        CTRL_BREAK_EVENT is the Windows signal that a process group can catch,
        and Python raises it as KeyboardInterrupt in the main thread.
        """
        p = self.replica_procs[index]
        if p is None or p.poll() is not None:
            return
        if os.name == "nt":
            p.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            p.send_signal(signal.SIGTERM)

    def restart_replica(self, index: int) -> None:
        sink = subprocess.DEVNULL if self.quiet else None
        self.replica_procs[index] = self._spawn_replica(index, sink)

    # -- observation --------------------------------------------------------

    def stats(self, timeout: float = 5.0) -> dict:
        with urllib.request.urlopen(
            f"http://{self.router_addr}/stats", timeout=timeout
        ) as r:
            return json.loads(r.read())

    def ready_count(self) -> int:
        try:
            return int(self.stats().get("ready", 0))
        except (urllib.error.URLError, OSError, ValueError, KeyError):
            return 0

    def wait_ready(self, want: int, timeout: float = 300.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.ready_count() >= want:
                return True
            time.sleep(1.0)
        return False

    def wait_until(self, predicate, timeout: float = 60.0, poll: float = 0.5) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if predicate():
                return True
            time.sleep(poll)
        return False


def _process_group() -> dict:
    """Spawn kwargs that make a graceful signal deliverable.

    CREATE_NEW_PROCESS_GROUP is required before CTRL_BREAK_EVENT can be sent to
    a child on Windows; without it the event goes to the whole console group
    and takes the test runner down with the replica.
    """
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {}


def docker(*args: str, check: bool = True) -> str:
    proc = subprocess.run(
        ["docker", *args], capture_output=True, text=True,
    )
    if check and proc.returncode != 0:
        raise RuntimeError(f"docker {' '.join(args)}: {proc.stderr.strip()}")
    return proc.stdout.strip()


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


class FaultProxy:
    """A TCP relay in front of a replica, with faults you can switch on.

    Needed because the two most interesting network faults cannot be produced
    portably any other way. A partition is not a crash: the process stays up,
    keeps renewing its etcd lease, and remains in the registry while being
    unreachable -- so the router must evict it on *health*, not on membership,
    and that distinction is exactly what the scenario is testing. Likewise a
    slow replica is not a dead one; it answers everything, just late, which is
    the case bounded-load routing exists to drain.

    SIGSTOP would do it on Linux and does not exist on Windows; iptables and
    `docker network disconnect` need root or containers the replicas are not
    in. Relaying bytes needs neither and behaves identically for gRPC, which
    is HTTP/2 over a plain socket.

    Modes: "pass" (transparent), "delay" (every chunk held), "blackhole"
    (connections accepted and then ignored -- deliberately not refused, since
    a refusal is instant and a partition is a timeout, and code that handles
    one often mishandles the other).
    """

    def __init__(self, listen_port: int, target: tuple[str, int]) -> None:
        self.listen_port = listen_port
        self.target = target
        self.mode = "pass"
        self.delay_ms = 0.0
        self._sock: "socket.socket | None" = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._conns: list = []

    def start(self) -> None:
        import socket

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", self.listen_port))
        self._sock.listen(128)
        self._sock.settimeout(0.5)
        self._thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._thread.start()

    def _accept_loop(self) -> None:
        import socket

        while not self._stop.is_set():
            try:
                client, _ = self._sock.accept()
            except (TimeoutError, OSError):
                continue
            if self.mode == "blackhole":
                # Held open and ignored. Closing would surface as a fast
                # "connection reset", which a client retries immediately; a
                # real partition makes it wait for a timeout instead.
                self._conns.append(client)
                continue
            try:
                upstream = socket.create_connection(self.target, timeout=5)
            except OSError:
                client.close()
                continue
            self._conns += [client, upstream]
            threading.Thread(target=self._pump, args=(client, upstream), daemon=True).start()
            threading.Thread(target=self._pump, args=(upstream, client), daemon=True).start()

    def _pump(self, src, dst) -> None:
        try:
            while not self._stop.is_set():
                if self.mode == "blackhole":
                    return
                data = src.recv(65536)
                if not data:
                    break
                if self.mode == "delay" and self.delay_ms > 0:
                    time.sleep(self.delay_ms / 1000.0)
                dst.sendall(data)
        except OSError:
            pass
        finally:
            for s in (src, dst):
                try:
                    s.close()
                except OSError:
                    pass

    def set_mode(self, mode: str, delay_ms: float = 0.0) -> None:
        self.mode = mode
        self.delay_ms = delay_ms

    def stop(self) -> None:
        self._stop.set()
        for c in self._conns:
            try:
                c.close()
            except OSError:
                pass
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=3)
