"""Replica self-registration in etcd, over the v3 HTTP/JSON gateway.

A replica announces itself under a *lease* and renews it. If the process dies
-- SIGKILL, a panic, a partition, a machine that simply stops -- renewals stop
and etcd deletes the key when the TTL expires. Nothing has to notice the death
and clean up, which is exactly the step that gets skipped in a crash. This is
the whole reason discovery uses a lease rather than a config file.

Why the HTTP gateway rather than a Python etcd client
-----------------------------------------------------
etcd exposes its full v3 API as JSON over HTTP at ``/v3/*`` (verified against
etcd 3.7.1). The Python clients -- ``python-etcd3``, ``etcd3gw``, ``etcd3-py``
-- are thin wrappers over that same surface, and the gRPC ones pin their own
``grpcio``, which this process cannot accept: it already runs a ``grpc.aio``
server on a pinned grpcio, and two different grpcio requirements in one
environment is a resolver fight with no winner. Four HTTP calls with
``urllib`` cost fewer lines than the dependency does, and add nothing to
resolve.

A trap worth naming: a keepalive on a lease that has already expired does not
fail. etcd answers 200 with the lease ID echoed back and **no TTL field**. A
caller that only catches exceptions will believe it is registered forever
while being invisible to the router -- alive, healthy, and receiving nothing.
Renewal is therefore checked by reading the returned TTL, not by the absence
of an error.
"""

from __future__ import annotations

import base64
import json
import random
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable

#: Must match RegistryPrefix in router/etcd.go. The router watches this range.
REGISTRY_PREFIX = "/nanoserve/replicas/"


class EtcdError(RuntimeError):
    """Any failure talking to etcd. Callers retry; they do not distinguish."""


def _b64(s: str) -> str:
    return base64.b64encode(s.encode()).decode()


def _unb64(s: str) -> str:
    return base64.b64decode(s).decode()


def _range_end(prefix: str) -> str:
    """The etcd range_end that makes a Range cover every key under `prefix`.

    etcd has no "prefix" flag on the wire; a prefix scan is the half-open
    range [prefix, prefix+1) where +1 increments the last byte. Getting this
    wrong reads one key instead of the whole registry, which presents as
    "only one replica ever registers".
    """
    raw = bytearray(prefix.encode())
    for i in range(len(raw) - 1, -1, -1):
        if raw[i] < 0xFF:
            raw[i] += 1
            return base64.b64encode(bytes(raw[: i + 1])).decode()
        raw[i] = 0
    return _b64("\0")


class EtcdClient:
    """Just enough of the etcd v3 API for service discovery."""

    def __init__(self, endpoints: list[str], timeout: float = 5.0) -> None:
        if not endpoints:
            raise ValueError("at least one etcd endpoint required")
        self._endpoints = [self._normalise(e) for e in endpoints]
        self._timeout = timeout
        self._i = 0

    @staticmethod
    def _normalise(endpoint: str) -> str:
        endpoint = endpoint.strip().rstrip("/")
        if "://" not in endpoint:
            endpoint = "http://" + endpoint
        return endpoint

    def _post(self, path: str, body: dict) -> dict:
        """POST to the first endpoint that answers, rotating on failure.

        Rotation is the client-side half of etcd's availability story: the
        cluster survives losing a member, but only if clients try another one.
        """
        errors = []
        for attempt in range(len(self._endpoints)):
            idx = (self._i + attempt) % len(self._endpoints)
            base = self._endpoints[idx]
            req = urllib.request.Request(
                base + path,
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                    raw = resp.read().decode()
            except (urllib.error.URLError, OSError, TimeoutError) as exc:
                errors.append(f"{base}: {exc}")
                continue
            # Stream endpoints (keepalive) answer newline-delimited JSON and
            # wrap each frame in {"result": ...}. One frame is all we send and
            # all we need.
            self._i = idx
            first = raw.strip().split("\n")[0] if raw.strip() else "{}"
            try:
                parsed = json.loads(first)
            except json.JSONDecodeError as exc:
                raise EtcdError(f"{path}: bad JSON from {base}: {exc}") from exc
            return parsed.get("result", parsed)
        joined = "; ".join(errors)
        raise EtcdError(f"{path}: no endpoint answered ({joined})")

    def grant(self, ttl_seconds: int) -> str:
        resp = self._post("/v3/lease/grant", {"TTL": str(ttl_seconds)})
        lease = resp.get("ID")
        if not lease or lease == "0":
            raise EtcdError(f"lease grant refused: {resp}")
        return lease

    def put(self, key: str, value: str, lease: str | None = None) -> None:
        body = {"key": _b64(key), "value": _b64(value)}
        if lease is not None:
            body["lease"] = str(lease)
        self._post("/v3/kv/put", body)

    def keepalive(self, lease: str) -> bool:
        """Renew. False means the lease is gone and must be re-granted.

        Reads the TTL rather than trusting the absence of an error: etcd
        answers a keepalive for an expired lease with 200 and no TTL.
        """
        resp = self._post("/v3/lease/keepalive", {"ID": str(lease)})
        return int(resp.get("TTL", 0)) > 0

    def revoke(self, lease: str) -> None:
        self._post("/v3/kv/lease/revoke", {"ID": str(lease)})

    def get_prefix(self, prefix: str) -> dict[str, str]:
        resp = self._post(
            "/v3/kv/range", {"key": _b64(prefix), "range_end": _range_end(prefix)}
        )
        return {
            _unb64(kv["key"]): _unb64(kv.get("value", ""))
            for kv in resp.get("kvs", [])
        }


@dataclass
class RegistrationConfig:
    endpoints: list[str]
    replica_id: str
    addr: str
    #: Detection time for a dead replica is bounded by this. Shorter finds
    #: failures faster and costs more renewal traffic; 10s with renewals every
    #: 1/3 TTL tolerates two consecutive lost renewals before the router drops
    #: the replica, which keeps a GC pause or a slow health check from
    #: deregistering a machine that is actually fine.
    ttl_seconds: int = 10


class ReplicaRegistration:
    """Keeps one replica present in etcd for as long as the process lives.

    Self-healing on purpose. If etcd restarts, or a partition outlasts the
    TTL, the lease is gone and a plain renewal loop would log an error and
    give up -- leaving a healthy replica running, serving nothing, until a
    human notices. That is the worst failure mode available here: a
    control-plane blip permanently removing data-plane capacity. So a lost
    lease re-announces with backoff instead.
    """

    def __init__(
        self,
        cfg: RegistrationConfig,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.cfg = cfg
        self._log = log or (lambda m: print(m, flush=True))
        self._client = EtcdClient(cfg.endpoints)
        self._key = REGISTRY_PREFIX + cfg.replica_id
        self._lease: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        #: Re-announcements since start. Asserted on by the chaos tests, which
        #: otherwise cannot tell self-healing from never having broken.
        self.reannounce_count = 0

    def start(self) -> None:
        """Announce, then renew in the background.

        The first announce is synchronous and raises. A replica that cannot
        register is useless -- no traffic will ever reach it -- so it should
        fail loudly at startup rather than run as a silent orphan.
        """
        self._announce()
        self._thread = threading.Thread(
            target=self._renew_forever, name="etcd-keepalive", daemon=True
        )
        self._thread.start()

    def _announce(self) -> None:
        lease = self._client.grant(self.cfg.ttl_seconds)
        self._client.put(self._key, self.cfg.addr, lease=lease)
        self._lease = lease
        self._log(
            f"registered {self.cfg.replica_id} -> {self.cfg.addr} "
            f"(lease {lease}, ttl {self.cfg.ttl_seconds}s)"
        )

    def _renew_forever(self) -> None:
        # A third of the TTL: two renewals may be lost before the key expires.
        interval = max(1.0, self.cfg.ttl_seconds / 3.0)
        backoff = 0.5
        while not self._stop.wait(interval):
            try:
                alive = self._lease is not None and self._client.keepalive(self._lease)
            except EtcdError as exc:
                alive = False
                self._log(f"etcd keepalive failed: {exc}")

            if alive:
                backoff = 0.5
                continue

            # The lease is gone: etcd restarted, or we were unreachable for
            # longer than the TTL. The router has already dropped us. Get back.
            while not self._stop.is_set():
                try:
                    self._announce()
                    self.reannounce_count += 1
                    backoff = 0.5
                    break
                except EtcdError as exc:
                    self._log(f"re-register failed, retrying in {backoff:.1f}s: {exc}")
                    # Jitter: every replica lost etcd at the same instant, and
                    # a synchronised retry storm is what keeps etcd down.
                    if self._stop.wait(backoff * (0.5 + random.random())):
                        return
                    backoff = min(backoff * 2, float(self.cfg.ttl_seconds))

    def stop(self) -> None:
        """Deregister immediately.

        Revoking beats waiting for the TTL: a planned shutdown removes the
        replica from rotation in milliseconds instead of seconds, so a rolling
        deploy drops nothing. The TTL is the fallback for deaths that never
        get to run cleanup -- it is not the normal path.
        """
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        if self._lease is not None:
            try:
                self._client.revoke(self._lease)
                self._log(f"deregistered {self.cfg.replica_id}")
            except EtcdError as exc:
                self._log(f"revoke failed, lease will expire on TTL: {exc}")
            self._lease = None


def lookup(endpoints: list[str]) -> dict[str, str]:
    """Current registry contents as {replica_id: addr}. Used by tests and CLI."""
    raw = EtcdClient(endpoints).get_prefix(REGISTRY_PREFIX)
    return {k[len(REGISTRY_PREFIX):]: v for k, v in raw.items()}
