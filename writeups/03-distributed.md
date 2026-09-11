# Phase 3 — The Distributed Serving Layer

Phase 2 made one GPU serve many requests well. Phase 3 makes many machines
behave like one service: requests land on the replica most likely to already
hold their KV cache, replicas join and leave without anyone editing a config
file, a dying replica costs no requests, and a control-plane outage degrades
the control plane rather than the service.

Everything below was measured on the machine described in
[`01-training.md`](01-training.md) — an RTX 3050 Laptop, 4 GB VRAM, Windows
with the WDDM driver model. Where that hardware distorts a number, it says so.

---

## 1. What the system is

```
                    ┌──────────────────────────────┐
   clients ────────▶│  Go router (N, stateless)    │
     SSE            │  consistent hashing +        │
                    │  bounded loads, retry,       │
                    │  in-flight migration         │
                    └───────┬──────────────────────┘
                            │ gRPC (HTTP/2, streaming)
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
        ┌──────────┐  ┌──────────┐  ┌──────────┐
        │ replica  │  │ replica  │  │ replica  │   Python, Phase 2 engine
        │ paged KV │  │ paged KV │  │ paged KV │   + prefix cache
        └────┬─────┘  └────┬─────┘  └────┬─────┘
             │ lease       │ lease       │ lease
             └─────────────┴─────────────┘
                           ▼
                     ┌───────────┐        ┌───────────┐
                     │   etcd    │        │   Redis   │
                     │ registry  │        │  limits   │
                     │ election  │        └───────────┘
                     └─────┬─────┘
                           │  desired replica count
                     ┌─────▼──────────────┐
                     │ autoscaler (2, one │
                     │ leader at a time)  │
                     └────────────────────┘
```

The split of languages is deliberate. The data plane stays Python because it
holds the model and the Phase 2 engine. The control plane is Go because it is
almost entirely concurrent I/O — thousands of simultaneous streams, health
checks, watches — and that is the workload where goroutines and a real
scheduler beat an event loop that still has a GIL behind it.

---

## 2. Routing: consistent hashing with bounded loads

### Why not round-robin

A request's cost is dominated by prefill, and prefill is skipped entirely when
the replica already holds the prompt's KV blocks. Round-robin destroys that: it
spreads a hot system prompt evenly, so *every* replica prefills it and no
replica's cache is worth anything. Routing by prompt prefix instead means the
same prefix lands on the same replica, which already has it.

### Why not plain consistent hashing

Because real prompt traffic is Zipfian — a handful of system prompts are used
constantly and a long tail almost never. Consistent hashing sends all of a hot
prefix to one replica and has no mechanism to notice that replica is on fire.

Measured on the ring, 2000 Zipfian requests over 8 replicas:

| | max load on one replica | min |
|---|---|---|
| plain consistent hashing | **1022** (51% of all traffic) | **0** |
| bounded loads, ε = 0.25 | 313 | 31 |

One replica taking half the traffic while another takes none is not a tuning
problem; it is the algorithm working as specified.

### The mechanism

Each replica may hold at most

```
cap = ceil(c · total_load / num_replicas),   c = 1 + ε
```

and the clockwise walk skips any replica already at the cap. Most requests
still land on their hashed replica, so locality survives; overflow spills to
the next replica clockwise, which is deterministic and therefore still
cache-friendly for the spill itself.

The ceiling is load-bearing rather than cosmetic. With `floor`, a small total
load gives `cap = 0` and nothing can be placed anywhere. With `ceil`, a valid
assignment provably always exists: `n · ceil(c·m/n) > m` for `c > 1`, so the
pigeonhole argument never fails and the walk always terminates.

Reference: Mirrokni, Thorup, Zadimoghaddam, *Consistent Hashing with Bounded
Loads*, SODA 2018 ([arXiv:1608.01350](https://arxiv.org/abs/1608.01350)).

### The hash was catastrophically broken

With 8 replicas and 1200 ring points, **390 of 400 keys landed on one replica**
and five replicas got nothing at all. Raw FNV-1a has weak avalanche on short,
similar inputs — and every input here is short and similar (`replica-0#0`,
`replica-0#1`, …; prompts sharing a system prompt).

The ring looked perfectly healthy from every angle a test usually checks: right
number of points, correct lookups, deterministic placement. It simply routed
everything to one backend, which in production reads as a capacity problem
rather than a hashing problem. Fixed by running the FNV output through
MurmurHash3's `fmix64` finalizer; distribution became 435–549 per replica over
4000 keys.

**The lesson worth keeping:** a hash function that is "fine" in general can be
useless for a specific key distribution, and a unit test that checks
*correctness* will never catch it. The test that caught it measured *spread*.

---

## 3. Service discovery: etcd leases

A replica writes its own key under a lease and renews it. If the process dies —
SIGKILL, a panic, a partition, a machine that stops — renewals stop and etcd
deletes the key when the TTL expires. Nothing has to notice the death and clean
up, which is exactly the step that gets skipped in a crash.

### Registration lives in Python, over HTTP

etcd exposes its full v3 API as JSON over HTTP at `/v3/*`. The Python etcd
clients (`python-etcd3`, `etcd3gw`, `etcd3-py`) are thin wrappers over that
same surface, and the gRPC ones pin their own `grpcio` — which this process
cannot accept, because it already runs a `grpc.aio` server on a pinned grpcio,
and two grpcio requirements in one environment is a resolver fight with no
winner. Four `urllib` calls cost fewer lines than the dependency does.

**A trap that cost real debugging time:** a keepalive on a lease that has
already expired *does not fail*. etcd answers `200` with the lease ID echoed
back and **no TTL field at all**. Code that only catches exceptions believes it
is registered forever while being invisible to the router — alive, healthy, and
receiving nothing. Renewal is therefore checked by reading the returned TTL.

### Registration order is the whole game

Replicas register **after** the model is loaded and the port is bound, and
deregister **before** draining. Presence in the registry has to mean "can
serve", not "exists".

An earlier version got this wrong in Go: `Ring.Add` marked replicas ready the
moment they were discovered. All 28 requests in the first end-to-end run failed
while three replicas were still loading checkpoints. A registry entry says a
replica *exists*; only a health check says it can serve.

---

## 4. Autoscaling

The algorithm is Kubernetes HPA's, including its measured defaults — tolerance
`0.1`, scale-down stabilization `300s`, scale-up `0s` — because the failure
modes it guards against are the same ones:

```
desired = ceil(current · observed / target)
```

Three deviations, each for an inference-specific reason:

**The metric is queue depth, not utilization.** A GPU at 100% with an empty
queue is working correctly — that is a saturated decode loop, which is exactly
what you want. Scaling on utilization would add replicas forever. Waiting
requests are what a user feels.

**KV utilization is a separate, non-proportional guard.** It is the real
capacity limit of a serving replica and it stays invisible in queue depth until
the moment it isn't: a replica at 97% KV is one long prompt away from preempting
sequences, and by the time that reaches the queue the damage is taken. The guard
raises a floor; it never caps the proportional result.

**Scale-down is extra conservative** because replicas are not fungible. Each
holds a warm prefix cache that cost real prefill to build, and killing one moves
every key it owned to another replica that must prefill them again. A premature
scale-down costs a latency spike on the way down *and* another on the way back
up.

### Leader election, and the half that gets left out

The controller is not idempotent. Two autoscalers observing the same overloaded
cluster each compute the same delta and each apply it; the cluster overshoots
2×, both then see the overshoot and scale down, and the pair oscillates. Running
a single instance is not an answer either — then the autoscaler is a single
point of failure whose death is silent, because nothing scaling is
indistinguishable from nothing needing to scale.

Winning an election is the easy half. **Noticing you have lost one is the half
that gets left out.** A partitioned leader stops renewing its session lease, so
etcd elects someone else — and if the old leader keeps acting, both are scaling
at once, which is the exact failure the election was meant to prevent, now
harder to see because each instance's logs show exactly one leader. So the
controller runs under a context cancelled the moment the session ends.

Verified against real etcd: leadership passed through 3 of 3 candidates and
**never overlapped**.

### A bug the tests found

The collector returned an error whenever no replica was ready, and `Run` skips a
tick on observation failure — sound on its own, since a broken metrics pipeline
must not be read as "load is zero". Together they meant a fleet that had
entirely died made every tick a skipped tick: the one situation most needing a
scale-up was the one situation the autoscaler refused to act on.

The fix is to distinguish the two zeros. **Empty registry** → request the floor;
the fleet is gone and must be rebuilt. **Registered but none healthy** → hold;
this is an outage, not a capacity shortage, and adding machines does not cure a
bad checkpoint or a dead dependency, it multiplies the failure.

---

## 5. Rate limiting: two dimensions, one Lua script

An LLM endpoint has two scarce resources and they do not move together.
Requests per minute bounds connection and scheduling overhead; **tokens** per
minute bounds the GPU. One request generating 4000 tokens costs the cluster far
more than forty generating ten, and a limiter counting only requests waves the
expensive one through. This is why the commercial APIs quote RPM and TPM
separately.

**Both buckets are checked and consumed in one script, all-or-nothing.** Two
round trips would be a correctness bug rather than an inefficiency: consume a
request token, fail the token bucket, and the client has been charged for a
request it was never allowed to make — a leak that only appears under the load
the limit exists for.

**Time comes from `redis.call('TIME')`, never the caller.** Routers do not share
a clock, and a second of NTP skew lets a client aim its burst at whichever
instance runs slow.

**Reserve-then-refund.** The true cost is unknowable in advance, so `max_tokens`
is reserved up front and the remainder returned when the stream ends — on every
exit path, including client disconnect. Charging only on completion would let a
client open a thousand concurrent maximum-length generations before any was
billed. The refund is clamped at capacity so reserve/refund cycles cannot mint
budget.

**It fails open.** Rate limiting is cost control and fairness, not correctness.
Failing closed converts a Redis outage into a total outage of a service that is
otherwise perfectly healthy — trading a billing problem for an availability
problem. Fail-open admissions are flagged `Degraded` and counted, because an
outage where everything succeeds is otherwise invisible.

Verified against real Redis 7.0.15 and 8.10.1: **60 concurrent racers against a
burst of 10 admitted exactly 10.**

---

## 6. Chaos: 7/7, and the three bugs it found

All seven PRD scenarios pass against a real cluster with real traffic. Every
assertion is judged **from outside the router**, on what a client holding a
socket actually received — the router retries, migrates in-flight requests, and
replays delivered tokens as context, all of which are supposed to be invisible.
A harness counting retries would report a disaster; one reading router internals
would grade the system on its own homework.

| Scenario | Result |
|---|---|
| SIGKILL a replica under load | 434 requests, **0 failed**; p99 TTFT 0.56 → 0.12 → 0.17 s |
| Graceful drain (SIGTERM) | 189 requests, **0 failed**; p99 TTFT *improved* 0.72 → 0.10 s |
| Network partition | evicted on health while the lease was still valid; **0 failed** |
| Redis down | 65 requests, 0 failed, **54 flagged degraded** |
| etcd down | 139 requests through the outage, **0 failed**; discovery resumed |
| 5× traffic spike | target 1 → 3 → 4 → 3, **0 direction changes**, 0 failed |
| Slow replica (+500 ms) | load share 0.42 → **0.00**, never evicted, 0 failed |

The absolute request rates are low: a 27M model on a 4 GB laptop GPU does not
reach the PRD's 2000 RPS. The failure *behaviours* — eviction, migration,
degradation, draining — are properties of the control plane and do not depend on
the rate. The throughput numbers do, and are labelled accordingly.

### Bug 1 — failover never worked

`pickUntried` picked a replica, saw it was one already tried, released it, and
picked again. But **`Pick` is deterministic for a given key**: it starts at the
same ring position and returns the same replica every time. The loop got the
identical answer on every iteration, then reported that no replica was available
— while the rest of a healthy cluster sat idle.

Killing one replica under load **dropped 539 of 715 requests**.

Every unit test had passed for the entire life of this code, because no unit
test ever made the first choice fail. Exclusion now happens *inside* the ring
walk, including in the least-loaded fallback, which had its own copy of the bug.

### Bug 2 — an etcd outage was a total outage

Stopping etcd made it broadcast lease-expiry deletes for every replica on the
way down — the leases genuinely *had* lapsed, because replicas cannot renew
against an etcd that is shutting down. The router believed it, emptied its ring,
and failed **26,437 requests** with `no replicas available` while two perfectly
healthy replicas answered every health check put to them.

A control-plane outage had become a total outage, which is the precise inversion
of what a control plane is for.

The fix is a rule worth stating plainly: **the registry says which replicas
*should* serve; health says which *can*. When the two disagree in the direction
of removing capacity, believe the one with direct evidence.** Healthy replicas
the registry has dropped are retained and reaped only once health also fails.
This is the same reasoning as Envoy's panic mode — when discovery claims nearly
everything is gone, the likelier explanation is that discovery is broken.

Retention is surfaced as `orphaned` in `/stats` and `/metrics`, because running
on retained membership is a degraded state and one nobody can see is one nobody
fixes.

### Bug 3 — the graceful path was never tested

On Windows, `Popen.terminate()` calls `TerminateProcess`: a hard kill, no signal
delivered, no cleanup handler run. The "graceful shutdown" scenario had been
silently measuring SIGKILL and reporting the drain as broken. Now it sends
`CTRL_BREAK_EVENT` to a real process group.

### And one scenario that passed while testing nothing

This machine runs a Redis inside WSL that owns Windows `localhost:6379` through
`wslrelay.exe`. `docker stop` on the chaos container left the port answering
`PING`, so `redis_down` was **green against an outage that never happened** —
the most dangerous possible state for a chaos suite. Moved to port 6380, with a
guard that fails the scenario outright if the port still answers after the stop.

It also means an earlier claim of "verified against Redis 8.10.1" was wrong: the
tests had been running against the WSL Redis 7.0.15. They now pass on both.

---

## 7. What I would do differently

**Measure distributions, not just correctness, from the start.** Both the
hashing disaster and the failover bug were invisible to correctness tests and
obvious the moment something measured spread or forced a failure.

**Write the chaos suite earlier.** It found three real bugs in an afternoon,
two of which had been in `main` for days behind a fully green test suite. Unit
tests verify the path you thought about; chaos tests verify the path you didn't.

**Be suspicious of a test that passes immediately.** Two here were worthless:
the compaction test never killed the watcher it claimed to test, and
`redis_down` graded an outage that never occurred. Both were caught only by
asking "what would this look like if it were lying?" — which is now an explicit
assertion in each (`Reestablished() > 0`, `_port_open(6380)`).

**Assumptions about *which layer* recovers are worth testing too.** I assumed my
watch supervisor handled etcd restarts; measured, `clientv3` absorbs a container
restart and even a rebuilt cluster entirely on its own, and the supervisor never
runs. The test that asserted otherwise would have failed on correct code, so the
assertion was dropped rather than kept.

---

## 8. Running it

```bash
# Infrastructure
docker run -d --name nanoserve-etcd -p 2379:2379 registry.k8s.io/etcd:3.7.1-0 \
  etcd --name n1 --listen-client-urls http://0.0.0.0:2379 \
       --advertise-client-urls http://127.0.0.1:2379 --data-dir /tmp/etcd
docker run -d --name nanoserve-redis -p 6380:6379 redis:8-alpine

# Control plane
cd router && go build -o router.exe ./cmd/router && go build -o autoscaler.exe ./cmd/autoscaler

# Everything, containerised, with Prometheus and Grafana
docker compose -f deploy/docker-compose.yml up --build

# The chaos suite
python scripts/chaos.py

# Tests (integration ones skip unless the env vars are set)
cd router && NANOSERVE_ETCD=127.0.0.1:2379 NANOSERVE_REDIS=127.0.0.1:6380 go test ./...
pytest
```

`registry.k8s.io/etcd`, not `gcr.io/etcd-development` or `quay.io/etcd`: both
were deprecated at v3.7 and will not receive 3.8+. The tag carries the `-0`
build-revision suffix that registry uses — plain `v3.7.1` does not exist there
and fails to pull.
