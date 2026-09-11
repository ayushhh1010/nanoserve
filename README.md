# Nanoserve

A 27M-parameter language model trained from scratch, an inference engine built
to serve it, and a distributed serving layer to run many of them — written
without an LLM serving framework, so that every mechanism that usually arrives
as a library call is something here that had to be made to work.

No vLLM, no TGI, no Ray. The transformer, the BPE tokenizer, the paged KV
cache, the continuous-batching scheduler, the prefix cache, the router, and the
autoscaler are all in this repository.

```bash
docker compose -f deploy/docker-compose.yml up --build
curl -N localhost:8080/generate -H 'content-type: application/json' \
     -d '{"prompt":"Once upon a time","max_tokens":60}'
```

```
, there was a little girl named Lily. She had a big, soft cushion that she
loved very much. It was...
```

---

## The measurements

Everything below was measured on an RTX 3050 Laptop — **4 GB VRAM**, Windows,
WDDM driver model. Where that hardware distorts a number, the writeups say so,
including the one place where it distorts a number in this project's favour.

### Phase 2 — serving one GPU well

| | before | after |
|---|---|---|
| throughput (concurrent) | 57 tok/s | **902 tok/s** — 15.8× |
| KV cache waste | 72.4% | **3.9%** |
| KV utilization | 27.6% | **96.1%** |
| goodput (met SLO) | 10% | **100%** |
| INT8 perplexity change | — | **0.02%** for 1.69× compression |

The speedup is quoted as **15.8×, not 27.7×**. The baseline runs ~30× longer on
the same workload and thermally throttles, so its median is depressed relative
to its best; comparing against its best is the conservative choice and agrees
with an independent warm single-run measurement of 15.3×.

### Phase 3 — serving many GPUs as one service

Load distribution over 2000 Zipfian requests across 8 replicas:

| | max on one replica | min |
|---|---|---|
| plain consistent hashing | **1022** (51% of traffic) | **0** |
| bounded loads (ε = 0.25) | 313 | 31 |

Chaos suite, **7/7 passing** against a real cluster with real traffic:

| Scenario | Result |
|---|---|
| SIGKILL a replica under load | 434 requests, **0 failed** |
| Graceful drain | 189 requests, **0 failed**; p99 TTFT *improved* 0.72 → 0.10 s |
| Network partition | evicted on health while the lease was still valid, **0 failed** |
| Redis down | **0 failed**, 54 admissions flagged degraded |
| etcd down | 139 requests through the outage, **0 failed** |
| 5× traffic spike | target 1 → 3 → 4 → 3, **0 direction changes** |
| Slow replica (+500 ms) | load share 0.42 → **0.00**, never evicted |

And on the containerised stack, killing a replica mid-flight under sustained
load: **1,034 requests, 0 failed**, 584 of them after the kill.

> **On the honest version of "p99 barely moves":** it does when there is spare
> capacity. Killing one of two CPU replicas removes half the fleet, and TTFT
> p99 rose from 232 ms to 1.5 s while every request still completed. The
> durable claim is **zero dropped requests**, not a flat latency line.

---

## How it fits together

```
clients ──SSE──▶ Go router (N) ──gRPC──▶ replicas (Python, paged KV + prefix cache)
                      │                        │
                      │                        └── lease ──┐
                      └── consistent hashing               ▼
                          + bounded loads               etcd  ◀── autoscaler (2, 1 leader)
                          + retry & in-flight migration  │
                                                      Redis (2-D rate limits, Lua)
```

The data plane is Python because it holds the model. The control plane is Go
because it is almost entirely concurrent I/O, which is the workload where
goroutines beat an event loop with a GIL behind it.

---

## The bugs worth reading about

This is the part a reader learns the most from, so the writeups keep them
rather than tidying them away.

**The hash was catastrophically broken.** 390 of 400 keys on one replica, five
replicas getting nothing. Raw FNV-1a has weak avalanche on short similar
inputs, and every input here is short and similar. The ring passed every
correctness test — right point count, right lookups, deterministic placement —
and simply routed everything to one backend, which in production reads as a
capacity problem. Fixed with MurmurHash3's `fmix64`.

**Failover never worked.** The retry path picked a replica, saw it was one it
had already tried, put it back, and picked again — but the hash is
deterministic, so it got the same answer every time, then reported that no
replica was available while the rest of a healthy cluster sat idle. Killing a
replica under load dropped **539 of 715 requests**. Every unit test had passed
for the life of the code, because none ever made the first choice fail.

**An etcd outage was a total outage.** Stopping etcd made it broadcast
lease-expiry deletes on the way down; the router believed them, emptied its
ring, and failed 26,437 requests while two healthy replicas answered every
health check. The rule that fixes it: the registry says which replicas *should*
serve, health says which *can*, and when they disagree about removing capacity,
believe the one with direct evidence.

**A chaos scenario that passed while testing nothing.** This machine runs a
Redis inside WSL that owns `localhost:6379`, so `docker stop` on the chaos
container left the port answering `PING`. `redis_down` was green against an
outage that never happened — the most dangerous state a chaos suite can be in.

**A quantization test that measured noise.** It asserted logits barely move and
saw 96% change. The model under test was randomly initialized, so its logits
*are* noise, and two random high-dimensional vectors differ by ~141% on average.
Rewritten to assert weight reconstruction error and perplexity on a trained
checkpoint.

**33 tok/s at 1% of roofline.** Not a synchronization stall, as first suspected
— `set_sync_debug_mode("error")` ruled that out. It was 701 kernel launches per
decode step at ~12.9 µs each under WDDM: 96% of the time spent launching work
rather than doing it.

---

## Layout

| | |
|---|---|
| `model/` | transformer, byte-level BPE tokenizer, training loop |
| `engine/` | paged KV cache, scheduler, prefix cache, admission, quantization, gRPC replica, etcd registration |
| `router/` | Go control plane: ring, client pool, health, proxy, autoscaler, rate limiter |
| `bench/` | benchmark harness, workload generation, chaos suite |
| `deploy/` | Dockerfiles, Compose, Kubernetes, Prometheus, Grafana |
| `writeups/` | [training](writeups/01-training.md) · [serving](writeups/02-serving.md) · [distributed](writeups/03-distributed.md) |

---

## Running it

```bash
# Infrastructure
docker run -d --name nanoserve-etcd -p 2379:2379 registry.k8s.io/etcd:3.7.1-0 \
  etcd --name n1 --listen-client-urls http://0.0.0.0:2379 \
       --advertise-client-urls http://127.0.0.1:2379 --data-dir /tmp/etcd
docker run -d --name nanoserve-redis -p 6380:6379 redis:8-alpine

# Tests — 294 Python, 43 Go
pytest
cd router && NANOSERVE_ETCD=127.0.0.1:2379 NANOSERVE_REDIS=127.0.0.1:6380 go test ./...

# The chaos suite (needs a GPU and the infrastructure above)
python scripts/chaos.py

# Everything containerised, with Prometheus and Grafana
docker compose -f deploy/docker-compose.yml up --build
#   router     localhost:8080
#   prometheus localhost:9090
#   grafana    localhost:3000   -> dashboard "Nanoserve — Router"

# Kubernetes
kind create cluster --name nanoserve
kubectl apply -f deploy/k8s/nanoserve.yaml
```

`registry.k8s.io/etcd`, not `gcr.io/etcd-development` or `quay.io/etcd` — both
were deprecated at v3.7. The tag needs the `-0` suffix; plain `v3.7.1` does not
exist there.
