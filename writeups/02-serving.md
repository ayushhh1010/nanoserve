# Phase 2 — The Inference Serving Engine

Phase 1 produced a 27M-parameter model that generates text. Phase 2 is about
the gap between *a model that works* and *a server that works*: the same GPU,
the same weights, and 15.8× the throughput, because the bottleneck was never
the matrix multiplies.

Hardware throughout: RTX 3050 Laptop, 4 GB VRAM, Windows/WDDM. Where that
distorts a measurement, it is called out — in one case it distorts the headline
number in the *flattering* direction, and that is stated too.

---

## 1. The thesis: prefill and decode are different programs

A forward pass over a 300-token prompt and a forward pass to produce token 301
run the same code and have nothing else in common.

**Prefill** processes every prompt token at once. It is compute-bound: large
matrix multiplies, high arithmetic intensity, the GPU doing what it is for.

**Decode** processes exactly one token per sequence per step. It is
memory-bound: every weight in the model is read from HBM to produce a single
token, and the arithmetic intensity is roughly 1. The GPU is idle waiting on
memory.

Everything in this phase follows from that split. Batching helps decode
enormously — the weights are read once for the whole batch — and helps prefill
comparatively little. Which is why the headline metric is **tokens/second across
concurrent requests at fixed VRAM**, not single-stream latency.

### The measurement that reframed the project

Single-stream decode ran at **33 tok/s** against a roofline of ~3,500–4,000.
One percent of achievable.

The first suspicion was a host/device synchronization stalling the pipeline. It
was worth checking that the evidence actually supported it — "host launch time
equals wall time" is equally consistent with a sync *and* with the host simply
being the bottleneck, so it proves neither.
`torch.cuda.set_sync_debug_mode("error")` settled it: no sync.

The real cause was **701 kernel launches per decode step at ~12.9 µs each under
WDDM** — Windows' driver model has far higher launch overhead than Linux, and
96% of the step was spent launching work rather than doing it.

That finding reframed everything downstream: this machine is *launch-bound* at
batch 1, which makes the single-stream baseline artificially slow and would
normally **inflate** any batching speedup. The numbers below are therefore
Windows-conservative in interpretation, and the headline benchmarks belong on
Linux.

The investigation also turned up one genuine synchronization — `int(position_ids.max())`
inside the RoPE path, a device-to-host read on every step — which was fixed.

---

## 2. Paged KV cache

A contiguous KV cache must reserve `max_seq_len` per sequence up front, because
it cannot know how long a generation will run. Almost every sequence then stops
far short of that, and the reserved remainder is unusable by anyone else.

Measured against the real allocator on this project's own length distribution
(400 sequences, mean length 339, p99 730, 512 MB budget):

| | contiguous | paged |
|---|---|---|
| mean waste | **72.4%** | **3.9%** |
| mean utilization | 27.6% | 96.1% |
| usable capacity | 1× | **3.5×** |

Published figures for vLLM's paged attention are 60–80% waste and under 4%
after; this lands inside both.

The design is the OS virtual-memory analogy taken seriously: fixed-size blocks,
a block table per sequence, a free list, refcounts, and copy-on-write so two
sequences sharing a prompt share its blocks until one of them diverges.

One implementation detail carried real weight: storage is flat,
`(num_blocks · block_size, heads, head_dim)`, so gather and scatter are single
index operations rather than per-block loops. On a launch-bound machine, the
difference between one kernel and *n* kernels is the difference between working
and not.

The waste accounting itself is broken out into reserved, internal, and external
fragmentation rather than reported as one number, because they have different
fixes and collapsing them hides which one is actually biting.

---

## 3. Continuous batching

Static batching runs a batch to completion, so every sequence waits for the
longest one. With realistic length variance that is most of the batch idling.

Continuous batching schedules at **iteration** granularity: a sequence that
finishes leaves the batch immediately and a waiting request takes its slot on
the very next step.

| | baseline (best) | continuous batching |
|---|---|---|
| throughput | 57 tok/s | **902 tok/s** |
| TTFT p99 | — | **126× lower** |
| goodput (met SLO) | 10% | **100%** |

**The speedup is 15.8×, not 27.7×.** The baseline runs ~30× longer on the same
workload and thermally throttles, so its *median* (32 tok/s) is depressed
relative to its *best* (57 tok/s). Comparing against its best is the
conservative choice, and it agrees with an independent back-to-back warm
single-run measurement of 15.3×. The larger number is available and is not
honest.

Preemption is by recompute rather than swap: when KV runs short, the
lowest-priority sequence is evicted and its prompt re-prefilled later.
Recompute costs GPU time; swapping costs PCIe bandwidth that decode is already
starved for.

---

## 4. Prefix caching

Content-addressed blocks: `hash(parent_hash, block_tokens)` chains so a block's
identity encodes its entire history, which makes sharing safe without comparing
token sequences. LRU eviction, refcount-aware so a block in use is never
reclaimed.

This is where the Zipfian workload matters. With uniformly random prompts,
prefix caching looks useless — so a benchmark on uniform traffic would be
measuring the cache against a workload that cannot show what it fixes.

---

## 5. Admission control and goodput

The distinction that matters: **throughput** counts tokens produced;
**goodput** counts tokens produced *for requests that met their deadline*. A
server thrashing on 200 simultaneous requests can have excellent throughput and
near-zero goodput, because everything finishes late and every result is
worthless.

Admission is a bounded queue plus a deadline projection: if the current queue
implies a request cannot finish in time, reject it *now*, while the rejection is
cheap and honest, rather than accepting it and failing slowly.

Two bugs here are worth recording because both produced 100% rejection and
neither was in the controller's logic:

**Clock mismatch.** `build_workload` emitted *relative* deadlines while the
controller compared against `perf_counter()` — a number in the hundreds of
thousands. Every request looked infinitely overdue. Fixed with a shared
`rebase()` that shifts arrival and deadline together.

**Cold start.** With no history, the controller assumed 50 tok/s against a real
900 and rejected every request in a startup burst. Fixed with a
`min_observations` floor before the projection is trusted.

---

## 6. INT8 weight-only quantization

Round-to-nearest, symmetric, group size 128, `lm_head` skipped.

| | dense | INT8 |
|---|---|---|
| perplexity | 3.2518 | 3.2525 |
| weights | 51.0 MB | 30.2 MB |

**0.02% perplexity change for 1.69× compression** across 56 quantized layers,
measured over 163,840 evaluation tokens.

### The test was measuring the wrong thing

The original test asserted that logits barely move after quantization, and saw
**96% relative change** — an apparent disaster.

The model under test was *randomly initialized*. Its logits are noise, and two
random high-dimensional vectors differ by ~141% on average, so 96% was not a
failure signal at all; it was a number with no meaning. A quantization test on
an untrained model cannot say anything about quality.

Rewritten to assert two things that are actually falsifiable: per-layer weight
reconstruction error against the dense weights, and end-to-end perplexity on a
**trained** checkpoint. The table above is the rewritten test's output.

---

## 7. Observability

Prometheus metrics follow vLLM's naming so dashboards transfer, with two bucket
families (`SECONDS` and `SECONDS_FAST`) because TTFT lives in the sub-second
range where end-to-end request buckets have almost no resolution — and a p99
measured through the wrong buckets is a guess. Tracing follows the OpenTelemetry
GenAI semantic conventions.

---

## 8. Things that turned out to be wrong

**The PRD's claim that a static preallocated KV cache is a big win.** Measured,
it is a wash on this workload. Reported as such rather than quietly dropped.

**`enable_gqa=True` was 20% *slower*.** Fused attention kernels require matching
head counts; with 8 query heads and 2 KV heads the mismatch drops execution to
the unfused MATH backend. Reverted.

**The first roofline measurement (7.9 GB/s) was garbage.** Too-small buffer,
only 5 warmup iterations, and run immediately after profiler teardown. The real
figure is **166 GB/s**, and the difference is the difference between "the
hardware is broken" and "our code is slow".

**`best.pt` pointed at a worse model.** The final evaluation ran *after* the
last checkpoint save and was never compared against `best_val`.

**A KV leak on the exception path.** Both cache pools released blocks only on
the happy path. `try/finally`.

**Three reporting bugs that flattered the results.** `aggregate()` reported the
middle run by *order* rather than by value — given `[100, 300, 200]` it called
300 the median. The printed summary disagreed with the saved JSON whenever
repeats > 1. And `_fanout` was O(n), degrading from 5.7 µs to 203.7 µs as the
run progressed.

**Batching changes bf16 outputs, and that is not a bug.** 5–6 of 24 sequences
differ in bf16; 0 of 24 differ in fp32. Batching changes the attention shape
from `(1, H, 1, T)` to `(B, H, 1, max_len)`, which selects a different kernel
with a different reduction order. Allocator changes, by contrast, *are*
bit-exact — which is the property worth asserting, and is asserted.

---

## 9. Running it

```bash
python scripts/run_suite.py          # the full benchmark matrix
python scripts/measure_kv_waste.py   # the paged-vs-contiguous table
python scripts/measure_quant.py      # the INT8 table
python scripts/serve.py              # single-node SSE server
pytest                               # 326 tests
```
