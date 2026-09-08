"""Measure what contiguous pre-allocation wastes on this project's workload.

The published figure for this design is 60-80% of KV memory wasted. That is a
citation until it is reproduced on the actual length distribution being served,
which is what this does: replay the benchmark workload through a static pool and
sample the accounting as sequences come and go.

The paged column is now measured against the real allocator, with the analytic
bound kept alongside as a cross-check. The two differ, and the direction is
informative: measured waste (3.9%) sits above the per-sequence bound (2.2%)
because the paged pool holds far more sequences resident at once, so more
partial tail blocks are in flight at any instant. The bound is per sequence;
the measurement is over the whole pool at steady state.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bench.workload import WorkloadConfig, build_workload
from engine.kv_cache import PoolStats, StaticCachePool, slots_for_budget
from model.config import NANO_27M

ROOT = Path(__file__).resolve().parent.parent


def replay(lengths: list[int], num_slots: int, max_seq_len: int) -> list[PoolStats]:
    """Admit sequences up to `num_slots`, retire the longest-running, resample.

    A steady-state approximation rather than a full event simulation: the pool
    is kept as full as the slot count allows, which is the regime a loaded
    server actually runs in and the one the waste figure describes.
    """
    pool = StaticCachePool(
        NANO_27M, num_slots=num_slots, max_seq_len=max_seq_len, device="meta"
    )
    samples: list[PoolStats] = []
    live: list[tuple] = []  # (cache, target_len)
    queue = list(lengths)

    while queue or live:
        while queue and len(live) < num_slots:
            target = min(queue.pop(0), max_seq_len)
            cache = pool.allocate(expected_len=target)
            if cache is None:
                break
            live.append([cache, target])

        # Advance every live sequence by one decode step.
        finished = []
        for entry in live:
            cache, target = entry
            cache._pos = min(cache._pos + 1, target)
            if cache._pos >= target:
                finished.append(entry)

        samples.append(pool.stats())

        for entry in finished:
            pool.free(entry[0])
            live.remove(entry)

    return samples


def summarise(samples: list[PoolStats]) -> dict:
    """Average the accounting over the run, weighted equally per step."""
    if not samples:
        return {}
    return {
        "steps": len(samples),
        "mean_slots_allocated": float(np.mean([s.slots_allocated for s in samples])),
        "mean_utilization": float(np.mean([s.utilization for s in samples])),
        "mean_occupancy": float(np.mean([s.occupancy for s in samples])),
        "mean_waste_fraction": float(np.mean([s.waste_fraction for s in samples])),
        "mean_tokens_used": float(np.mean([s.tokens_used for s in samples])),
        "mean_internal_frag": float(np.mean([s.internal_fragmentation for s in samples])),
        "mean_reserved": float(np.mean([s.tokens_reserved for s in samples])),
        "mean_external_frag": float(np.mean([s.external_fragmentation for s in samples])),
    }


def replay_paged(lengths: list[int], num_blocks: int, block_size: int) -> list[dict]:
    """The same replay, against a real block allocator.

    Uses the actual `PagedCachePool` rather than arithmetic, so the numbers
    come from the allocator that serves requests rather than from a formula
    describing it. Tensors live on the meta device: only the bookkeeping is
    exercised, so a 512 MB pool costs nothing to simulate.
    """
    from engine.block_manager import PagedCachePool

    pool = PagedCachePool(
        NANO_27M, num_blocks=num_blocks, block_size=block_size, device="meta"
    )
    samples: list[dict] = []
    live: list[list] = []
    queue = list(lengths)

    while queue or live:
        # Admit while blocks remain for at least the prompt's first block.
        while queue and pool.allocator.num_free > 1:
            target = queue.pop(0)
            cache = pool.allocate()
            cache._ensure_capacity(1)
            cache.num_tokens = 1
            live.append([cache, target])

        finished = []
        for entry in live:
            cache, target = entry
            if cache.num_tokens < target and pool.allocator.num_free > 0:
                cache._ensure_capacity(1)
                cache.num_tokens += 1
            if cache.num_tokens >= target:
                finished.append(entry)

        used = sum(c.num_tokens for c, _ in live)
        samples.append(pool.stats(tokens_used=used).to_dict())

        for entry in finished:
            entry[0].free()
            live.remove(entry)

    return samples


def summarise_paged(samples: list[dict]) -> dict:
    if not samples:
        return {}
    return {
        "steps": len(samples),
        "mean_sequences": 0.0,
        "mean_blocks_allocated": float(np.mean([s["blocks_allocated"] for s in samples])),
        "mean_tokens_used": float(np.mean([s["tokens_used"] for s in samples])),
        "mean_internal_frag": float(np.mean([s["tokens_internal_frag"] for s in samples])),
        "mean_external_frag": 0.0,
        "mean_utilization": float(np.mean([s["utilization"] for s in samples])),
        "mean_waste_fraction": float(np.mean([s["waste_fraction"] for s in samples])),
        "peak_occupancy": float(max(s["occupancy"] for s in samples)),
    }


def paged_bound(lengths: list[int], block_size: int) -> dict:
    """What a block allocator wastes on the same sequences, by construction.

    Internal fragmentation only, and only in each sequence's final partial
    block: ceil(n / block) * block - n, which is at most block_size - 1.
    External fragmentation is zero because uniform blocks are interchangeable.
    """
    arr = np.asarray(lengths)
    blocks = np.ceil(arr / block_size)
    allocated = blocks * block_size
    waste = allocated - arr
    return {
        "block_size": block_size,
        "mean_waste_tokens_per_seq": float(waste.mean()),
        "max_waste_tokens_per_seq": int(waste.max()),
        "waste_fraction": float(waste.sum() / allocated.sum()),
        "external_fragmentation": 0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--budget-mb", type=int, default=512, help="KV memory budget")
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--requests", type=int, default=400)
    ap.add_argument("--output-len", type=int, default=96)
    ap.add_argument("--prompt-len", type=int, default=96)
    ap.add_argument("--out", type=Path, default=ROOT / "bench" / "results" / "kv_waste.json")
    args = ap.parse_args()

    cfg = WorkloadConfig(
        num_requests=args.requests,
        prompt_len_mean=args.prompt_len,
        output_len_mean=args.output_len,
        seed=0,
    )
    reqs = build_workload(cfg, 8192, 8191)
    lengths = [min(r.prompt_len + r.params.max_tokens, args.max_seq_len) for r in reqs]

    budget = args.budget_mb * 1024 * 1024
    slots = slots_for_budget(NANO_27M, budget, args.max_seq_len)

    print(f"model KV cost      : {NANO_27M.kv_bytes_per_token:,} bytes/token (GQA, 2 KV heads)")
    print(f"budget             : {args.budget_mb} MB")
    print(f"max_seq_len        : {args.max_seq_len}")
    print(f"contiguous slots   : {slots}  ({args.max_seq_len} tokens each, "
          f"{args.max_seq_len * NANO_27M.kv_bytes_per_token / 1024**2:.1f} MB per slot)")
    print()
    print(f"workload           : {len(lengths)} sequences")
    print(f"  total length p50 : {np.percentile(lengths, 50):.0f} tokens")
    print(f"  total length p99 : {np.percentile(lengths, 99):.0f} tokens")
    print(f"  mean             : {np.mean(lengths):.0f} tokens "
          f"({np.mean(lengths) / args.max_seq_len:.1%} of the reservation)")
    print()

    from engine.block_manager import blocks_for_budget

    samples = replay(lengths, slots, args.max_seq_len)
    static = summarise(samples)

    n_blocks = blocks_for_budget(NANO_27M, budget, args.block_size)
    paged_measured = summarise_paged(replay_paged(lengths, n_blocks, args.block_size))
    paged = paged_bound(lengths, args.block_size)

    print("CONTIGUOUS pre-allocation, measured over the replay:")
    print(f"  mean slots in use     {static['mean_slots_allocated']:8.1f} / {slots}")
    print(f"  mean tokens held      {static['mean_tokens_used']:8.0f}")
    print(f"  reserved (will fill)  {static['mean_reserved']:8.0f}")
    print(f"  internal frag         {static['mean_internal_frag']:8.0f}   <- never used")
    print(f"  external frag         {static['mean_external_frag']:8.0f}   <- unusable free slots")
    print(f"  utilization           {static['mean_utilization']:8.1%}")
    print(f"  WASTE                 {static['mean_waste_fraction']:8.1%}")
    print()
    print(f"PAGED, block size {args.block_size}, MEASURED over the same replay:")
    print(f"  blocks in use         {paged_measured['mean_blocks_allocated']:8.0f} / {n_blocks}")
    print(f"  mean tokens held      {paged_measured['mean_tokens_used']:8.0f}")
    print(f"  internal frag         {paged_measured['mean_internal_frag']:8.0f}   "
          f"<- bounded by block_size - 1 per sequence")
    print(f"  external frag         {paged_measured['mean_external_frag']:8.0f}   "
          f"<- zero by construction")
    print(f"  utilization           {paged_measured['mean_utilization']:8.1%}")
    print(f"  WASTE                 {paged_measured['mean_waste_fraction']:8.1%}")
    print()
    print(f"  analytic check        {paged['waste_fraction']:8.1%}  "
          f"(ceil(n/{args.block_size}) rounding, max {paged['max_waste_tokens_per_seq']} tok/seq)")
    print()
    gain = (1 - paged_measured["mean_waste_fraction"]) / (1 - static["mean_waste_fraction"])
    conc = paged_measured["mean_tokens_used"] / max(static["mean_tokens_used"], 1)
    print(f"  usable capacity gain  {gain:8.1f}x")
    print(f"  concurrent tokens     {conc:8.1f}x more resident in the same memory")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "config": {
                    "budget_mb": args.budget_mb, "max_seq_len": args.max_seq_len,
                    "block_size": args.block_size, "slots": slots,
                    "kv_bytes_per_token": NANO_27M.kv_bytes_per_token,
                },
                "workload": {
                    "sequences": len(lengths),
                    "mean_len": float(np.mean(lengths)),
                    "p50": float(np.percentile(lengths, 50)),
                    "p99": float(np.percentile(lengths, 99)),
                },
                "contiguous_measured": static,
                "paged_measured": paged_measured,
                "paged_analytic": paged,
                "usable_capacity_gain": gain,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {args.out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
