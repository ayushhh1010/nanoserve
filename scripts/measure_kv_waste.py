"""Measure what contiguous pre-allocation wastes on this project's workload.

The published figure for this design is 60-80% of KV memory wasted. That is a
citation until it is reproduced on the actual length distribution being served,
which is what this does: replay the benchmark workload through a static pool and
sample the accounting as sequences come and go.

It also states the paged bound, analytically, as the target step 3 has to hit.
That number is derived rather than measured -- with 16-token blocks, a sequence
can waste at most 15 tokens in its final partial block, and uniform blocks make
external fragmentation impossible by construction. Step 3 replaces the derived
number with a measured one.
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

    samples = replay(lengths, slots, args.max_seq_len)
    static = summarise(samples)
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
    print(f"PAGED, block size {args.block_size}, by construction:")
    print(f"  mean waste/sequence   {paged['mean_waste_tokens_per_seq']:8.1f} tokens")
    print(f"  max waste/sequence    {paged['max_waste_tokens_per_seq']:8d} tokens "
          f"(bounded at block_size - 1)")
    print(f"  external frag         {paged['external_fragmentation']:8d}   <- uniform blocks")
    print(f"  WASTE                 {paged['waste_fraction']:8.1%}")
    print()
    gain = (1 - paged["waste_fraction"]) / (1 - static["mean_waste_fraction"])
    print(f"  usable capacity gain  {gain:8.1f}x")

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
