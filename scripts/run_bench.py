"""Run a benchmark against an engine and write a results file.

    python scripts/run_bench.py --engine baseline-nocache --requests 40
    python scripts/run_bench.py --engine baseline --rate 4 --requests 200

Named run_bench rather than bench: a script named bench.py sits in scripts/,
which Python puts first on sys.path, so `import bench.metrics` would resolve to
the script itself rather than the package.

Results land in bench/results/ as JSON with the full environment attached, so a
number can always be traced back to the machine state that produced it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from bench.metrics import (
    RunResult,
    aggregate,
    format_repeats,
    format_summary,
    save_result,
)
from bench.runner import RunnerConfig, run
from bench.workload import WorkloadConfig, build_workload, load_prompt_pool, summarise
from engine.baseline import BaselineEngine
from model.generate import load_model
from model.tokenizer import BPETokenizer

ROOT = Path(__file__).resolve().parent.parent

ENGINES = {
    "baseline": dict(use_cache=True),          # cache grows by torch.cat
    "baseline-nocache": dict(use_cache=False),  # no cache: re-attend every token
    "baseline-static": dict(use_cache=True),    # preallocated contiguous slot
    "baseline-paged": dict(use_cache=True),     # block table + free list
    "continuous": {},                           # iteration-level scheduling
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine", choices=list(ENGINES), default="baseline")
    ap.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/run1/best.pt")
    ap.add_argument("--tokenizer", type=Path, default=ROOT / "model/tokenizer.json")
    ap.add_argument("--corpus", type=Path, default=ROOT / "data/TinyStoriesV2-GPT4-valid.txt")

    ap.add_argument("--requests", type=int, default=200)
    ap.add_argument("--rate", type=float, default=None,
                    help="requests/sec (Poisson). Omit for all-at-once throughput mode.")
    ap.add_argument("--output-len", type=int, default=96, help="median output length")
    ap.add_argument("--prompt-len", type=int, default=96, help="median unique prompt length")
    ap.add_argument("--prefixes", type=int, default=16)
    ap.add_argument("--prefix-len", type=int, default=128,
                    help="tokens in each shared prefix (the system-prompt length)")
    ap.add_argument("--prefix-fraction", type=float, default=0.8,
                    help="fraction of requests carrying a shared prefix")
    ap.add_argument("--zipf", type=float, default=1.0)
    ap.add_argument("--slo", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--warmup", type=int, default=12)
    ap.add_argument("--synthetic", action="store_true",
                    help="use random token ids instead of real text")
    ap.add_argument("--kv-budget-mb", type=int, default=512,
                    help="KV pool budget for --engine baseline-static")
    ap.add_argument("--max-seq-len", type=int, default=1024,
                    help="per-slot reservation for the contiguous pool")
    ap.add_argument("--max-batch-size", type=int, default=64,
                    help="cap on concurrent sequences for --engine continuous")
    ap.add_argument("--admission", action="store_true",
                    help="enable admission control (bounded queue + deadline projection)")
    ap.add_argument("--max-queue-depth", type=int, default=64)
    ap.add_argument("--admission-slack", type=float, default=1.25)
    ap.add_argument("--no-prefix-cache", action="store_true",
                    help="disable prefix-block reuse for --engine continuous")
    ap.add_argument("--reserve-blocks", type=int, default=8,
                    help="floor on KV headroom kept for growing sequences")
    ap.add_argument("--block-size", type=int, default=16,
                    help="tokens per block for --engine baseline-paged")
    ap.add_argument("--repeats", type=int, default=1,
                    help="run the identical workload N times and report the spread")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--tag", default="", help="suffix for the results filename")
    args = ap.parse_args()

    for p in (args.checkpoint, args.tokenizer):
        if not p.exists():
            print(f"missing {p}", file=sys.stderr)
            return 1

    tok = BPETokenizer.load(args.tokenizer)
    model, meta = load_model(args.checkpoint, device=args.device, dtype=getattr(torch, args.dtype))
    print(f"model: {model.num_parameters():,} params, step {meta.get('step')}, "
          f"val {meta.get('best_val')}")

    prompt_pool = None
    if not args.synthetic and args.corpus.exists():
        prompt_pool = load_prompt_pool(args.corpus, tok)
        print(f"prompt pool: {len(prompt_pool)} real documents")

    wcfg = WorkloadConfig(
        num_requests=args.requests,
        request_rate=args.rate,
        num_prefixes=args.prefixes,
        prefix_len=args.prefix_len,
        prefix_fraction=args.prefix_fraction,
        zipf_alpha=args.zipf,
        prompt_len_mean=args.prompt_len,
        output_len_mean=args.output_len,
        slo_seconds=args.slo,
        seed=args.seed,
    )
    requests = build_workload(wcfg, tok.vocab_size, tok.eos_id, prompt_pool)
    wsummary = summarise(requests)
    print(
        f"workload: {wsummary['num_requests']:,} requests, "
        f"{wsummary['output_tokens']:,} output tokens, "
        f"rate={'burst' if args.rate is None else f'{args.rate}/s'}"
    )

    def make_kv_pool():
        budget = args.kv_budget_mb * 1024 * 1024
        dtype = getattr(torch, args.dtype)
        if args.engine in ("baseline-paged", "continuous"):
            from engine.block_manager import PagedCachePool, blocks_for_budget

            kv = PagedCachePool(
                model.cfg, num_blocks=blocks_for_budget(model.cfg, budget, args.block_size),
                block_size=args.block_size, device=args.device, dtype=dtype,
            )
        else:
            from engine.kv_cache import StaticCachePool, slots_for_budget

            kv = StaticCachePool(
                model.cfg, num_slots=slots_for_budget(model.cfg, budget, args.max_seq_len),
                max_seq_len=args.max_seq_len, device=args.device, dtype=dtype,
            )
        print(f"kv pool: {kv}")
        return kv

    summaries = []
    for rep in range(args.repeats):
        if args.repeats > 1:
            print(f"\n--- repeat {rep + 1}/{args.repeats} ---", flush=True)
        if args.engine == "continuous":
            from engine.admission import AdmissionConfig
            from engine.scheduler import ContinuousBatchingEngine

            engine = ContinuousBatchingEngine(
                model, make_kv_pool(), eos_token_id=tok.eos_id, device=args.device,
                max_batch_size=args.max_batch_size, reserve_blocks=args.reserve_blocks,
                enable_prefix_cache=not args.no_prefix_cache,
                admission=AdmissionConfig(
                    max_queue_depth=args.max_queue_depth, slack=args.admission_slack
                ) if args.admission else None,
            )
        else:
            engine = BaselineEngine(
                model, eos_token_id=tok.eos_id, device=args.device,
                pool=make_kv_pool()
                if args.engine in ("baseline-static", "baseline-paged")
                else None,
                **ENGINES[args.engine],
            )
        engine.name = args.engine
        # Rebuild the workload each repeat: Request objects carry mutable
        # timing state, so reusing them would measure the second run against
        # the first run's timestamps.
        reqs = build_workload(wcfg, tok.vocab_size, tok.eos_id, prompt_pool)
        result = run(
            engine, reqs, wsummary,
            RunnerConfig(warmup_requests=args.warmup, verbose=args.repeats == 1),
            vocab_size=tok.vocab_size,
        )
        summaries.append(result.summary())

    agg = aggregate(summaries)
    # Report and save the SAME run. Previously the printed block was the
    # middle repeat while the saved JSON held the last one, so a results file
    # never matched the numbers anyone had actually looked at.
    summary = agg["representative"]
    print()
    print(format_summary(summary))
    if args.repeats > 1:
        print(format_repeats(agg))

    name = args.tag or f"{args.engine}_n{args.requests}" + (
        f"_r{args.rate:g}" if args.rate else "_burst"
    )
    out = args.out or ROOT / "bench" / "results" / f"{name}.json"
    save_result(
        RunResult(engine=summary["engine"], wall_seconds=summary["wall_seconds"],
                  requests=[], workload=wsummary, extra={}),
        out,
        extra={"workload_config": wcfg.to_dict(), "repeats": agg, "result": summary},
    )
    print(f"\nwrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
