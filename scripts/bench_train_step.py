"""How big a micro-batch fits, and how fast a training step runs.

Inference memory says nothing useful about training memory. A forward pass
under `inference_mode` frees every activation as it goes; a training step keeps
all of them alive for the backward pass, and AdamW adds two fp32 moment buffers
the size of the model. On a 4 GB card that difference decides the batch size,
which decides the wall-clock of the whole run.

Measures, for each micro-batch size: peak VRAM, step time, and the resulting
tokens/second and projected time to 536M tokens.
"""

from __future__ import annotations

import argparse
import time

import torch

from model import NANO_27M, NanoForCausalLM


def try_micro_batch(mb: int, seq_len: int, steps: int, compile_model: bool = False) -> dict | None:
    """One micro-batch size, or None if it does not fit."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    try:
        torch.manual_seed(0)
        model = NanoForCausalLM(NANO_27M).cuda()
        opt = torch.optim.AdamW(
            model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1
        )
        ids = torch.randint(0, NANO_27M.vocab_size, (mb, seq_len), device="cuda")

        def step():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = model(ids, labels=ids)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            return loss

        for _ in range(3):  # warm up: allocator, autotune, clocks
            step()
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(steps):
            step()
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / steps

        peak = torch.cuda.max_memory_allocated() / 1024**2
        reserved = torch.cuda.max_memory_reserved() / 1024**2
        result = {
            "micro_batch": mb,
            "tokens_per_step": mb * seq_len,
            "ms": dt * 1000,
            "tok_s": mb * seq_len / dt,
            "peak_mb": peak,
            "reserved_mb": reserved,
        }
    except torch.OutOfMemoryError:
        result = None
    finally:
        for name in ("model", "opt", "ids"):
            if name in dir():
                pass
        torch.cuda.empty_cache()

    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--target-tokens", type=float, default=536e6)
    ap.add_argument("--tokens-per-step", type=int, default=32768)
    args = ap.parse_args()

    props = torch.cuda.get_device_properties(0)
    print(f"{props.name}  {props.total_memory / 1024**3:.1f} GB  seq_len={args.seq_len}\n")

    param_mb = 26_747_392 * 4 / 1024**2
    print(f"fp32 weights          : {param_mb:6.0f} MB")
    print(f"gradients             : {param_mb:6.0f} MB")
    print(f"AdamW moments (2x)    : {2 * param_mb:6.0f} MB")
    print(f"optimizer state total : {4 * param_mb:6.0f} MB   <- before a single activation\n")

    print(f"{'micro':>5} {'tokens':>8} {'step ms':>9} {'tok/s':>9} "
          f"{'peak MB':>9} {'resv MB':>9} {'accum':>6} {'run time':>10}")
    print("-" * 76)

    vram_mb = props.total_memory / 1024**2
    results = []
    for mb in (1, 2, 4, 8, 12, 16, 24, 32):
        r = try_micro_batch(mb, args.seq_len, args.steps)
        if r is None:
            print(f"{mb:>5} {'':>8} {'OOM':>9}")
            break
        accum = max(1, round(args.tokens_per_step / r["tokens_per_step"]))
        hours = args.target_tokens / r["tok_s"] / 3600
        spill = " SPILL" if r["reserved_mb"] > vram_mb else ""
        print(
            f"{mb:>5} {r['tokens_per_step']:>8,} {r['ms']:>9.1f} {r['tok_s']:>9,.0f} "
            f"{r['peak_mb']:>9,.0f} {r['reserved_mb']:>9,.0f} {accum:>6} {hours:>9.1f}h{spill}"
        )
        results.append(r)

    if not results:
        return

    # The fastest, NOT the largest that runs without raising. On Windows the
    # WDDM driver silently backs an over-committed allocation with system RAM
    # over PCIe instead of refusing it, so a micro-batch far past VRAM still
    # "fits" -- at up to 14x the step time. Reporting the largest working size
    # would recommend the worst option on the table.
    best = max(results, key=lambda r: r["tok_s"])
    hours = args.target_tokens / best["tok_s"] / 3600
    accum = max(1, round(args.tokens_per_step / best["tokens_per_step"]))

    print(f"\nfastest micro-batch : {best['micro_batch']}  ({best['tok_s']:,.0f} tok/s)")
    print(f"grad-accum for {args.tokens_per_step:,} tokens/step: {accum}")
    print(f"peak VRAM           : {best['peak_mb']:,.0f} MB of {vram_mb:,.0f} MB")
    print(f"536M tokens         : {hours:.1f} h")

    spilled = [r for r in results if r["reserved_mb"] > vram_mb]
    if spilled:
        first = spilled[0]
        print(
            f"\nnote: micro-batch >= {first['micro_batch']} over-commits VRAM "
            f"({first['reserved_mb']:,.0f} MB reserved of {vram_mb:,.0f} MB). It does not "
            f"OOM -- it pages to system RAM and runs {best['tok_s'] / first['tok_s']:.1f}x slower."
        )


if __name__ == "__main__":
    main()
