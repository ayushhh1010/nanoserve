"""Find out where batch-1 decode time actually goes.

Wall clock alone cannot distinguish "the GPU is slow" from "the GPU is idle
waiting for Python" from "the GPU is downclocked". This measures all three:

  * wall time          -- what a user experiences
  * GPU busy time      -- CUDA events, i.e. time kernels were actually running
  * SM / memory clocks -- sampled during the loop, from a background thread

If wall >> GPU busy, the bottleneck is host-side: launch overhead or a sync.
If GPU busy is itself large, the bottleneck is on-device, and the clocks say
whether that is real work or a downclocked card.
"""

from __future__ import annotations

import argparse
import subprocess
import threading
import time

import torch

from model import NANO_27M, DynamicCache, NanoForCausalLM


class ClockSampler(threading.Thread):
    """Poll nvidia-smi in the background so we see clocks under load, not idle."""

    def __init__(self, interval: float = 0.05) -> None:
        super().__init__(daemon=True)
        self.interval = interval
        self.samples: list[tuple[int, int, float]] = []
        self._done = threading.Event()

    def run(self) -> None:
        query = "clocks.sm,clocks.mem,power.draw"
        while not self._done.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    timeout=2,
                ).stdout.strip()
                sm, mem, pwr = (p.strip() for p in out.split(","))
                self.samples.append((int(sm), int(mem), float(pwr)))
            except Exception:
                pass
            self._done.wait(self.interval)

    def stop(self) -> dict[str, float]:
        self._done.set()
        self.join(timeout=3)
        if not self.samples:
            return {}
        sm = [s[0] for s in self.samples]
        mem = [s[1] for s in self.samples]
        pwr = [s[2] for s in self.samples]
        return {
            "sm_peak": max(sm),
            "sm_mean": sum(sm) / len(sm),
            "mem_peak": max(mem),
            "mem_mean": sum(mem) / len(mem),
            "power_peak": max(pwr),
            "n": len(self.samples),
        }


def decode_loop(model, cache, tok, steps: int) -> torch.Tensor:
    for _ in range(steps):
        logits, _ = model(tok, cache=cache)
        tok = logits[:, -1].argmax(-1, keepdim=True)
    return tok


def measure(model, prefill_len: int, steps: int, label: str, sample_clocks: bool = False) -> dict:
    """One decode run, reported three ways."""
    device = next(model.parameters()).device
    ids = torch.randint(0, model.cfg.vocab_size, (1, prefill_len), device=device)

    with torch.inference_mode():
        cache = DynamicCache()
        model(ids, cache=cache)  # prefill, not measured
        tok = ids[:, :1]

        decode_loop(model, cache, tok, 8)  # warm the loop itself
        torch.cuda.synchronize()

        sampler = ClockSampler() if sample_clocks else None
        if sampler:
            sampler.start()

        start_ev, end_ev = torch.cuda.Event(True), torch.cuda.Event(True)
        start_ev.record()
        t0 = time.perf_counter()
        decode_loop(model, cache, tok, steps)
        t_launch = time.perf_counter() - t0  # returns before the GPU is done
        end_ev.record()
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0

        clocks = sampler.stop() if sampler else {}

    gpu_ms = start_ev.elapsed_time(end_ev)
    return {
        "label": label,
        "ms_per_token_wall": wall * 1000 / steps,
        "ms_per_token_gpu": gpu_ms / steps,
        "ms_per_token_launch": t_launch * 1000 / steps,
        "tok_per_s": steps / wall,
        **clocks,
    }


def report(r: dict) -> None:
    print(
        f"{r['label']:<34} {r['ms_per_token_wall']:7.2f} ms/tok  "
        f"({r['tok_per_s']:7.0f} tok/s)   "
        f"gpu={r['ms_per_token_gpu']:6.2f}  host-launch={r['ms_per_token_launch']:6.2f}"
    )
    if "sm_peak" in r:
        print(
            f"{'':<34} clocks: SM {r['sm_mean']:.0f}/{r['sm_peak']:.0f} MHz  "
            f"MEM {r['mem_mean']:.0f}/{r['mem_peak']:.0f} MHz  "
            f"peak {r['power_peak']:.0f} W  ({r['n']:.0f} samples)"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--prefill", type=int, default=512)
    args = ap.parse_args()

    torch.manual_seed(0)
    model = NanoForCausalLM(NANO_27M).cuda().to(torch.bfloat16).eval()
    weight_mb = model.num_parameters() * 2 / 1024**2

    print(f"model: {model.num_parameters():,} params, {weight_mb:.1f} MB @ bf16")
    print(f"decode: batch 1, {args.prefill} tokens prefilled, {args.steps} steps\n")

    # Cold: whatever power state the card happens to be in.
    report(measure(model, args.prefill, args.steps, "cold (as benchmarked before)", True))
    print()

    # Hot: hammer the GPU first so it clocks up, then measure.
    big = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    for _ in range(60):
        big = big @ big.T / 64
    torch.cuda.synchronize()
    del big
    torch.cuda.empty_cache()

    report(measure(model, args.prefill, args.steps, "hot (GPU clocked up first)", True))
    print()

    for n in (128, 512, 1024):
        report(measure(model, n, args.steps, f"hot, {n} tokens cached"))

    print()
    for bs in (1, 8, 32):
        device = "cuda"
        ids = torch.randint(0, NANO_27M.vocab_size, (bs, 256), device=device)
        with torch.inference_mode():
            cache = DynamicCache()
            model(ids, cache=cache)
            tok = ids[:, :1]
            decode_loop(model, cache, tok, 8)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            decode_loop(model, cache, tok, 64)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
        print(
            f"batch {bs:<3} decode: {dt * 1000 / 64:6.2f} ms/step  "
            f"-> {bs * 64 / dt:8.0f} tok/s aggregate"
        )


if __name__ == "__main__":
    main()
