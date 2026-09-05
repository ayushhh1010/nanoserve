"""Answer two questions about a decode step, without inferring from wall clock.

1. Does a decode step synchronise with the host at all?
   `torch.cuda.set_sync_debug_mode("error")` turns every synchronising op into
   an exception, so this is a yes/no with a stack trace, not a guess.

2. If not, where does the time go?
   The profiler gives total CUDA kernel time. Compare it to wall time: the
   difference is the GPU sitting idle waiting for Python to issue the next op.
"""

from __future__ import annotations

import time

import torch
from torch.profiler import ProfilerActivity, profile

from model import NANO_27M, DynamicCache, NanoForCausalLM


def build():
    torch.manual_seed(0)
    model = NanoForCausalLM(NANO_27M).cuda().to(torch.bfloat16).eval()
    ids = torch.randint(0, NANO_27M.vocab_size, (1, 512), device="cuda")
    cache = DynamicCache()
    with torch.inference_mode():
        model(ids, cache=cache)
    return model, cache, ids[:, :1]


def check_for_sync(model, cache, tok) -> None:
    print("=" * 72)
    print("1. Does a decode step synchronise?")
    print("=" * 72)
    with torch.inference_mode():
        for _ in range(3):
            logits, _ = model(tok, cache=cache)
            tok = logits[:, -1].argmax(-1, keepdim=True)
        torch.cuda.synchronize()

        torch.cuda.set_sync_debug_mode("error")
        try:
            logits, _ = model(tok, cache=cache)
            _ = logits[:, -1].argmax(-1, keepdim=True)
            print("   NO SYNC. The decode step queues work and returns.\n")
        except Exception as exc:  # noqa: BLE001
            print(f"   SYNC FOUND: {type(exc).__name__}: {exc}\n")
        finally:
            torch.cuda.set_sync_debug_mode("default")


def kernel_time_vs_wall(model, cache, tok, steps: int = 64) -> None:
    print("=" * 72)
    print("2. Kernel time vs wall time")
    print("=" * 72)
    with torch.inference_mode():
        for _ in range(8):
            logits, _ = model(tok, cache=cache)
            tok = logits[:, -1].argmax(-1, keepdim=True)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(steps):
                logits, _ = model(tok, cache=cache)
                tok = logits[:, -1].argmax(-1, keepdim=True)
            torch.cuda.synchronize()
        wall = time.perf_counter() - t0

    evts = prof.key_averages()
    cuda_us = sum(e.self_device_time_total for e in evts)
    cpu_us = sum(e.self_cpu_time_total for e in evts)
    n_launches = sum(e.count for e in evts if e.self_device_time_total > 0)

    print(f"   wall            : {wall * 1e3 / steps:8.3f} ms/token")
    print(f"   GPU kernel time : {cuda_us / steps / 1e3:8.3f} ms/token   <- real device work")
    print(f"   CPU op time     : {cpu_us / steps / 1e3:8.3f} ms/token")
    print(f"   GPU utilisation : {cuda_us / 1e6 / wall * 100:8.1f} %")
    print(f"   kernels/token   : {n_launches / steps:8.1f}")
    print(f"   per-kernel wall : {wall * 1e6 / n_launches:8.1f} us  (launch-bound if >> 5us)\n")

    print("   Top 12 ops by GPU time:")
    print(evts.table(sort_by="self_cuda_time_total", row_limit=12, max_name_column_width=42))


def roofline() -> None:
    """What the card could do if decode were purely bandwidth-bound."""
    print("=" * 72)
    print("3. Roofline")
    print("=" * 72)
    # 128 MB of real data, generously warmed. An earlier version of this used a
    # 16 MB `torch.empty` buffer with 5 warmup iterations, immediately after the
    # profiler section above, and reported 7.9 GB/s -- 20x below spec. Small
    # buffers make the measurement launch-bound rather than bandwidth-bound,
    # and profiler teardown was still contending. The lesson generalises to
    # every number in bench/: warm up properly and size the work to the thing
    # you are trying to measure.
    n = 64 * 1024 * 1024
    x = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    y = torch.empty_like(x)
    for _ in range(20):
        y.copy_(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(50):
        y.copy_(x)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / 50
    bw = 2 * x.numel() * 2 / dt / 1e9  # read + write

    weights_mb = 26_747_392 * 2 / 1024**2
    print(f"   measured copy bandwidth : {bw:8.1f} GB/s")
    print(f"   model weights           : {weights_mb:8.1f} MB")
    print(f"   bandwidth-bound decode  : {bw * 1e9 / (weights_mb * 1024**2):8.0f} tok/s\n")


def main() -> None:
    model, cache, tok = build()
    check_for_sync(model, cache, tok)
    kernel_time_vs_wall(model, cache, tok)
    roofline()


if __name__ == "__main__":
    main()
