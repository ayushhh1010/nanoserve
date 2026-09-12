"""Write a randomly-initialised checkpoint, so CI can run without the weights.

    python scripts/make_test_checkpoint.py --out checkpoints/run1/best.pt

The trained checkpoint is 321 MB and is not in the repository -- it does not
belong in git, and a CI job that had to download it would spend most of its
runtime doing so.

This is sound for the chaos suite and only for it. Those scenarios assert on
request *lifecycle* -- that a SIGKILLed replica's in-flight generations migrate,
that a drain drops nothing, that the limiter admits exactly the burst -- and
none of that reads the weights. A random model emits nonsense at the same shape,
rate and KV cost as a trained one, which is the entire surface chaos touches.

What this must never be used for is quality: loss, perplexity, or generation
samples from these weights are noise. Phase 1 learned that the hard way in the
other direction -- a quantization test asserted logits barely moved on a
randomly-initialised model, saw 96% movement, and looked like a serious bug. It
was measuring two random vectors, which differ by ~141% on average. Random
weights are fine for testing plumbing and worthless for testing numbers.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.config import NanoConfig  # noqa: E402
from model.transformer import NanoForCausalLM  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "checkpoints/run1/best.pt")
    ap.add_argument("--seed", type=int, default=0,
                    help="fixed so a CI rerun produces byte-identical weights")
    args = ap.parse_args()

    if args.out.exists():
        print(f"{args.out} already exists; refusing to overwrite.\n"
              "If this is a real trained checkpoint, that is the point.",
              file=sys.stderr)
        return 1

    torch.manual_seed(args.seed)
    cfg = NanoConfig()
    model = NanoForCausalLM(cfg)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Same shape the trainer writes, so load_model takes this path unchanged
    # and CI exercises the real loading code rather than a test-only branch.
    torch.save(
        {
            "model": model.state_dict(),
            "config": {"model": asdict(cfg)},
            "step": 0,
            "tokens_seen": 0,
            "best_val": float("inf"),
            "RANDOM_INIT": True,  # so nobody mistakes this for trained weights
        },
        args.out,
    )

    size_mb = args.out.stat().st_size / 1e6
    params = sum(p.numel() for p in model.parameters())
    print(f"wrote {args.out} ({size_mb:.1f} MB, {params:,} random parameters)")
    print("NOT TRAINED -- valid for chaos and plumbing tests, meaningless for quality.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
