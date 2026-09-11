"""Measure what INT8 weight-only quantization actually costs.

Reports the perplexity delta on held-out data alongside the memory saved.
Reporting the cost of your own optimisation is the point: a quantizer that
only ever shows the gain is not a measurement, it is an advertisement.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from engine.quant import DEFAULT_GROUP_SIZE, quantization_error, quantize_model
from model.generate import load_model
from model.train.data import TokenDataset

ROOT = Path(__file__).resolve().parent.parent


@torch.inference_mode()
def evaluate(model, data: TokenDataset, batches: int, batch_size: int, device: str) -> float:
    """Mean cross-entropy over a fixed set of windows.

    Seeded so both models see byte-identical data -- a different sample would
    swamp the effect being measured.
    """
    gen = torch.Generator().manual_seed(1234)
    total = 0.0
    for _ in range(batches):
        tokens = data.batch(batch_size, device, gen)
        _, loss = model(tokens, labels=tokens)
        total += loss.item()
    return total / batches


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/run1/best.pt")
    ap.add_argument("--val-bin", type=Path, default=ROOT / "data/val.bin")
    ap.add_argument("--group-size", type=int, default=DEFAULT_GROUP_SIZE)
    ap.add_argument("--batches", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out", type=Path, default=ROOT / "bench" / "results" / "quant.json")
    args = ap.parse_args()

    if not args.val_bin.exists():
        print(f"missing {args.val_bin}; run scripts/prepare_data.py")
        return 1

    dtype = getattr(torch, args.dtype)
    data = TokenDataset(args.val_bin, seq_len=args.seq_len)

    model, meta = load_model(args.checkpoint, device=args.device, dtype=dtype)
    print(f"model: {model.num_parameters():,} params, val {meta.get('best_val')}")
    print(f"eval : {args.batches} batches x {args.batch_size} x {args.seq_len} tokens "
          f"= {args.batches * args.batch_size * args.seq_len:,} tokens\n")

    dense_loss = evaluate(model, data, args.batches, args.batch_size, args.device)
    dense_mb = sum(
        p.numel() * p.element_size() for p in model.parameters()
    ) / 1024**2

    # Per-layer reconstruction error, before the model is mutated.
    errors = {
        name: quantization_error(m.weight, args.group_size)["relative_error"]
        for name, m in model.named_modules()
        if isinstance(m, torch.nn.Linear)
        and "lm_head" not in name
        and m.in_features % args.group_size == 0
    }

    stats = quantize_model(model, group_size=args.group_size)
    quant_loss = evaluate(model, data, args.batches, args.batch_size, args.device)

    d = stats.to_dict()
    delta = (quant_loss - dense_loss) / dense_loss
    ppl_dense, ppl_quant = float(np.exp(dense_loss)), float(np.exp(quant_loss))

    print(f"{'':22}{'loss':>10}{'perplexity':>13}")
    print("-" * 46)
    print(f"{'bf16 (dense)':22}{dense_loss:>10.4f}{ppl_dense:>13.4f}")
    print(f"{'int8 weight-only':22}{quant_loss:>10.4f}{ppl_quant:>13.4f}")
    print(f"{'delta':22}{quant_loss - dense_loss:>+10.4f}"
          f"{ppl_quant - ppl_dense:>+13.4f}")
    print(f"{'relative':22}{delta:>+10.2%}"
          f"{(ppl_quant - ppl_dense) / ppl_dense:>+13.2%}")
    print()
    print(f"layers quantized      {d['layers_quantized']}")
    print(f"layers skipped        {d['layers_skipped']}  (lm_head is tied to the "
          f"embedding table)")
    print(f"linear weights        {d['original_mb']:.2f} MB -> {d['quantized_mb']:.2f} MB "
          f"({d['compression']:.2f}x)")
    print(f"whole model (bf16)    {dense_mb:.2f} MB")
    print()
    worst = max(errors.items(), key=lambda kv: kv[1])
    print(f"worst layer reconstruction: {worst[0]} at {worst[1]:.3%}")
    print(f"mean layer reconstruction : {np.mean(list(errors.values())):.3%}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "group_size": args.group_size,
                "eval_tokens": args.batches * args.batch_size * args.seq_len,
                "dense": {"loss": dense_loss, "perplexity": ppl_dense},
                "int8": {"loss": quant_loss, "perplexity": ppl_quant},
                "delta": {
                    "loss": quant_loss - dense_loss,
                    "loss_relative": delta,
                    "perplexity_relative": (ppl_quant - ppl_dense) / ppl_dense,
                },
                "memory": d,
                "layer_reconstruction_error": errors,
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
