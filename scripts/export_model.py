"""Export a training checkpoint as a distributable model directory.

A training checkpoint is 307 MB, most of which is AdamW's two moment buffers
and RNG state -- useless to anyone loading the model for inference. This writes
just the weights, the config, the tokenizer and a model card.

Nothing here uploads anything. It writes a directory; publishing it is a
separate, deliberate act.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import date
from pathlib import Path

import torch
from safetensors.torch import save_file

from model.config import NanoConfig, count_parameters

ROOT = Path(__file__).resolve().parent.parent

CARD = """---
license: mit
datasets:
  - roneneldan/TinyStories
language:
  - en
pipeline_tag: text-generation
tags:
  - tinystories
  - from-scratch
  - small-language-model
---

# Nanoserve {params_m:.0f}M

A {params:,}-parameter decoder-only transformer trained from scratch on
TinyStories V2. The architecture, the tokenizer and the training loop were all
written by hand -- no `AutoModel`, no `tokenizers` library.

It writes short children's stories. It is not a general-purpose assistant and
cannot answer questions, follow instructions, or reason. That is the
specification, not a shortfall.

## Results

| | |
|---|---|
| Parameters | {params:,} |
| Training tokens | {tokens:,} |
| Validation loss | **{val_loss:.4f} nats/token** |
| Perplexity | **{ppl:.3f}** |
| Hardware | single RTX 3050 Laptop (4 GB) |
| Wall clock | {hours:.2f} hours |

## Architecture

| | |
|---|---|
| `d_model` / layers / heads | {hidden} / {layers} / {heads} |
| KV heads | {kv_heads} (grouped-query attention) |
| `d_ff` | {ffn} (SwiGLU) |
| Norm / position | RMSNorm pre-norm / RoPE |
| Vocabulary / context | {vocab:,} / {ctx:,} |
| Embeddings | {tied} |

Parameter names mirror HuggingFace's Llama, so the weights load into a
`LlamaForCausalLM` of matching shape.

## Usage

```python
from model.tokenizer import BPETokenizer
from model.generate import load_model, generate, SamplingParams

tok = BPETokenizer.load("tokenizer.json")
model, _ = load_model("model.safetensors", device="cuda")

print(generate(
    model, tok,
    "Once upon a time, there was a little girl named Lily who",
    SamplingParams(max_new_tokens=200, temperature=0.8),
))
```

## Training data

TinyStories V2 (GPT-4 generated). {tokens:,} tokens under a purpose-built
8,192-token byte-level BPE vocabulary, which compresses this corpus to
{bpt:.3f} bytes/token with {single_pct:.1f}% of words encoding to a single
token.

Token count was chosen to sit at the Chinchilla-optimal ratio for the parameter
count (~20 tokens per parameter).

## Limitations

- Vocabulary and world model are limited to the simple English of TinyStories.
- 1,024-token context.
- No instruction tuning, no alignment, no safety filtering of any kind.
- Will confidently produce factually wrong statements; it models story-shaped
  text, not truth.

## Source

Trained as Phase 1 of [Nanoserve](https://github.com/{github}/nanoserve), a
project that trains a small language model and then builds the distributed
system that serves it.
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/run1/best.pt")
    ap.add_argument("--tokenizer", type=Path, default=ROOT / "model/tokenizer.json")
    ap.add_argument("--out", type=Path, default=ROOT / "export/nanoserve-27m")
    ap.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    ap.add_argument("--val-loss", type=float, default=1.1485)
    ap.add_argument("--tokens", type=int, default=536_174_127)
    ap.add_argument("--hours", type=float, default=3.27)
    ap.add_argument("--github", default="YOUR-GITHUB-USERNAME")
    args = ap.parse_args()

    for path in (args.checkpoint, args.tokenizer):
        if not path.exists():
            print(f"missing {path}", file=sys.stderr)
            return 1

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = NanoConfig(**ckpt["config"]["model"])
    dtype = getattr(torch, args.dtype)

    args.out.mkdir(parents=True, exist_ok=True)

    # Tied weights are one tensor under two names; safetensors refuses aliases,
    # so the head is dropped and reconstructed at load time from the config.
    weights = {k: v.to(dtype).contiguous() for k, v in ckpt["model"].items()}
    if cfg.tie_word_embeddings:
        weights.pop("lm_head.weight", None)

    save_file(weights, args.out / "model.safetensors", metadata={"format": "pt"})
    (args.out / "config.json").write_text(
        json.dumps({**cfg.to_dict(), "model_type": "nanoserve", "torch_dtype": args.dtype}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    shutil.copy(args.tokenizer, args.out / "tokenizer.json")

    counts = count_parameters(cfg)
    import math

    (args.out / "README.md").write_text(
        CARD.format(
            params=counts["total"],
            params_m=counts["total"] / 1e6,
            tokens=args.tokens,
            val_loss=args.val_loss,
            ppl=math.exp(args.val_loss),
            hours=args.hours,
            hidden=cfg.hidden_size,
            layers=cfg.num_hidden_layers,
            heads=cfg.num_attention_heads,
            kv_heads=cfg.num_key_value_heads,
            ffn=cfg.intermediate_size,
            vocab=cfg.vocab_size,
            ctx=cfg.max_position_embeddings,
            tied="tied" if cfg.tie_word_embeddings else "untied",
            bpt=3.965,
            single_pct=98.1,
            github=args.github,
        ),
        encoding="utf-8",
    )
    (args.out / "EXPORTED.json").write_text(
        json.dumps(
            {
                "exported": date.today().isoformat(),
                "checkpoint": str(args.checkpoint.name),
                "step": ckpt.get("step"),
                "val_loss": args.val_loss,
                "dtype": args.dtype,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    total = sum(p.stat().st_size for p in args.out.iterdir() if p.is_file())
    print(f"exported to {args.out}  ({total / 1024**2:.1f} MB)")
    for p in sorted(args.out.iterdir()):
        print(f"  {p.stat().st_size / 1024**2:8.2f} MB  {p.name}")
    print("\nNothing has been uploaded. To publish, from that directory:")
    print("  huggingface-cli login")
    print(f"  huggingface-cli upload <your-username>/nanoserve-27m {args.out} .")
    return 0


if __name__ == "__main__":
    sys.exit(main())
