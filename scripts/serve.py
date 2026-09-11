"""Run the Nanoserve HTTP server.

    python scripts/serve.py --port 8000

    curl -N localhost:8000/generate -H 'content-type: application/json' \
      -d '{"prompt":"Once upon a time","max_tokens":100}'
"""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from engine.server import build_app

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--checkpoint", default=str(ROOT / "checkpoints/run1/best.pt"))
    ap.add_argument("--tokenizer", default=str(ROOT / "model/tokenizer.json"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--kv-budget-mb", type=int, default=1024)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--max-batch-size", type=int, default=64)
    ap.add_argument("--max-queue-depth", type=int, default=64)
    ap.add_argument("--no-prefix-cache", action="store_true")
    args = ap.parse_args()

    app = build_app(
        checkpoint=args.checkpoint, tokenizer_path=args.tokenizer,
        device=args.device, dtype=args.dtype, kv_budget_mb=args.kv_budget_mb,
        block_size=args.block_size, max_batch_size=args.max_batch_size,
        max_queue_depth=args.max_queue_depth,
        enable_prefix_cache=not args.no_prefix_cache,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
