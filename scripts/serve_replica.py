"""Run one gRPC inference replica.

    python scripts/serve_replica.py --port 9101 --kv-budget-mb 512
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from engine.replica import serve

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=9101,
                    help="Windows reserves 50000-50559, so not 50051")
    ap.add_argument("--checkpoint", default=str(ROOT / "checkpoints/run1/best.pt"))
    ap.add_argument("--tokenizer", default=str(ROOT / "model/tokenizer.json"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--kv-budget-mb", type=int, default=512)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--max-batch-size", type=int, default=64)
    ap.add_argument("--max-queue-depth", type=int, default=64)
    ap.add_argument("--replica-id", default=None)
    ap.add_argument("--etcd", default=None,
                    help="comma-separated etcd endpoints; omit to skip registration")
    ap.add_argument("--advertise", default=None,
                    help="address the router should dial; defaults to 127.0.0.1:PORT")
    ap.add_argument("--lease-ttl", type=int, default=10,
                    help="seconds before a dead replica leaves the registry")
    args = ap.parse_args()

    asyncio.run(
        serve(
            port=args.port, checkpoint=args.checkpoint, tokenizer_path=args.tokenizer,
            device=args.device, dtype=args.dtype, kv_budget_mb=args.kv_budget_mb,
            block_size=args.block_size, max_batch_size=args.max_batch_size,
            max_queue_depth=args.max_queue_depth, replica_id=args.replica_id,
            etcd=args.etcd, advertise=args.advertise, lease_ttl=args.lease_ttl,
        )
    )


if __name__ == "__main__":
    main()
