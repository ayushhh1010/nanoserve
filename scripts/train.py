"""Train the model. Resumes automatically if a checkpoint is present.

    python scripts/train.py                      # start or resume the full run
    python scripts/train.py --max-steps 200      # short smoke run
    python scripts/train.py --fresh               # ignore existing checkpoints

The default behaviour is to resume, not to restart. Getting that backwards is
how a thermal event at hour four turns into starting again from zero.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from model.train.train import TrainConfig, Trainer

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-bin", type=Path, default=ROOT / "data" / "train.bin")
    ap.add_argument("--val-bin", type=Path, default=ROOT / "data" / "val.bin")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "checkpoints")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--micro-batch", type=int, default=None)
    ap.add_argument("--grad-accum", type=int, default=None)
    ap.add_argument("--eval-every", type=int, default=None)
    ap.add_argument("--ckpt-every", type=int, default=None)
    ap.add_argument("--log-every", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--no-compile", action="store_true", help="disable torch.compile")
    ap.add_argument("--fresh", action="store_true", help="ignore existing checkpoints")
    args = ap.parse_args()

    for path in (args.train_bin, args.val_bin):
        if not path.exists():
            print(f"missing {path}", file=sys.stderr)
            print("run scripts/download_data.py, then scripts/prepare_data.py", file=sys.stderr)
            return 1

    overrides = {
        k: v
        for k, v in {
            "max_steps": args.max_steps,
            "micro_batch": args.micro_batch,
            "grad_accum": args.grad_accum,
            "eval_every": args.eval_every,
            "ckpt_every": args.ckpt_every,
            "log_every": args.log_every,
        }.items()
        if v is not None
    }

    cfg = TrainConfig(
        train_bin=args.train_bin,
        val_bin=args.val_bin,
        out_dir=args.out_dir,
        device=args.device,
        dtype=args.dtype,
        seed=args.seed,
        compile=not args.no_compile,
        **overrides,
    )

    latest = None if args.fresh else Trainer.latest_checkpoint(cfg.out_dir)
    if latest is not None:
        print(f"resuming from {latest.name}", flush=True)
        trainer = Trainer.resume(latest, cfg)
    else:
        trainer = Trainer(cfg)

    trainer.fit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
