"""Tokenise TinyStories into flat uint16 token files for training.

Drops the fragment document on each side of the official train/valid boundary.
That split is a byte offset rather than a document boundary: the train file
ends mid-word inside "You" and the valid file opens with "u don't have to be
scared...". Same story, both sides. One story out of ~2.7M is a negligible
leak, but it costs nothing to remove and "we checked" is a better answer than
"probably fine".
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from model.tokenizer import BPETokenizer
from model.train.data import prepare

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

SPLITS = {
    "train": {
        "corpus": "TinyStoriesV2-GPT4-train.txt",
        "out": "train.bin",
        # The trailing story continues into the valid file.
        "drop_last_document": True,
        "drop_first_document": False,
    },
    "valid": {
        "corpus": "TinyStoriesV2-GPT4-valid.txt",
        "out": "val.bin",
        # The opening fragment is the tail of the last train story.
        "drop_first_document": True,
        "drop_last_document": False,
    },
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokenizer", type=Path, default=ROOT / "model" / "tokenizer.json")
    ap.add_argument("--data-dir", type=Path, default=DATA)
    ap.add_argument("--splits", nargs="+", choices=list(SPLITS), default=list(SPLITS))
    args = ap.parse_args()

    if not args.tokenizer.exists():
        print(f"tokenizer not found: {args.tokenizer}", file=sys.stderr)
        print("run scripts/train_tokenizer.py first", file=sys.stderr)
        return 1

    tok = BPETokenizer.load(args.tokenizer)
    print(f"tokenizer: {tok.vocab_size:,} ids, {len(tok.merges):,} merges\n", flush=True)

    for name in args.splits:
        spec = SPLITS[name]
        corpus = args.data_dir / spec["corpus"]
        if not corpus.exists():
            print(f"missing {corpus}, skipping {name}", file=sys.stderr)
            continue

        print(f"{name}: {corpus.name} ({corpus.stat().st_size / 1024**3:.2f} GB)", flush=True)
        prepare(
            corpus=corpus,
            tokenizer=tok,
            out=args.data_dir / spec["out"],
            drop_first_document=spec["drop_first_document"],
            drop_last_document=spec["drop_last_document"],
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
