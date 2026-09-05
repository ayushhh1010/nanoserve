"""Train the BPE tokenizer on TinyStories and save it.

Trained on the *training* split only. Letting validation text influence the
vocabulary is a small but real leak: tokens learned from held-out data make
held-out text compress better than it should, and validation loss is measured
per token.

The corpus is streamed rather than read into memory. `BPETokenizer.train`
collapses it to a unique-pretoken frequency table on the way through, so peak
memory is set by the number of distinct words, not by the 2 GB file.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from model.tokenizer import BPETokenizer
from model.train.data import stream_documents

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CORPUS = ROOT / "data" / "TinyStoriesV2-GPT4-train.txt"
DEFAULT_OUT = ROOT / "model" / "tokenizer.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--vocab-size", type=int, default=8192)
    ap.add_argument(
        "--max-bytes",
        type=int,
        default=None,
        help="train on only the first N bytes of the corpus (for quick iteration)",
    )
    args = ap.parse_args()

    if not args.corpus.exists():
        print(f"corpus not found: {args.corpus}", file=sys.stderr)
        print("run scripts/download_data.py first", file=sys.stderr)
        return 1

    size = args.corpus.stat().st_size
    budget = min(size, args.max_bytes) if args.max_bytes else size
    print(f"corpus : {args.corpus.name} ({size / 1024**3:.2f} GB)")
    print(f"using  : {budget / 1024**3:.2f} GB")
    print(f"vocab  : {args.vocab_size:,}\n")

    t0 = time.perf_counter()
    tok = BPETokenizer.train(
        stream_documents(args.corpus, args.max_bytes),
        vocab_size=args.vocab_size,
        verbose=True,
    )
    elapsed = time.perf_counter() - t0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tok.save(args.out)

    print(f"\ntrained in {elapsed / 60:.1f} min ({budget / 1024**2 / elapsed:.1f} MB/s)")
    print(f"vocab_size : {tok.vocab_size:,} ({len(tok.merges):,} merges)")
    print(f"saved      : {args.out}  ({args.out.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
