"""Fetch the TinyStories corpus.

V2 rather than V1: the dataset card states V1 contains GPT-3.5 generations "of
lesser quality", while V2 is GPT-4 only and strictly larger. Phase 1's
acceptance bar is grammatical English stories, so corpus quality is the thing
most directly under test.

Downloads straight into `data/` with `local_dir=`, which writes the file once
rather than materialising it in the shared HuggingFace cache on C: and again
here. At 2 GB that distinction is worth the one keyword argument.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "roneneldan/TinyStories"

FILES = {
    "v2": {
        "train": "TinyStoriesV2-GPT4-train.txt",
        "valid": "TinyStoriesV2-GPT4-valid.txt",
    },
    "v1": {
        "train": "TinyStories-train.txt",
        "valid": "TinyStories-valid.txt",
    },
}

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def fetch(filename: str, dest: Path) -> Path:
    target = dest / filename
    if target.exists():
        print(f"  {filename}: already present ({target.stat().st_size / 1024**3:.2f} GB)")
        return target

    print(f"  {filename}: downloading...")
    path = hf_hub_download(
        repo_id=REPO,
        filename=filename,
        repo_type="dataset",
        local_dir=str(dest),
    )
    got = Path(path)
    print(f"  {filename}: {got.stat().st_size / 1024**3:.2f} GB")
    return got


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--version", choices=["v1", "v2"], default="v2")
    ap.add_argument("--split", choices=["train", "valid", "both"], default="both")
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = ap.parse_args()

    args.data_dir.mkdir(parents=True, exist_ok=True)
    print(f"{REPO} ({args.version}) -> {args.data_dir}")

    splits = ["valid", "train"] if args.split == "both" else [args.split]
    paths = {}
    for split in splits:  # valid first: small, so failures surface fast
        paths[split] = fetch(FILES[args.version][split], args.data_dir)

    print("\ndone:")
    for split, path in paths.items():
        print(f"  {split:>5}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
