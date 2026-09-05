"""Corpus -> flat token file -> training batches.

Tokenised text is stored as one flat `uint16` array per split, with documents
concatenated and separated by the EOS token. Two consequences worth stating:

**uint16, not int32.** A vocabulary of 8192 fits in 16 bits, so the token file
is half the size it would otherwise be -- ~1.0 GB instead of ~2.1 GB for
TinyStories. `prepare` refuses to write a vocabulary that would not fit rather
than silently truncating ids.

**Flat and packed, not padded per document.** Sampling a random window of
`seq_len` tokens from the stream means every position in every batch carries
gradient. Padding each story to a fixed length would waste most of the batch on
padding, since stories average ~200 tokens against a 1024-token context. Windows
do sometimes straddle a document boundary; the EOS token is what teaches the
model that the text before it does not predict the text after.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch import Tensor

from model.tokenizer import END_OF_TEXT, BPETokenizer

#: uint16 caps the vocabulary. Well above the 8192 this project uses.
MAX_VOCAB_FOR_UINT16 = 2**16

CHUNK_BYTES = 8 * 1024 * 1024
FLUSH_TOKENS = 8_000_000  # ~16 MB of uint16 held before writing


def stream_documents(path: Path, max_bytes: int | None = None) -> Iterator[str]:
    """Yield documents from a `<|endoftext|>`-separated corpus.

    Chunked, with the trailing partial document carried across reads so a story
    spanning a chunk boundary is not cut in half.
    """
    read = 0
    tail = ""
    with path.open("r", encoding="utf-8") as f:
        while True:
            want = CHUNK_BYTES
            if max_bytes is not None:
                # Shrink the read rather than checking after the fact, so
                # max_bytes is honoured to the character instead of rounding up
                # to the next 8 MB chunk.
                want = min(want, max_bytes - read)
                if want <= 0:
                    break
            chunk = f.read(want)
            if not chunk:
                break
            read += len(chunk)
            parts = (tail + chunk).split(END_OF_TEXT)
            tail = parts.pop()
            for doc in parts:
                doc = doc.strip()
                if doc:
                    yield doc
    if tail.strip():
        yield tail.strip()


def prepare(
    corpus: Path,
    tokenizer: BPETokenizer,
    out: Path,
    drop_first_document: bool = False,
    drop_last_document: bool = False,
    verbose: bool = True,
) -> dict:
    """Tokenise `corpus` into a flat uint16 file, and write a manifest beside it.

    `drop_first_document` / `drop_last_document` exist because TinyStories'
    official train/valid split is a byte offset, not a document boundary: the
    train file ends mid-word inside a story whose remainder opens the valid
    file. Dropping the fragment on each side removes the resulting overlap.

    Documents are encoded with `allowed_special=False` so a story that happens
    to contain the literal text "<|endoftext|>" cannot inject a spurious
    boundary. The separator is appended explicitly instead.
    """
    if tokenizer.vocab_size > MAX_VOCAB_FOR_UINT16:
        raise ValueError(
            f"vocab_size={tokenizer.vocab_size} does not fit in uint16; "
            f"widen the dtype in prepare() before training this tokenizer"
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    eos = tokenizer.eos_id

    docs = stream_documents(corpus)
    if drop_first_document:
        next(docs, None)

    n_tokens = 0
    n_docs = 0
    pending: str | None = None
    buf: list[int] = []
    t0 = time.perf_counter()

    with out.open("wb") as f:
        for doc in docs:
            # Hold one document back so the last can be dropped if asked.
            if drop_last_document:
                doc, pending = pending, doc
                if doc is None:
                    continue

            ids = tokenizer.encode(doc, allowed_special=False)
            ids.append(eos)
            buf.extend(ids)
            n_tokens += len(ids)
            n_docs += 1

            if len(buf) >= FLUSH_TOKENS:
                np.asarray(buf, dtype=np.uint16).tofile(f)
                buf.clear()
                if verbose:
                    rate = n_tokens / (time.perf_counter() - t0) / 1e6
                    print(
                        f"  {n_docs:>9,} docs  {n_tokens / 1e6:>7.1f}M tokens  "
                        f"{rate:.2f}M tok/s",
                        end="\r",
                    )

        if buf:
            np.asarray(buf, dtype=np.uint16).tofile(f)

    elapsed = time.perf_counter() - t0
    manifest = {
        "tokens": n_tokens,
        "documents": n_docs,
        "vocab_size": tokenizer.vocab_size,
        "dtype": "uint16",
        "eos_id": eos,
        "source": corpus.name,
        "source_bytes": corpus.stat().st_size,
        "bytes_per_token": corpus.stat().st_size / max(n_tokens, 1),
        "dropped_first_document": drop_first_document,
        "dropped_last_document": drop_last_document,
        "seconds": round(elapsed, 1),
    }
    out.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    if verbose:
        print(
            f"\n  {out.name}: {n_tokens:,} tokens from {n_docs:,} documents "
            f"({out.stat().st_size / 1024**3:.2f} GB) in {elapsed / 60:.1f} min"
        )
    return manifest


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


@dataclass
class TokenDataset:
    """Random fixed-length windows over a memory-mapped token file.

    The file is memory-mapped rather than loaded: 1 GB of tokens would
    otherwise sit in RAM alongside the model, and only a few thousand tokens
    are touched per step. The OS page cache keeps the hot pages resident and
    the rest costs nothing.

    Sampling is with replacement and without epoch boundaries. At ~524M tokens
    against roughly 17k steps x 32k tokens/step, a run sees each token about
    once in expectation, so imposing a strict epoch buys nothing and costs a
    shuffle over a billion-element index.
    """

    path: Path
    seq_len: int

    def __post_init__(self) -> None:
        self.tokens = np.memmap(self.path, dtype=np.uint16, mode="r")
        if len(self.tokens) <= self.seq_len:
            raise ValueError(
                f"{self.path.name} holds {len(self.tokens):,} tokens, "
                f"too few for seq_len={self.seq_len}"
            )

    def __len__(self) -> int:
        return len(self.tokens)

    def batch(
        self,
        batch_size: int,
        device: torch.device | str = "cpu",
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """One batch of shape (batch_size, seq_len), dtype int64.

        Returned as a single tensor rather than an (input, target) pair: the
        model shifts internally, so training is `model(tokens, labels=tokens)`.
        That costs one position of supervision per window -- 1023 predictions
        from 1024 tokens, or 0.1% -- in exchange for one tensor and one
        transfer instead of two.
        """
        high = len(self.tokens) - self.seq_len
        starts = torch.randint(high, (batch_size,), generator=generator).tolist()

        # np.stack on uint16 views, cast once. Building int64 per-window would
        # allocate 4x the memory and copy it twice.
        window = np.stack([self.tokens[i : i + self.seq_len] for i in starts])
        out = torch.from_numpy(window.astype(np.int64))

        if str(device) != "cpu":
            # pin_memory + non_blocking lets the copy overlap with compute.
            out = out.pin_memory().to(device, non_blocking=True)
        return out

    def manifest(self) -> dict | None:
        path = self.path.with_suffix(".json")
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
