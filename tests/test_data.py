"""Tests for the corpus -> token file -> batch pipeline.

The expensive failures here are silent ones. A document splitter that drops a
story every 8 MB, or a uint16 cast that wraps ids above 65535, produces a
training file that looks entirely normal and a model that is quietly worse.
Both are asserted directly.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from model.tokenizer import END_OF_TEXT, BPETokenizer
from model.train import data as D

STORIES = [
    "Once upon a time there was a little girl named Lily.",
    "Tom had a big red ball and he liked to play with it every day.",
    "The cat sat on the mat and the dog ran in the park.",
    "Lily and Tom went to the shop to buy some bread and milk.",
]


def write_corpus(path, stories, trailing_separator: bool = True):
    body = f"\n{END_OF_TEXT}\n".join(stories)
    if trailing_separator:
        body += f"\n{END_OF_TEXT}\n"
    path.write_text(body, encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def tok() -> BPETokenizer:
    return BPETokenizer.train(STORIES * 30, vocab_size=400)


# ---------------------------------------------------------------------------
# Document streaming
# ---------------------------------------------------------------------------


def test_streams_every_document(tmp_path):
    path = write_corpus(tmp_path / "c.txt", STORIES)
    assert list(D.stream_documents(path)) == STORIES


def test_documents_spanning_a_chunk_boundary_are_not_split(tmp_path, monkeypatch):
    """The bug this rules out loses one story per chunk, silently.

    Chunked reading is only correct if the trailing partial document is carried
    into the next read. Forcing a tiny chunk size makes almost every document
    straddle a boundary.
    """
    stories = [f"story number {i} " + "word " * 40 for i in range(50)]
    path = write_corpus(tmp_path / "c.txt", stories)

    monkeypatch.setattr(D, "CHUNK_BYTES", 64)  # far smaller than any document
    got = list(D.stream_documents(path))

    assert len(got) == len(stories)
    assert got == [s.strip() for s in stories]


def test_no_trailing_separator_still_yields_the_last_document(tmp_path):
    path = write_corpus(tmp_path / "c.txt", STORIES, trailing_separator=False)
    assert list(D.stream_documents(path)) == STORIES


def test_blank_documents_are_skipped(tmp_path):
    path = write_corpus(tmp_path / "c.txt", ["real story", "", "   ", "another story"])
    assert list(D.stream_documents(path)) == ["real story", "another story"]


def test_max_bytes_truncates(tmp_path):
    stories = [f"story {i} with some words in it" for i in range(200)]
    path = write_corpus(tmp_path / "c.txt", stories)
    partial = list(D.stream_documents(path, max_bytes=200))
    assert 0 < len(partial) < len(stories)


# ---------------------------------------------------------------------------
# Tokenising to a flat file
# ---------------------------------------------------------------------------


def test_prepare_writes_uint16_and_a_manifest(tmp_path, tok):
    corpus = write_corpus(tmp_path / "c.txt", STORIES)
    out = tmp_path / "tokens.bin"
    manifest = D.prepare(corpus, tok, out, verbose=False)

    tokens = np.memmap(out, dtype=np.uint16, mode="r")
    assert len(tokens) == manifest["tokens"]
    assert out.stat().st_size == manifest["tokens"] * 2  # 2 bytes per token
    assert manifest["documents"] == len(STORIES)

    on_disk = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert on_disk == manifest
    assert on_disk["dtype"] == "uint16"
    assert on_disk["vocab_size"] == tok.vocab_size


def test_every_document_ends_with_eos(tmp_path, tok):
    """EOS is what teaches the model that a story has finished."""
    corpus = write_corpus(tmp_path / "c.txt", STORIES)
    out = tmp_path / "tokens.bin"
    D.prepare(corpus, tok, out, verbose=False)

    tokens = np.asarray(np.memmap(out, dtype=np.uint16, mode="r"))
    assert tokens[-1] == tok.eos_id
    assert int((tokens == tok.eos_id).sum()) == len(STORIES)


def test_tokens_decode_back_to_the_original_stories(tmp_path, tok):
    """End to end: nothing is lost between the .txt and the .bin."""
    corpus = write_corpus(tmp_path / "c.txt", STORIES)
    out = tmp_path / "tokens.bin"
    D.prepare(corpus, tok, out, verbose=False)

    tokens = np.asarray(np.memmap(out, dtype=np.uint16, mode="r")).tolist()
    text = tok.decode(tokens)
    assert text == END_OF_TEXT.join(STORIES) + END_OF_TEXT


def test_literal_eos_in_a_story_cannot_inject_a_boundary(tmp_path, tok):
    """A story containing the separator's text must not become two documents."""
    sneaky = f"Lily said {END_OF_TEXT} and then ran away"
    corpus = write_corpus(tmp_path / "c.txt", [STORIES[0], sneaky])

    # The corpus splitter sees the literal text and does split on it -- that is
    # the file format. What must not happen is a *second* EOS appearing inside
    # an encoded document.
    out = tmp_path / "tokens.bin"
    D.prepare(corpus, tok, out, verbose=False)
    tokens = np.asarray(np.memmap(out, dtype=np.uint16, mode="r"))

    manifest = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert int((tokens == tok.eos_id).sum()) == manifest["documents"]


def test_boundary_documents_can_be_dropped(tmp_path, tok):
    """TinyStories' official split cuts a story in half across the two files."""
    corpus = write_corpus(tmp_path / "c.txt", STORIES)

    full = D.prepare(corpus, tok, tmp_path / "a.bin", verbose=False)
    no_first = D.prepare(corpus, tok, tmp_path / "b.bin", drop_first_document=True, verbose=False)
    no_last = D.prepare(corpus, tok, tmp_path / "c.bin", drop_last_document=True, verbose=False)
    neither = D.prepare(
        corpus,
        tok,
        tmp_path / "d.bin",
        drop_first_document=True,
        drop_last_document=True,
        verbose=False,
    )

    assert full["documents"] == len(STORIES)
    assert no_first["documents"] == len(STORIES) - 1
    assert no_last["documents"] == len(STORIES) - 1
    assert neither["documents"] == len(STORIES) - 2

    kept = tok.decode(np.asarray(np.memmap(tmp_path / "b.bin", dtype=np.uint16, mode="r")).tolist())
    assert STORIES[0] not in kept
    assert STORIES[1] in kept


def test_prepare_refuses_a_vocab_too_large_for_uint16(tmp_path, tok):
    """Silently wrapping ids at 65536 would poison every downstream number."""
    corpus = write_corpus(tmp_path / "c.txt", STORIES)

    class Oversized:
        vocab_size = D.MAX_VOCAB_FOR_UINT16 + 1
        eos_id = 0

    with pytest.raises(ValueError, match="uint16"):
        D.prepare(corpus, Oversized(), tmp_path / "x.bin", verbose=False)


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------


@pytest.fixture
def dataset(tmp_path, tok) -> D.TokenDataset:
    corpus = write_corpus(tmp_path / "c.txt", STORIES * 200)
    out = tmp_path / "tokens.bin"
    D.prepare(corpus, tok, out, verbose=False)
    return D.TokenDataset(out, seq_len=32)


def test_batch_shape_and_dtype(dataset):
    batch = dataset.batch(8)
    assert batch.shape == (8, 32)
    assert batch.dtype == torch.int64  # embedding lookup requires int64


def test_batch_ids_are_within_vocab(dataset, tok):
    batch = dataset.batch(16)
    assert int(batch.min()) >= 0
    assert int(batch.max()) < tok.vocab_size


def test_batch_is_reproducible_with_a_generator(dataset):
    """Needed for resumable training: the same seed must replay the same data."""
    g1 = torch.Generator().manual_seed(1234)
    g2 = torch.Generator().manual_seed(1234)
    assert torch.equal(dataset.batch(4, generator=g1), dataset.batch(4, generator=g2))

    g3 = torch.Generator().manual_seed(999)
    assert not torch.equal(dataset.batch(4, generator=g1), dataset.batch(4, generator=g3))


def test_windows_are_contiguous_slices_of_the_stream(dataset):
    """A window must be real adjacent text, not a gather of scattered tokens."""
    raw = np.asarray(dataset.tokens)
    batch = dataset.batch(4, generator=torch.Generator().manual_seed(7))

    for row in batch:
        row_np = row.numpy().astype(np.uint16)
        # Find this window in the stream and confirm it appears verbatim.
        first = row_np[0]
        candidates = np.flatnonzero(raw[: len(raw) - dataset.seq_len] == first)
        assert any(
            np.array_equal(raw[i : i + dataset.seq_len], row_np) for i in candidates
        ), "batch row is not a contiguous slice of the token stream"


def test_dataset_rejects_a_file_shorter_than_one_window(tmp_path, tok):
    corpus = write_corpus(tmp_path / "c.txt", ["tiny"])
    out = tmp_path / "tokens.bin"
    D.prepare(corpus, tok, out, verbose=False)
    with pytest.raises(ValueError, match="too few"):
        D.TokenDataset(out, seq_len=4096)


def test_manifest_is_reachable_from_the_dataset(dataset):
    m = dataset.manifest()
    assert m is not None
    assert m["tokens"] == len(dataset)
