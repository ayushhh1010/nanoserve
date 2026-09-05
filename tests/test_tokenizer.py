"""Tests for the hand-written byte-level BPE tokenizer.

The property that matters most is lossless round-tripping on *arbitrary* input,
including text no one trained on. A byte-level tokenizer has no UNK and no
out-of-vocabulary branch, so any input that fails to round-trip is a bug in the
merge logic rather than a limitation of the vocabulary -- which makes the test
sharp rather than aspirational.

`test_known_merges_on_the_classic_example` pins the exact merge sequence on a
corpus small enough to work out on paper. Everything else in this file could
pass with a subtly wrong tie-break rule; that one could not.
"""

from __future__ import annotations

import pytest

from model.tokenizer import (
    BYTE_VOCAB_SIZE,
    END_OF_TEXT,
    SPLIT_PATTERN,
    BPETokenizer,
    _apply_merge,
)

#: Enough distinct words that BPE does not exhaust its pairs immediately.
_WORDS = (
    "quick brown fox jumps lazy dog bright yellow flower garden river mountain "
    "castle dragon wizard silver golden ancient forest whisper thunder shadow "
    "crystal meadow harbour lantern compass voyage marble feather blossom"
).split()

CORPUS = [
    "Once upon a time there was a little girl named Lily.",
    "She saw a big red ball and wanted to play with it.",
    "The cat sat on the mat. The cat was happy.",
    "Tom and Lily went to the park to play with the ball.",
    "They were very happy and played all day long.",
] * 20


@pytest.fixture(scope="module")
def tok() -> BPETokenizer:
    return BPETokenizer.train(CORPUS, vocab_size=400)


# ---------------------------------------------------------------------------
# The merge algorithm, pinned exactly
# ---------------------------------------------------------------------------


def test_known_merges_on_the_classic_example():
    """The textbook BPE example, worked out by hand.

    "aaabdaaabac" is a single pre-token (all letters, no boundaries), so it is
    11 bytes: a a a b d a a a b a c.

      pass 1: (a,a) occurs 4x, the clear winner        -> 256
              a a a b d a a a b a c  ->  X a b d X a b a c   where X = 256
      pass 2: (X,a) and (a,b) both occur 2x. The tie is broken on the pair
              itself, and 256 > 97, so (X,a) wins     -> 257
              X a b d X a b a c      ->  Y b d Y b a c       where Y = 257
      pass 3: (Y,b) occurs 2x                          -> 258
              Y b d Y b a c          ->  Z d Z a c           where Z = 258
      pass 4: every remaining pair occurs once, below min_frequency=2. Stop.

    Three merges, and "aaabdaaabac" encodes to [Z, d, Z, a, c].
    """
    t = BPETokenizer.train(["aaabdaaabac"], vocab_size=512, min_frequency=2)

    a, b, c, d = ord("a"), ord("b"), ord("c"), ord("d")
    assert list(t.merges.items()) == [
        ((a, a), 256),
        ((256, a), 257),
        ((257, b), 258),
    ]
    assert t.encode("aaabdaaabac") == [258, d, 258, a, c]
    assert t.decode([258, d, 258, a, c]) == "aaabdaaabac"


def test_merge_helper_handles_overlaps():
    """`aaa` contains (a,a) twice but they overlap; only one may be merged."""
    assert _apply_merge([1, 1, 1], (1, 1), 9) == [9, 1]
    assert _apply_merge([1, 1, 1, 1], (1, 1), 9) == [9, 9]
    assert _apply_merge([2, 1, 1, 2], (1, 1), 9) == [2, 9, 2]
    assert _apply_merge([1, 2, 1], (1, 1), 9) == [1, 2, 1]
    assert _apply_merge([], (1, 1), 9) == []


def test_training_is_deterministic():
    """Same corpus, same merges -- byte for byte.

    Without an explicit tie-break, pairs of equal frequency resolve in
    whatever order the counter happens to yield, and two runs produce
    different tokenizers from identical input.
    """
    a = BPETokenizer.train(CORPUS, vocab_size=400)
    b = BPETokenizer.train(CORPUS, vocab_size=400)
    assert a.merges == b.merges
    assert a.special_tokens == b.special_tokens


def test_vocab_size_accounts_for_every_id(tok: BPETokenizer):
    """Bytes + merges + specials, with no gaps and nothing double-counted."""
    assert tok.vocab_size == BYTE_VOCAB_SIZE + len(tok.merges) + len(tok.special_tokens)
    assert sorted(tok.vocab) == list(range(tok.vocab_size))


def test_vocab_size_is_a_target_not_a_guarantee():
    """`vocab_size` is an upper bound. A corpus can simply run out of pairs.

    CORPUS has only 40 distinct pre-tokens. After 96 merges every one of them
    is a single token, no adjacent pair exists anywhere, and training stops at
    353 rather than the requested 400 -- with min_frequency=1, so this is
    genuine exhaustion and not the frequency floor.

    Worth pinning because the failure mode it rules out is a loop that keeps
    "merging" nothing and pads the vocabulary with unreachable ids.
    """
    t = BPETokenizer.train(CORPUS, vocab_size=400, min_frequency=1)
    assert t.vocab_size < 400

    pieces = {p for line in CORPUS for p in SPLIT_PATTERN.findall(line)}
    assert len(pieces) == 40
    for piece in pieces:
        assert len(t.encode(piece, allowed_special=False)) == 1, (
            f"{piece!r} should have collapsed to one token"
        )


def test_target_vocab_is_reached_on_a_corpus_with_room():
    """When the corpus is rich enough, the requested size is hit exactly."""
    varied = [f"story number {i} about a {w} thing" for i, w in enumerate(_WORDS * 4)]
    t = BPETokenizer.train(varied, vocab_size=400, min_frequency=1)
    assert t.vocab_size == 400
    assert len(t.merges) == 400 - BYTE_VOCAB_SIZE - 1  # 1 special token


def test_min_frequency_stops_early_rather_than_padding():
    """A tiny corpus cannot fill a large vocabulary, and should not pretend to.

    Merges learned from a single occurrence memorise one training example.
    Stopping short is correct; the caller finds out via `vocab_size`.
    """
    t = BPETokenizer.train(["hello world"], vocab_size=5000, min_frequency=2)
    assert t.vocab_size < 5000


def test_vocab_size_below_byte_floor_is_rejected():
    with pytest.raises(ValueError, match="smaller than"):
        BPETokenizer.train(CORPUS, vocab_size=100)


# ---------------------------------------------------------------------------
# The byte-level guarantee
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Once upon a time there was a little girl.",
        "text the tokenizer has never seen: xyzzy plugh",
        "unicode: café naïve résumé Straße",
        "cyrillic and greek: Привет κόσμε",
        "cjk: 日本語のテキスト 中文字符",
        "emoji: 🚀🔥👍 family 👨‍👩‍👧‍👦",
        "math: ∀x∈ℝ, x²≥0 ∧ π≈3.14159",
        "whitespace\tand\nnewlines\r\nand   runs",
        "punctuation!?;:'\"[]{}()<>@#$%^&*",
        "numbers 0 42 1234567890 3.14 -7",
        "  leading and trailing spaces  ",
        "",
        " ",
        "\n",
        "a",
        "\x00\x01\x02 control bytes \x7f",
        "mixed 中文 and English and 🎉 in one line",
    ],
    ids=lambda s: (s[:22] or "empty").replace("\n", "\\n").replace("\r", "\\r"),
)
def test_roundtrip_is_lossless(tok: BPETokenizer, text: str):
    """Byte-level means there is no input that cannot be represented."""
    assert tok.decode(tok.encode(text)) == text


def test_every_id_is_in_range(tok: BPETokenizer):
    ids = tok.encode("Anything at all: 日本 🚀 ¿qué?")
    assert all(0 <= i < tok.vocab_size for i in ids)


def test_unseen_characters_decompose_to_bytes(tok: BPETokenizer):
    """No UNK: a character absent from training becomes its UTF-8 bytes."""
    ids = tok.encode("𝕏")  # 4 UTF-8 bytes, certainly not in CORPUS
    assert all(i < BYTE_VOCAB_SIZE for i in ids)
    assert len(ids) == 4
    assert tok.decode(ids) == "𝕏"


# ---------------------------------------------------------------------------
# Pre-tokenization
# ---------------------------------------------------------------------------


def test_split_pattern_covers_input_exactly():
    """The split must be a partition -- lose a character here and it is gone."""
    for text in ["Hello, world!", "  spaced  out  ", "don't stop", "a1b2 3.14", "🚀 x"]:
        assert "".join(SPLIT_PATTERN.findall(text)) == text


def test_leading_space_stays_attached():
    """` the` and `the` must be distinguishable, or decoding has to guess."""
    assert SPLIT_PATTERN.findall("the the") == ["the", " the"]


def test_contractions_split_as_units():
    assert SPLIT_PATTERN.findall("don't") == ["don", "'t"]
    assert SPLIT_PATTERN.findall("they're") == ["they", "'re"]


def test_merges_never_cross_a_pretoken_boundary():
    """The reason pre-tokenization exists.

    Trained on text where ". The" is extremely frequent, an unconstrained BPE
    would learn it as one token. Every learned token must decode to something
    the splitter would have produced as a single piece.
    """
    corpus = ["The cat sat. The dog ran. The bird flew."] * 100
    t = BPETokenizer.train(corpus, vocab_size=340)

    for tid in range(BYTE_VOCAB_SIZE, BYTE_VOCAB_SIZE + len(t.merges)):
        piece = t.vocab[tid].decode("utf-8", errors="replace")
        assert len(SPLIT_PATTERN.findall(piece)) == 1, (
            f"token {tid} = {piece!r} spans a pre-token boundary"
        )


# ---------------------------------------------------------------------------
# Special tokens
# ---------------------------------------------------------------------------


def test_special_token_encodes_as_one_id(tok: BPETokenizer):
    ids = tok.encode(f"a{END_OF_TEXT}b")
    assert tok.eos_id in ids
    assert ids.count(tok.eos_id) == 1
    assert tok.decode(ids) == f"a{END_OF_TEXT}b"


def test_special_tokens_can_be_disabled_for_untrusted_input(tok: BPETokenizer):
    """A user typing the literal text must not be able to inject a boundary."""
    ids = tok.encode(END_OF_TEXT, allowed_special=False)
    assert tok.eos_id not in ids
    assert tok.decode(ids) == END_OF_TEXT
    assert len(ids) > 1


def test_eos_is_the_highest_id(tok: BPETokenizer):
    """Specials sit above the merges, so truncating the vocab never drops them."""
    assert tok.eos_id == tok.vocab_size - 1


# ---------------------------------------------------------------------------
# Persistence and caching
# ---------------------------------------------------------------------------


def test_save_load_roundtrip(tok: BPETokenizer, tmp_path):
    path = tmp_path / "tokenizer.json"
    tok.save(path)
    loaded = BPETokenizer.load(path)

    assert loaded.merges == tok.merges
    assert loaded.special_tokens == tok.special_tokens
    assert loaded.vocab_size == tok.vocab_size

    text = "Lily and Tom played with the red ball. 日本 🚀"
    assert loaded.encode(text) == tok.encode(text)
    assert loaded.decode(loaded.encode(text)) == text


def test_load_rejects_unknown_format(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"version": 99, "merges": [], "special_tokens": {}}', encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported tokenizer format"):
        BPETokenizer.load(path)


def test_cache_does_not_change_results(tok: BPETokenizer):
    """Encoding is memoised; a cache hit must equal a cache miss."""
    text = "the cat sat on the mat with the cat"
    cold = BPETokenizer(merges=tok.merges, special_tokens=tok.special_tokens).encode(text)
    warm = tok.encode(text)
    assert cold == warm
    assert tok.encode(text) == warm  # second call, now certainly cached


# ---------------------------------------------------------------------------
# Does it actually compress?
# ---------------------------------------------------------------------------


def test_learned_merges_beat_raw_bytes(tok: BPETokenizer):
    """A tokenizer that does not compress its training domain is broken."""
    text = " ".join(CORPUS[:5])
    n_tokens = len(tok.encode(text))
    n_bytes = len(text.encode("utf-8"))

    ratio = n_bytes / n_tokens
    assert ratio > 2.0, f"only {ratio:.2f} bytes/token -- merges are not being applied"


def test_larger_vocab_compresses_better():
    """More merges must mean fewer tokens. If not, ranks are being misapplied."""
    text = " ".join(CORPUS[:10])
    small = len(BPETokenizer.train(CORPUS, vocab_size=300).encode(text))
    large = len(BPETokenizer.train(CORPUS, vocab_size=500).encode(text))
    assert large < small, f"vocab 500 gave {large} tokens, vocab 300 gave {small}"
