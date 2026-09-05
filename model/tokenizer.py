"""A byte-level BPE tokenizer, written by hand.

No `tokenizers`, no `sentencepiece`. The training loop, the merge application
and the encoder are all here.

Three design choices, each with a reason:

**Byte-level base vocabulary.** The 256 byte values are the starting alphabet,
so every possible input is representable and there is no UNK token and no
out-of-vocabulary path to get wrong. Text is UTF-8 encoded first, so a
character outside the training data decomposes into bytes the model has seen.
This is what GPT-2, GPT-4 and Llama 3 all do.

**Pre-tokenization before BPE.** Text is split on a regex first, and merges
never cross those boundaries. Without it, BPE happily learns `". The"` as one
token and the vocabulary fills with punctuation-glued fragments that only match
in one context.

**Training on a frequency table, not on the corpus.** Merging on a 550M-token
stream directly is the trap that makes hand-written BPE look impractical.
Collapsing to unique pre-token -> count first turns 550M tokens into a few
hundred thousand rows, and every merge then costs work proportional to the
number of *distinct* words containing the pair, not to corpus size.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

#: Splits text before BPE runs. Merges never cross a match boundary.
#:
#: This is the GPT-2 pattern rewritten for Python's stdlib `re`, which has no
#: `\p{L}`. `[^\W\d_]` is the standard equivalent: `\W` removes non-word
#: characters, then digits and underscore are excluded, leaving letters. The
#: alternative is a dependency on the third-party `regex` module, which is not
#: worth taking for one character class.
#:
#: The leading optional space is what keeps ` the` and `the` distinct, so the
#: tokenizer never has to guess where whitespace went at decode time.
SPLIT_PATTERN = re.compile(
    r"'(?:[sdmt]|ll|ve|re)"  # contractions: 's 'd 'm 't 'll 've 're
    r"| ?[^\W\d_]+"  # optional space + letters
    r"| ?\d+"  # optional space + digits
    r"| ?[^\s\w]+"  # optional space + punctuation/symbols
    r"|\s+(?!\S)"  # trailing whitespace
    r"|\s+"  # any other whitespace run
)

#: Marks a document boundary. Doubles as the generation stop token.
END_OF_TEXT = "<|endoftext|>"

BYTE_VOCAB_SIZE = 256


class BPETokenizer:
    """Byte-level BPE. Train it, or load one that was trained."""

    def __init__(
        self,
        merges: dict[tuple[int, int], int] | None = None,
        special_tokens: dict[str, int] | None = None,
    ) -> None:
        # (left_id, right_id) -> merged_id. Insertion order IS merge rank:
        # earlier merges must be applied first, or encoding diverges from
        # training and the same string tokenises differently.
        self.merges: dict[tuple[int, int], int] = merges or {}
        self.special_tokens: dict[str, int] = special_tokens or {}
        self._rebuild()

    # -- derived tables -----------------------------------------------------

    def _rebuild(self) -> None:
        """Recompute everything derivable from `merges` and `special_tokens`."""
        # id -> the bytes it stands for. Needed only for decoding.
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(BYTE_VOCAB_SIZE)}
        for (a, b), new_id in self.merges.items():
            self.vocab[new_id] = self.vocab[a] + self.vocab[b]
        for text, tid in self.special_tokens.items():
            self.vocab[tid] = text.encode("utf-8")

        self.id_to_special = {v: k for k, v in self.special_tokens.items()}
        # Rank = position in the merge order. Encoding must apply the
        # lowest-rank applicable merge first.
        self.ranks: dict[tuple[int, int], int] = {p: i for i, p in enumerate(self.merges)}
        self._cache: dict[bytes, list[int]] = {}

        if self.special_tokens:
            self._special_re = re.compile(
                "(" + "|".join(re.escape(t) for t in self.special_tokens) + ")"
            )
        else:
            self._special_re = None

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def eos_id(self) -> int:
        return self.special_tokens[END_OF_TEXT]

    # -- training -----------------------------------------------------------

    @classmethod
    def train(
        cls,
        corpus: Iterable[str],
        vocab_size: int,
        special_tokens: Iterable[str] = (END_OF_TEXT,),
        min_frequency: int = 2,
        verbose: bool = False,
    ) -> BPETokenizer:
        """Learn merges from `corpus` until the vocabulary reaches `vocab_size`.

        `corpus` is an iterable of strings so it can be a generator over a file
        rather than a list held in memory.

        `min_frequency` stops the loop when the best remaining pair is rarer
        than this. A merge learned from a single occurrence is memorisation of
        one training example, not a useful subword.
        """
        specials = list(special_tokens)
        n_merges = vocab_size - BYTE_VOCAB_SIZE - len(specials)
        if n_merges < 0:
            raise ValueError(
                f"vocab_size={vocab_size} is smaller than the {BYTE_VOCAB_SIZE} byte "
                f"tokens plus {len(specials)} special tokens"
            )

        # --- collapse the corpus to unique pre-tokens and their counts ------
        # This is the step that makes hand-written BPE tractable. Everything
        # below is proportional to the number of *distinct* pre-tokens.
        counts: Counter[bytes] = Counter()
        for text in corpus:
            for piece in SPLIT_PATTERN.findall(text):
                counts[piece.encode("utf-8")] += 1

        if verbose:
            total = sum(counts.values())
            print(f"  {total:,} pre-tokens, {len(counts):,} distinct")

        words: list[list[int]] = [list(w) for w in counts]
        freqs: list[int] = list(counts.values())

        # --- index which words contain which pairs --------------------------
        pair_counts: Counter[tuple[int, int]] = Counter()
        where: defaultdict[tuple[int, int], set[int]] = defaultdict(set)
        for wi, word in enumerate(words):
            f = freqs[wi]
            for pair in zip(word, word[1:]):
                pair_counts[pair] += f
                where[pair].add(wi)

        # --- merge ----------------------------------------------------------
        merges: dict[tuple[int, int], int] = {}
        next_id = BYTE_VOCAB_SIZE

        for step in range(n_merges):
            if not pair_counts:
                break
            # Tie-break on the pair itself so a rerun on the same corpus
            # produces byte-identical merges.
            best = max(pair_counts, key=lambda p: (pair_counts[p], p))
            if pair_counts[best] < min_frequency:
                if verbose:
                    print(f"  stopping at {step} merges: best pair seen {pair_counts[best]}x")
                break

            merges[best] = next_id
            affected = where[best]

            for wi in list(affected):
                word, f = words[wi], freqs[wi]

                # Retract this word's contribution, rewrite it, then re-add.
                # Simpler and far less error-prone than patching only the
                # neighbours of each merge site, and the cost is the same order.
                for p in zip(word, word[1:]):
                    pair_counts[p] -= f
                    if pair_counts[p] <= 0:
                        del pair_counts[p]
                    where[p].discard(wi)

                new_word = _apply_merge(word, best, next_id)
                words[wi] = new_word

                for p in zip(new_word, new_word[1:]):
                    pair_counts[p] += f
                    where[p].add(wi)

            where.pop(best, None)
            next_id += 1

            if verbose and (step + 1) % 1000 == 0:
                print(f"  {step + 1:,}/{n_merges:,} merges, {len(pair_counts):,} pairs live")

        special_ids = {tok: next_id + i for i, tok in enumerate(specials)}
        return cls(merges=merges, special_tokens=special_ids)

    # -- encoding -----------------------------------------------------------

    def _encode_piece(self, piece: bytes) -> list[int]:
        """BPE one pre-token. Memoised -- most words recur constantly."""
        cached = self._cache.get(piece)
        if cached is not None:
            return cached

        ids = list(piece)
        while len(ids) > 1:
            # Apply the merge that was learned earliest among those available.
            # Taking any other order would produce a different tokenisation
            # than training did.
            best, best_rank = None, None
            for pair in zip(ids, ids[1:]):
                rank = self.ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best, best_rank = pair, rank
            if best is None:
                break
            ids = _apply_merge(ids, best, self.merges[best])

        self._cache[piece] = ids
        return ids

    def encode(self, text: str, allowed_special: bool = True) -> list[int]:
        """Text -> token ids.

        With `allowed_special`, occurrences of a special token's literal text
        become that single token. Turn it off for untrusted input, so a user
        who types "<|endoftext|>" cannot inject a document boundary -- the
        string then encodes as ordinary bytes.
        """
        if not allowed_special or self._special_re is None:
            return self._encode_ordinary(text)

        out: list[int] = []
        for chunk in self._special_re.split(text):
            if not chunk:
                continue
            tid = self.special_tokens.get(chunk)
            if tid is not None:
                out.append(tid)
            else:
                out.extend(self._encode_ordinary(chunk))
        return out

    def _encode_ordinary(self, text: str) -> list[int]:
        out: list[int] = []
        for piece in SPLIT_PATTERN.findall(text):
            out.extend(self._encode_piece(piece.encode("utf-8")))
        return out

    def decode(self, ids: Iterable[int]) -> str:
        """Token ids -> text.

        Bytes are concatenated before decoding, not decoded per token: one
        multi-byte character is frequently split across two tokens, so
        per-token decoding would produce replacement characters mid-word.
        `errors="replace"` then only triggers on a genuinely truncated
        sequence -- which happens legitimately when streaming a partial
        generation.
        """
        buf = b"".join(self.vocab[i] for i in ids)
        return buf.decode("utf-8", errors="replace")

    # -- persistence --------------------------------------------------------

    def save(self, path: str | Path) -> None:
        payload = {
            "version": 1,
            "vocab_size": self.vocab_size,
            "special_tokens": self.special_tokens,
            # Stored as [left, right, merged] triples in rank order. The order
            # is the tokenizer; a shuffled file is a different tokenizer.
            "merges": [[a, b, i] for (a, b), i in self.merges.items()],
        }
        Path(path).write_text(json.dumps(payload, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> BPETokenizer:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("version") != 1:
            raise ValueError(f"unsupported tokenizer format: {payload.get('version')!r}")
        merges = {(a, b): i for a, b, i in payload["merges"]}
        return cls(merges=merges, special_tokens=payload["special_tokens"])

    def __repr__(self) -> str:
        return (
            f"BPETokenizer(vocab_size={self.vocab_size}, merges={len(self.merges)}, "
            f"special={list(self.special_tokens)})"
        )


def _apply_merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every non-overlapping occurrence of `pair` in `ids` with `new_id`."""
    out: list[int] = []
    i, n = 0, len(ids)
    a, b = pair
    while i < n:
        if i < n - 1 and ids[i] == a and ids[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out
