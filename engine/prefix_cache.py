"""Prefix caching: reuse KV blocks across requests that share a prefix.

A serving workload is full of repeated prefixes -- the same system prompt on
every request, the same few-shot examples, the same conversation replayed with
one more turn. Prefilling them again on every request is pure waste, and the
waste scales with prefix length: an 800-token system prompt costs 800 tokens of
prefill per request and is byte-for-byte identical every time.

**Content-addressed blocks, not a literal radix tree.** Each full block is keyed
by `hash(parent_hash, block_tokens)`, chaining the hash along the sequence so a
block's identity includes everything before it. Two requests reach the same key
only if their entire preceding context matches, which is exactly the condition
for their KV to be interchangeable.

The alternative is SGLang's radix tree, which matches at token rather than block
granularity and so recovers a little more on conversational traffic. vLLM's
documentation notes that block hashing with LRU "effectively implements the
exact policy as RadixAttention" for full attention, and the hash scheme drops
straight onto a refcounted block allocator that already exists here. Matching at
token granularity would mean partial blocks, and a partial block cannot be
shared without copy-on-write on every reuse.

**Only full blocks are cacheable.** A partially filled block's contents depend
on tokens that have not arrived yet, so its hash is not yet determined. The tail
block of every sequence is therefore private.

**Eviction is LRU over unpinned blocks.** The cache holds one reference on every
block it tracks, so a cached prefix survives the request that created it. A
block referenced by a *running* sequence has a higher count and is skipped --
evicting it would pull KV out from under a live generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from engine.block_manager import PagedCachePool, PagedKVCache

#: Seed for the hash chain. Any fixed value works; naming it makes clear the
#: chain has a defined start rather than an accidental one.
ROOT_HASH = 0x9E3779B97F4A7C15


def block_hash(parent: int, tokens: tuple[int, ...]) -> int:
    """Key for a block, chained onto everything before it.

    Chaining is what makes the key safe. Hashing only the block's own tokens
    would collide two blocks with identical content but different history --
    and their KV differs, because attention inside a block depends on the
    context preceding it. Reusing one for the other silently conditions a
    generation on the wrong text.
    """
    return hash((parent, tokens))


@dataclass
class CacheEntry:
    block_id: int
    parent_hash: int
    #: Monotonic counter, not wall time: two blocks touched in the same
    #: millisecond still need a defined order for LRU.
    last_used: int = 0
    #: Blocks further from the root are cheaper to lose -- their prefix is
    #: shared by fewer sequences. Used to break LRU ties.
    depth: int = 0


@dataclass
class PrefixCacheStats:
    queries: int = 0
    hit_blocks: int = 0
    query_blocks: int = 0
    evictions: int = 0
    entries: int = 0

    @property
    def block_hit_rate(self) -> float:
        """Fraction of requested prefix blocks served from cache.

        Reported in blocks rather than requests: a request that reuses 40 of
        50 prefix blocks is mostly a hit, and a per-request hit/miss counter
        would score it the same as one that reused nothing.
        """
        return self.hit_blocks / self.query_blocks if self.query_blocks else 0.0

    @property
    def tokens_saved(self) -> int:
        return self.hit_blocks

    def to_dict(self, block_size: int = 16) -> dict:
        return {
            "queries": self.queries,
            "hit_blocks": self.hit_blocks,
            "query_blocks": self.query_blocks,
            "block_hit_rate": round(self.block_hit_rate, 4),
            "prefill_tokens_saved": self.hit_blocks * block_size,
            "evictions": self.evictions,
            "entries": self.entries,
        }


class PrefixCache:
    """A content-addressed store of full KV blocks, over a paged pool."""

    def __init__(self, pool: PagedCachePool, enabled: bool = True) -> None:
        self.pool = pool
        self.enabled = enabled
        self._entries: dict[int, CacheEntry] = {}  # hash -> entry
        self._by_block: dict[int, int] = {}  # block id -> hash
        self._clock = 0
        self.stats = PrefixCacheStats()

    # -- keying -------------------------------------------------------------

    def hash_chain(self, token_ids: list[int]) -> list[int]:
        """Chained hashes for each *full* block of `token_ids`.

        The trailing partial block is deliberately omitted: its contents are
        not yet final, so it has no stable identity.
        """
        bs = self.pool.block_size
        hashes: list[int] = []
        parent = ROOT_HASH
        for start in range(0, len(token_ids) - bs + 1, bs):
            parent = block_hash(parent, tuple(token_ids[start : start + bs]))
            hashes.append(parent)
        return hashes

    # -- lookup -------------------------------------------------------------

    def match(self, token_ids: list[int]) -> list[int]:
        """Physical blocks for the longest cached prefix of `token_ids`.

        Stops at the first miss rather than continuing to look. A gap in the
        chain means every later block's key was computed from a parent this
        cache does not hold, so nothing beyond the gap can match anyway.

        Each returned block gains a reference, transferred to the caller.
        """
        if not self.enabled:
            return []

        hashes = self.hash_chain(token_ids)
        self.stats.queries += 1
        self.stats.query_blocks += len(hashes)

        matched: list[int] = []
        for h in hashes:
            entry = self._entries.get(h)
            if entry is None:
                break
            self._clock += 1
            entry.last_used = self._clock
            self.pool.allocator.incref(entry.block_id)
            matched.append(entry.block_id)

        self.stats.hit_blocks += len(matched)
        return matched

    # -- insertion ----------------------------------------------------------

    def insert(self, token_ids: list[int], block_table: list[int]) -> None:
        """Register a sequence's full blocks under their chained hashes.

        Only blocks fully covered by `token_ids` are registered. The cache
        takes its own reference on each new block, which is what lets a cached
        prefix outlive the request that produced it.
        """
        if not self.enabled:
            return

        bs = self.pool.block_size
        hashes = self.hash_chain(token_ids)
        for depth, h in enumerate(hashes):
            if depth >= len(block_table):
                break
            block = block_table[depth]
            existing = self._entries.get(h)
            if existing is not None:
                # Already cached, possibly under a different physical block if
                # two sequences prefilled the same prefix concurrently. Keep
                # the incumbent: swapping would orphan references to it.
                self._clock += 1
                existing.last_used = self._clock
                continue
            if block in self._by_block:
                # This physical block is already tracked under another key --
                # it belongs to a different prefix. Do not double-register.
                continue

            self.pool.allocator.incref(block)
            self._clock += 1
            self._entries[h] = CacheEntry(
                block_id=block, parent_hash=ROOT_HASH if depth == 0 else hashes[depth - 1],
                last_used=self._clock, depth=depth,
            )
            self._by_block[block] = h
        self.stats.entries = len(self._entries)
        _ = bs  # block size is implicit in hash_chain

    # -- eviction -----------------------------------------------------------

    def evict(self, num_blocks: int = 1) -> int:
        """Release up to `num_blocks` cached blocks, coldest first.

        Only blocks whose sole reference is the cache's own are eligible: a
        higher count means a running sequence is reading them, and evicting
        those would pull KV out from under a live generation.

        Ties on last-used are broken by depth, evicting the block furthest
        from the root first -- its prefix is shared by fewer sequences, so
        losing it costs the least.
        """
        evictable = [
            (e.last_used, -e.depth, h)
            for h, e in self._entries.items()
            if self.pool.allocator.ref_count(e.block_id) == 1
        ]
        if not evictable:
            return 0

        evictable.sort()
        freed = 0
        for _, _, h in evictable[:num_blocks]:
            entry = self._entries.pop(h)
            self._by_block.pop(entry.block_id, None)
            self.pool.allocator.decref(entry.block_id)
            freed += 1
        self.stats.evictions += freed
        self.stats.entries = len(self._entries)
        return freed

    def ensure_free(self, num_blocks: int) -> bool:
        """Evict until the pool has `num_blocks` free, if it can."""
        while self.pool.allocator.num_free < num_blocks:
            if self.evict(num_blocks - self.pool.allocator.num_free) == 0:
                return False
        return True

    def clear(self) -> None:
        for h, entry in list(self._entries.items()):
            self.pool.allocator.decref(entry.block_id)
            del self._entries[h]
        self._by_block.clear()
        self.stats.entries = 0

    # -- integration --------------------------------------------------------

    def attach(self, cache: PagedKVCache, blocks: list[int]) -> int:
        """Seed a fresh sequence with already-populated blocks.

        Returns the number of tokens the sequence starts with, so the caller
        knows how much of the prompt still needs prefilling. The references
        were taken by `match`; this transfers them onto the sequence.
        """
        if not blocks:
            return 0
        for block in blocks:
            cache._append_block(block)
        cache.num_tokens = len(blocks) * self.pool.block_size
        return cache.num_tokens

    def __repr__(self) -> str:
        return (
            f"PrefixCache({len(self._entries)} blocks, "
            f"hit rate {self.stats.block_hit_rate:.1%}, "
            f"{self.stats.evictions} evictions)"
        )
