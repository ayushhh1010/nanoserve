"""Statically pre-allocated, contiguous KV cache -- and the waste it creates.

Step 2 of the engine. Two things happen here, and the second is the point.

**It gets faster.** `DynamicCache` calls `torch.cat` on every decode step,
which reallocates and copies the entire cache each token: O(n) work per token,
O(n^2) over a generation. Pre-allocating the whole slab once and writing into a
slice makes the append O(1).

**It gets wasteful, measurably.** A contiguous cache has to reserve for the
worst case, because a sequence's KV must live in one unbroken run and nobody
knows in advance how many tokens will be generated. So every sequence takes
`max_seq_len` worth of memory regardless of how much it uses. The published
measurement for this design is 60-80% of KV memory wasted, with effective
utilisation as low as 20%.

That waste decomposes three ways, and `PoolStats` reports each separately
because they have different fixes:

  reserved     slots a running sequence will legitimately fill later. Real,
               but unavailable to anyone else in the meantime.
  internal     reserved for a max length the sequence never reaches. Pure
               loss -- this is the bulk of it.
  external     whole slots that are free but unusable, because a contiguous
               allocator hands out fixed-size runs.

Paged attention (step 3) bounds internal fragmentation to at most
`block_size - 1` tokens per sequence and removes external fragmentation
entirely, since uniform blocks are interchangeable. The comparison between
this file's numbers and that one's is the memory headline of the phase.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from model.config import NanoConfig


@dataclass
class PoolStats:
    """Memory accounting for a KV pool, in tokens and bytes."""

    slots_total: int
    slots_allocated: int
    tokens_per_slot: int
    tokens_used: int
    #: Tokens a running sequence is expected still to generate.
    tokens_reserved: int
    bytes_per_token: int

    @property
    def tokens_capacity(self) -> int:
        return self.slots_total * self.tokens_per_slot

    @property
    def tokens_allocated(self) -> int:
        """Capacity handed out to sequences, used or not."""
        return self.slots_allocated * self.tokens_per_slot

    @property
    def internal_fragmentation(self) -> int:
        """Allocated, never going to be used. The bulk of the waste."""
        return max(0, self.tokens_allocated - self.tokens_used - self.tokens_reserved)

    @property
    def external_fragmentation(self) -> int:
        """Free capacity that a contiguous allocator cannot subdivide.

        Every unallocated slot is a full `max_seq_len` run. A request needing
        10 tokens still consumes one, so the remainder of that slot is
        unusable by anyone -- which is what external fragmentation means here.
        """
        return (self.slots_total - self.slots_allocated) * self.tokens_per_slot

    @property
    def utilization(self) -> float:
        """Fraction of *allocated* memory holding real tokens."""
        return self.tokens_used / self.tokens_allocated if self.tokens_allocated else 0.0

    @property
    def occupancy(self) -> float:
        """Fraction of the whole pool holding real tokens."""
        return self.tokens_used / self.tokens_capacity if self.tokens_capacity else 0.0

    @property
    def waste_fraction(self) -> float:
        """The number the literature quotes as 60-80% for this design."""
        return 1.0 - self.utilization

    def to_dict(self) -> dict:
        mb = 1024 * 1024
        return {
            "slots_total": self.slots_total,
            "slots_allocated": self.slots_allocated,
            "tokens_per_slot": self.tokens_per_slot,
            "tokens_used": self.tokens_used,
            "tokens_reserved": self.tokens_reserved,
            "tokens_internal_frag": self.internal_fragmentation,
            "tokens_external_frag": self.external_fragmentation,
            "utilization": round(self.utilization, 4),
            "occupancy": round(self.occupancy, 4),
            "waste_fraction": round(self.waste_fraction, 4),
            "allocated_mb": round(self.tokens_allocated * self.bytes_per_token / mb, 2),
            "used_mb": round(self.tokens_used * self.bytes_per_token / mb, 2),
            "capacity_mb": round(self.tokens_capacity * self.bytes_per_token / mb, 2),
        }


class StaticKVCache:
    """One sequence's view of a slot in a `StaticCachePool`.

    Implements the model's `KVCache` protocol, so attention cannot tell the
    difference between this and the growing cache it replaces.
    """

    def __init__(self, pool: StaticCachePool, slot: int) -> None:
        self.pool = pool
        self.slot = slot
        self._pos = 0  # tokens written so far
        self._start = 0  # where the current forward pass began writing

    def __len__(self) -> int:
        return self._pos

    def update(self, layer_idx: int, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        n = k.shape[-2]

        # All layers of one forward pass write at the same offset. The position
        # advances once, on layer 0, rather than once per layer -- getting this
        # wrong scatters a single step's KV across n_layers different offsets
        # and produces garbage that still runs.
        if layer_idx == 0:
            self._start = self._pos
            self._pos += n
            if self._pos > self.pool.max_seq_len:
                raise RuntimeError(
                    f"sequence exceeded slot capacity: {self._pos} > "
                    f"{self.pool.max_seq_len} tokens"
                )

        start, end = self._start, self._start + n
        self.pool.k[layer_idx][self.slot, :, start:end] = k[0]
        self.pool.v[layer_idx][self.slot, :, start:end] = v[0]

        # A view into the slab, not a copy: attention reads the pool directly.
        return (
            self.pool.k[layer_idx][self.slot : self.slot + 1, :, :end],
            self.pool.v[layer_idx][self.slot : self.slot + 1, :, :end],
        )

    def free(self) -> None:
        self.pool.free(self)


class StaticCachePool:
    """A fixed number of fixed-size contiguous slots.

    This is the design the paged allocator replaces. It is not a strawman: it
    is what you write when the cache must be contiguous, and it is what
    pre-vLLM serving systems did.
    """

    def __init__(
        self,
        cfg: NanoConfig,
        num_slots: int,
        max_seq_len: int | None = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.cfg = cfg
        self.num_slots = num_slots
        self.max_seq_len = max_seq_len or cfg.max_position_embeddings
        self.device = torch.device(device)
        self.dtype = dtype

        shape = (num_slots, cfg.num_key_value_heads, self.max_seq_len, cfg.head_dim)
        # One slab per layer. Allocated once, up front -- which is the speed
        # win and the memory problem in the same decision.
        self.k = [torch.zeros(shape, device=self.device, dtype=dtype) for _ in range(cfg.num_hidden_layers)]
        self.v = [torch.zeros(shape, device=self.device, dtype=dtype) for _ in range(cfg.num_hidden_layers)]

        self._free: list[int] = list(range(num_slots))
        self._live: dict[int, StaticKVCache] = {}
        #: Expected final length per slot, for the reserved-vs-internal split.
        self._expected: dict[int, int] = {}

    # -- allocation ---------------------------------------------------------

    def allocate(self, expected_len: int | None = None) -> StaticKVCache | None:
        """Take a slot, or None if the pool is full.

        Returning None rather than raising: a full pool is an ordinary
        scheduling condition, not an error. Week 7's admission control is
        built on exactly this signal.
        """
        if not self._free:
            return None
        slot = self._free.pop()
        cache = StaticKVCache(self, slot)
        self._live[slot] = cache
        self._expected[slot] = min(expected_len or self.max_seq_len, self.max_seq_len)
        return cache

    def free(self, cache: StaticKVCache) -> None:
        if cache.slot in self._live:
            del self._live[cache.slot]
            self._expected.pop(cache.slot, None)
            self._free.append(cache.slot)

    def reset(self) -> None:
        self._live.clear()
        self._expected.clear()
        self._free = list(range(self.num_slots))

    # -- accounting ---------------------------------------------------------

    @property
    def bytes_per_token(self) -> int:
        """Across all layers, both K and V, at this pool's dtype."""
        per_layer = 2 * self.cfg.num_key_value_heads * self.cfg.head_dim
        return per_layer * self.cfg.num_hidden_layers * self.dtype.itemsize

    @property
    def total_bytes(self) -> int:
        return self.num_slots * self.max_seq_len * self.bytes_per_token

    def stats(self) -> PoolStats:
        used = sum(len(c) for c in self._live.values())
        reserved = sum(
            max(0, self._expected[slot] - len(c)) for slot, c in self._live.items()
        )
        return PoolStats(
            slots_total=self.num_slots,
            slots_allocated=len(self._live),
            tokens_per_slot=self.max_seq_len,
            tokens_used=used,
            tokens_reserved=reserved,
            bytes_per_token=self.bytes_per_token,
        )

    def __repr__(self) -> str:
        return (
            f"StaticCachePool({self.num_slots} slots x {self.max_seq_len} tokens, "
            f"{self.total_bytes / 1024**2:.0f} MB, {len(self._live)} live)"
        )


def slots_for_budget(
    cfg: NanoConfig,
    budget_bytes: int,
    max_seq_len: int | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> int:
    """How many contiguous slots fit in `budget_bytes`.

    The whole argument for paging in one function: this number is fixed by the
    *worst case* length, not by what sequences actually use. A pool sized for
    1,024-token sequences serves the same number of concurrent requests whether
    they average 1,000 tokens or 50.
    """
    max_seq_len = max_seq_len or cfg.max_position_embeddings
    per_layer = 2 * cfg.num_key_value_heads * cfg.head_dim
    bytes_per_token = per_layer * cfg.num_hidden_layers * dtype.itemsize
    return max(0, budget_bytes // (bytes_per_token * max_seq_len))
