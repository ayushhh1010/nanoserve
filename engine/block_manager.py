"""Paged KV cache: fixed-size blocks, a block table, and copy-on-write sharing.

This is textbook OS virtual memory applied to attention, and the mapping is
exact:

    block table  ->  page table
    block        ->  page
    free list    ->  frame allocator
    copy-on-write ->  copy-on-write

A sequence's KV no longer has to live in one unbroken run. It gets a *block
table* -- a list of physical block indices -- and its tokens are scattered
across the pool. Two consequences, and the second is the one that pays:

**Internal fragmentation is bounded.** A sequence wastes at most
`block_size - 1` tokens, in its final partial block, instead of
`max_seq_len - n`. Measured on this project's workload, that is 72.4% waste
down to about 2%.

**External fragmentation is impossible.** Every block is the same size, so any
free block satisfies any request. There is no "enough free memory, wrong
shape" state.

**On attention without a custom kernel.** vLLM ships hand-written CUDA that
reads the block table inside the attention kernel. This project deliberately
dropped GPU-kernel work, so instead the blocks are gathered into a contiguous
view before `scaled_dot_product_attention` sees them. That gather is one
`index_select` per layer per step -- O(tokens), which is exactly what the
`torch.cat` in `DynamicCache` already cost. So paging is not slower than the
cache it replaces; it is dramatically smaller. The throughput win comes later,
from fitting more sequences at once, not from a faster single stream.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from model.config import NanoConfig

#: 16 tokens. Small enough that the final partial block wastes little, large
#: enough that block-granular bookkeeping stays cheap. The value every
#: production implementation uses.
DEFAULT_BLOCK_SIZE = 16


class OutOfBlocks(RuntimeError):
    """The pool is exhausted. A scheduling condition, raised only where the
    caller has no `None` channel to report it on."""


@dataclass
class BlockStats:
    total: int
    allocated: int
    block_size: int
    tokens_used: int
    #: Blocks referenced by more than one sequence, i.e. shared prefixes.
    shared: int
    bytes_per_token: int

    @property
    def free(self) -> int:
        return self.total - self.allocated

    @property
    def tokens_allocated(self) -> int:
        return self.allocated * self.block_size

    @property
    def internal_fragmentation(self) -> int:
        """Allocated but unused. Bounded by block_size - 1 per sequence."""
        return max(0, self.tokens_allocated - self.tokens_used)

    @property
    def external_fragmentation(self) -> int:
        """Always zero. Uniform blocks are interchangeable by construction."""
        return 0

    @property
    def utilization(self) -> float:
        return self.tokens_used / self.tokens_allocated if self.tokens_allocated else 0.0

    @property
    def waste_fraction(self) -> float:
        return 1.0 - self.utilization

    @property
    def occupancy(self) -> float:
        return self.allocated / self.total if self.total else 0.0

    def to_dict(self) -> dict:
        mb = 1024 * 1024
        return {
            "blocks_total": self.total,
            "blocks_allocated": self.allocated,
            "blocks_free": self.free,
            "blocks_shared": self.shared,
            "block_size": self.block_size,
            "tokens_used": self.tokens_used,
            "tokens_allocated": self.tokens_allocated,
            "tokens_internal_frag": self.internal_fragmentation,
            "tokens_external_frag": 0,
            "utilization": round(self.utilization, 4),
            "occupancy": round(self.occupancy, 4),
            "waste_fraction": round(self.waste_fraction, 4),
            "used_mb": round(self.tokens_used * self.bytes_per_token / mb, 2),
            "allocated_mb": round(self.tokens_allocated * self.bytes_per_token / mb, 2),
        }


class BlockAllocator:
    """A free list over uniform blocks, with reference counts.

    Reference counting is what makes prefix sharing possible: two sequences
    with a common prefix point at the *same* physical blocks, and the pool
    stores one copy. The count says how many sequences would notice if the
    block changed, which is exactly the copy-on-write trigger.
    """

    def __init__(self, num_blocks: int) -> None:
        self.num_blocks = num_blocks
        # Reverse order so the first allocation is block 0, which makes traces
        # and test failures far easier to read.
        self._free: list[int] = list(reversed(range(num_blocks)))
        self._refs: list[int] = [0] * num_blocks

    def allocate(self) -> int | None:
        """One block with refcount 1, or None when exhausted."""
        if not self._free:
            return None
        block = self._free.pop()
        self._refs[block] = 1
        return block

    def allocate_many(self, n: int) -> list[int] | None:
        """`n` blocks, all-or-nothing.

        Partial allocation would leave a sequence holding blocks it cannot
        use while starving another, which is how an allocator deadlocks.
        """
        if n > len(self._free):
            return None
        return [self.allocate() for _ in range(n)]  # type: ignore[misc]

    def incref(self, block: int) -> None:
        if self._refs[block] <= 0:
            raise RuntimeError(f"incref on free block {block}")
        self._refs[block] += 1

    def decref(self, block: int) -> bool:
        """Drop one reference. Returns True if the block became free."""
        if self._refs[block] <= 0:
            raise RuntimeError(f"decref on free block {block}")
        self._refs[block] -= 1
        if self._refs[block] == 0:
            self._free.append(block)
            return True
        return False

    def ref_count(self, block: int) -> int:
        return self._refs[block]

    @property
    def num_free(self) -> int:
        return len(self._free)

    @property
    def num_allocated(self) -> int:
        return self.num_blocks - len(self._free)

    @property
    def num_shared(self) -> int:
        return sum(1 for r in self._refs if r > 1)


class PagedKVCache:
    """One sequence's block table. Implements the model's `KVCache` protocol.

    Attention cannot tell this apart from a contiguous cache: it calls
    `update` and receives a normal `(1, H, T, D)` tensor.
    """

    def __init__(self, pool: PagedCachePool) -> None:
        self.pool = pool
        #: logical block index -> physical block id
        self.block_table: list[int] = []
        self.num_tokens = 0
        self._start = 0  # where the current forward pass began writing
        # Physical slot index per token position, kept on device so the gather
        # never needs a host round-trip. Extended a block at a time.
        self._slots = torch.empty(0, dtype=torch.long, device=pool.device)

    def __len__(self) -> int:
        return self.num_tokens

    @property
    def num_blocks(self) -> int:
        return len(self.block_table)

    # -- growth -------------------------------------------------------------

    def _append_block(self, block: int) -> None:
        """Record a new physical block and its token slots."""
        bs = self.pool.block_size
        base = block * bs
        slots = torch.arange(base, base + bs, dtype=torch.long, device=self.pool.device)
        self._slots = torch.cat([self._slots, slots])
        self.block_table.append(block)

    def _ensure_capacity(self, n_new: int) -> None:
        """Allocate blocks so `num_tokens + n_new` fits.

        Blocks are taken only when a boundary is actually crossed, which is
        the whole point: a 20-token sequence holds 2 blocks, not a
        1,024-token reservation.
        """
        bs = self.pool.block_size
        needed = -(-(self.num_tokens + n_new) // bs)  # ceil division
        for _ in range(needed - len(self.block_table)):
            block = self.pool.allocator.allocate()
            if block is None:
                raise OutOfBlocks(
                    f"pool exhausted: {self.pool.allocator.num_free} free, "
                    f"sequence holds {len(self.block_table)}"
                )
            self._append_block(block)

    def _unshare_tail(self) -> None:
        """Copy-on-write the block about to be written into.

        Only the final, partially-filled block can ever need this. Every full
        block is immutable -- its tokens are fixed once written -- so sequences
        sharing a prefix keep sharing it forever. Copying only the tail is what
        makes prefix sharing nearly free.
        """
        if not self.block_table:
            return
        tail = self.block_table[-1]
        if self.pool.allocator.ref_count(tail) <= 1:
            return

        fresh = self.pool.allocator.allocate()
        if fresh is None:
            raise OutOfBlocks("no block available for copy-on-write")

        self.pool.copy_block(tail, fresh)
        self.pool.allocator.decref(tail)
        self.block_table[-1] = fresh

        bs = self.pool.block_size
        base = fresh * bs
        self._slots[-bs:] = torch.arange(
            base, base + bs, dtype=torch.long, device=self.pool.device
        )

    # -- the protocol -------------------------------------------------------

    def update(self, layer_idx: int, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        n = k.shape[-2]

        # Position advances once per forward pass, on layer 0 -- not once per
        # layer. Blocks are allocated on layer 0 too, so every layer of a step
        # writes to the same physical slots.
        if layer_idx == 0:
            self._start = self.num_tokens
            # Only a partially filled tail block can need copy-on-write. When
            # the sequence ends exactly on a boundary the next token opens a
            # fresh block, so there is nothing shared to copy.
            if self.num_tokens % self.pool.block_size != 0:
                self._unshare_tail()
            self._ensure_capacity(n)
            self.num_tokens += n

        start, end = self._start, self._start + n
        slots = self._slots[start:end]

        # One scatter per layer, not one per block. Writing block-by-block
        # would launch ceil(n/16) kernels per layer, which on a launch-bound
        # machine costs more than the copy itself.
        self.pool.k[layer_idx][slots] = k[0].transpose(0, 1)
        self.pool.v[layer_idx][slots] = v[0].transpose(0, 1)

        # Gather the whole sequence into a contiguous view for attention. One
        # index_select per layer, O(tokens) -- the same order of work the
        # torch.cat in DynamicCache already did.
        live = self._slots[:end]
        kk = self.pool.k[layer_idx][live].transpose(0, 1).unsqueeze(0)
        vv = self.pool.v[layer_idx][live].transpose(0, 1).unsqueeze(0)
        return kk, vv

    # -- sharing ------------------------------------------------------------

    def fork(self) -> PagedKVCache:
        """A second sequence sharing every block of this one.

        No KV is copied. Both sequences read the same physical blocks; the
        first write past the shared region triggers copy-on-write of the tail
        block only.
        """
        child = PagedKVCache(self.pool)
        for block in self.block_table:
            self.pool.allocator.incref(block)
        child.block_table = list(self.block_table)
        child.num_tokens = self.num_tokens
        child._slots = self._slots.clone()
        return child

    def free(self) -> None:
        """Release every block. Shared blocks survive until the last holder."""
        for block in self.block_table:
            self.pool.allocator.decref(block)
        self.block_table.clear()
        self.num_tokens = 0
        self._slots = torch.empty(0, dtype=torch.long, device=self.pool.device)

    def __repr__(self) -> str:
        return (
            f"PagedKVCache({self.num_tokens} tokens, {len(self.block_table)} blocks, "
            f"table={self.block_table[:6]}{'...' if len(self.block_table) > 6 else ''})"
        )


class PagedCachePool:
    """Physical block storage plus the allocator over it."""

    def __init__(
        self,
        cfg: NanoConfig,
        num_blocks: int,
        block_size: int = DEFAULT_BLOCK_SIZE,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.cfg = cfg
        self.block_size = block_size
        self.device = torch.device(device)
        self.dtype = dtype
        self.allocator = BlockAllocator(num_blocks)

        # Stored flat as (num_blocks * block_size, heads, head_dim) rather than
        # (num_blocks, block_size, heads, head_dim). The flat form makes both
        # the scatter and the gather a single index operation on a 1-D index
        # tensor; the nested form would need an unravel on every access.
        n_slots = num_blocks * block_size
        shape = (n_slots, cfg.num_key_value_heads, cfg.head_dim)
        self.k = [
            torch.zeros(shape, device=self.device, dtype=dtype)
            for _ in range(cfg.num_hidden_layers)
        ]
        self.v = [
            torch.zeros(shape, device=self.device, dtype=dtype)
            for _ in range(cfg.num_hidden_layers)
        ]

    # -- operations ---------------------------------------------------------

    def allocate(self, expected_len: int | None = None, **_) -> PagedKVCache | None:
        """A new sequence, or None when the pool cannot even start one.

        Returns None rather than a doomed cache, matching StaticCachePool's
        contract so the scheduler has one way to ask "is there room?". Only
        one block is required up front -- paging exists so capacity is not
        reserved for the worst case -- but a pool with zero free blocks cannot
        make progress and should say so here instead of raising OutOfBlocks
        several layers into a forward pass.
        """
        if self.allocator.num_free == 0:
            return None
        return PagedKVCache(self)

    def can_fit(self, num_tokens: int) -> bool:
        """Whether `num_tokens` could be admitted right now.

        Used by admission control in week 7, which needs to decide before
        committing rather than discover mid-generation.
        """
        needed = -(-num_tokens // self.block_size)
        return self.allocator.num_free >= needed

    def copy_block(self, src: int, dst: int) -> None:
        """Duplicate a block's contents across every layer. The COW copy."""
        bs = self.block_size
        s, d = src * bs, dst * bs
        for layer in range(self.cfg.num_hidden_layers):
            self.k[layer][d : d + bs] = self.k[layer][s : s + bs]
            self.v[layer][d : d + bs] = self.v[layer][s : s + bs]

    def reset(self) -> None:
        self.allocator = BlockAllocator(self.allocator.num_blocks)

    # -- accounting ---------------------------------------------------------

    @property
    def bytes_per_token(self) -> int:
        per_layer = 2 * self.cfg.num_key_value_heads * self.cfg.head_dim
        return per_layer * self.cfg.num_hidden_layers * self.dtype.itemsize

    @property
    def total_bytes(self) -> int:
        return self.allocator.num_blocks * self.block_size * self.bytes_per_token

    def stats(self, tokens_used: int = 0) -> BlockStats:
        return BlockStats(
            total=self.allocator.num_blocks,
            allocated=self.allocator.num_allocated,
            block_size=self.block_size,
            tokens_used=tokens_used,
            shared=self.allocator.num_shared,
            bytes_per_token=self.bytes_per_token,
        )

    def __repr__(self) -> str:
        a = self.allocator
        return (
            f"PagedCachePool({a.num_blocks} blocks x {self.block_size} tokens, "
            f"{self.total_bytes / 1024**2:.0f} MB, {a.num_allocated} allocated)"
        )


def blocks_for_budget(
    cfg: NanoConfig,
    budget_bytes: int,
    block_size: int = DEFAULT_BLOCK_SIZE,
    dtype: torch.dtype = torch.bfloat16,
) -> int:
    """How many blocks fit in `budget_bytes`.

    Contrast `slots_for_budget` in kv_cache.py, which divides by the *worst
    case* sequence length. This divides by the block size, so capacity is
    measured in tokens actually stored rather than tokens someone might
    eventually store.
    """
    per_layer = 2 * cfg.num_key_value_heads * cfg.head_dim
    bytes_per_token = per_layer * cfg.num_hidden_layers * dtype.itemsize
    return max(0, budget_bytes // (bytes_per_token * block_size))
