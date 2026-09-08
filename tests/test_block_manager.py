"""The paged KV cache: correctness, allocation, and copy-on-write.

An allocator bug here does not crash. It returns the wrong KV for some token
in some sequence, and the model carries on producing fluent text conditioned on
the wrong context. So the top-level assertion is the strongest available: paged
output must be *byte-identical* to the contiguous cache it replaces, across
prefill, decode, chunked prefill, and block boundaries.

The block-boundary cases get their own sweep. With `block_size=16`, the
interesting positions are 15/16/17 -- one before, exactly on, and one after --
and those are where an off-by-one in `_ensure_capacity` or the slot index will
show up.

Copy-on-write gets the most attention because it is the only place two
sequences can corrupt each other. The invariant: a fork must be able to diverge
without the parent's tokens changing, and full blocks must stay shared forever
since their contents are immutable.
"""

from __future__ import annotations

import pytest
import torch

from engine.block_manager import (
    DEFAULT_BLOCK_SIZE,
    BlockAllocator,
    OutOfBlocks,
    PagedCachePool,
    blocks_for_budget,
)
from model.config import NANO_27M, NanoConfig
from model.transformer import DynamicCache, NanoForCausalLM

CFG = NanoConfig(
    vocab_size=256,
    hidden_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    intermediate_size=176,
    max_position_embeddings=512,
)


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return NanoForCausalLM(CFG).eval()


def make_pool(num_blocks=64, block_size=DEFAULT_BLOCK_SIZE):
    return PagedCachePool(
        CFG, num_blocks=num_blocks, block_size=block_size,
        device="cpu", dtype=torch.float32,
    )


def decode(model, cache, ids, n):
    """Prefill `ids`, then greedily decode `n` tokens."""
    with torch.inference_mode():
        logits, _ = model(ids, cache=cache)
        out = []
        for _ in range(n):
            nxt = logits[:, -1].argmax(-1, keepdim=True)
            out.append(int(nxt))
            logits, _ = model(nxt, cache=cache)
    return out


# ---------------------------------------------------------------------------
# The headline invariant
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("prompt_len", [1, 8, 15, 16, 17, 31, 32, 33, 64, 100])
def test_paged_output_is_identical_to_contiguous(model, prompt_len):
    """Byte-identical, across every block-boundary case.

    15/16/17 and 31/32/33 straddle block edges at block_size=16, which is
    where an off-by-one in capacity or slot indexing surfaces.
    """
    torch.manual_seed(prompt_len)
    ids = torch.randint(0, CFG.vocab_size, (1, prompt_len))

    want = decode(model, DynamicCache(), ids, 20)
    got = decode(model, make_pool().allocate(), ids, 20)
    assert got == want


def test_long_generation_crosses_many_blocks(model):
    """120 tokens is 8 blocks. Drift would compound across every boundary."""
    torch.manual_seed(1)
    ids = torch.randint(0, CFG.vocab_size, (1, 9))
    assert decode(model, make_pool().allocate(), ids, 120) == decode(
        model, DynamicCache(), ids, 120
    )


@pytest.mark.parametrize("block_size", [1, 2, 4, 16, 64])
def test_any_block_size_gives_the_same_answer(model, block_size):
    """Block size is a memory/bookkeeping trade-off, never a correctness one."""
    torch.manual_seed(2)
    ids = torch.randint(0, CFG.vocab_size, (1, 37))
    want = decode(model, DynamicCache(), ids, 25)
    got = decode(model, make_pool(num_blocks=256, block_size=block_size).allocate(), ids, 25)
    assert got == want


def test_chunked_prefill_matches_single_shot(model):
    """Prefill arriving in uneven pieces must equal one call."""
    torch.manual_seed(3)
    ids = torch.randint(0, CFG.vocab_size, (1, 50))
    with torch.inference_mode():
        whole, _ = model(ids, cache=make_pool().allocate())
        c = make_pool().allocate()
        pieces = [model(ids[:, s : s + 13], cache=c)[0] for s in range(0, 50, 13)]
    assert torch.allclose(torch.cat(pieces, dim=1), whole, atol=2e-5)


def test_cache_reports_length_for_rope(model):
    c = make_pool().allocate()
    assert len(c) == 0
    with torch.inference_mode():
        model(torch.randint(0, CFG.vocab_size, (1, 20)), cache=c)
        assert len(c) == 20
        model(torch.tensor([[7]]), cache=c)
        assert len(c) == 21


def test_blocks_are_taken_only_when_a_boundary_is_crossed(model):
    """The entire memory argument: capacity tracks tokens, not reservations."""
    pool = make_pool()
    c = pool.allocate()
    with torch.inference_mode():
        model(torch.randint(0, CFG.vocab_size, (1, 16)), cache=c)
        assert c.num_blocks == 1, "16 tokens is exactly one block"
        model(torch.tensor([[3]]), cache=c)
        assert c.num_blocks == 2, "the 17th token must open a second block"
        for _ in range(14):
            model(torch.tensor([[3]]), cache=c)
        assert len(c) == 31 and c.num_blocks == 2, "still inside block 2"
        model(torch.tensor([[3]]), cache=c)
        assert len(c) == 32 and c.num_blocks == 2, "32 tokens is exactly two blocks"
        model(torch.tensor([[3]]), cache=c)
        assert len(c) == 33 and c.num_blocks == 3, "the 33rd token opens block 3"


# ---------------------------------------------------------------------------
# The allocator
# ---------------------------------------------------------------------------


def test_allocate_and_free_round_trip():
    a = BlockAllocator(4)
    assert a.num_free == 4
    blocks = [a.allocate() for _ in range(4)]
    assert sorted(blocks) == [0, 1, 2, 3]
    assert a.allocate() is None
    for b in blocks:
        assert a.decref(b) is True
    assert a.num_free == 4


def test_allocate_many_is_all_or_nothing():
    """A partial allocation strands blocks and is how an allocator deadlocks."""
    a = BlockAllocator(4)
    assert a.allocate_many(5) is None
    assert a.num_free == 4, "a failed allocation must take nothing"
    got = a.allocate_many(4)
    assert got is not None and len(got) == 4


def test_refcounts_gate_freeing():
    a = BlockAllocator(2)
    b = a.allocate()
    a.incref(b)
    assert a.ref_count(b) == 2
    assert a.decref(b) is False, "still referenced"
    assert a.num_free == 1
    assert a.decref(b) is True
    assert a.num_free == 2


def test_refcount_errors_are_loud():
    """Silent underflow would hand a live block to a second sequence."""
    a = BlockAllocator(2)
    with pytest.raises(RuntimeError, match="decref on free block"):
        a.decref(0)
    with pytest.raises(RuntimeError, match="incref on free block"):
        a.incref(0)


def test_exhaustion_raises_with_a_useful_message(model):
    pool = make_pool(num_blocks=2)  # 32 tokens
    c = pool.allocate()
    with torch.inference_mode(), pytest.raises(OutOfBlocks, match="pool exhausted"):
        model(torch.randint(0, CFG.vocab_size, (1, 40)), cache=c)


def test_freeing_a_sequence_returns_every_block(model):
    pool = make_pool()
    c = pool.allocate()
    with torch.inference_mode():
        model(torch.randint(0, CFG.vocab_size, (1, 50)), cache=c)
    assert pool.allocator.num_allocated == 4
    c.free()
    assert pool.allocator.num_allocated == 0
    assert len(c) == 0


# ---------------------------------------------------------------------------
# Copy-on-write
# ---------------------------------------------------------------------------


def test_fork_shares_every_block_without_copying(model):
    pool = make_pool()
    parent = pool.allocate()
    with torch.inference_mode():
        model(torch.randint(0, CFG.vocab_size, (1, 48)), cache=parent)

    before = pool.allocator.num_allocated
    child = parent.fork()

    assert pool.allocator.num_allocated == before, "fork must not allocate"
    assert child.block_table == parent.block_table
    assert len(child) == len(parent)
    for b in parent.block_table:
        assert pool.allocator.ref_count(b) == 2


def test_forked_sequences_diverge_without_corrupting_each_other(model):
    """The invariant COW exists to guarantee.

    Two sequences share a 48-token prefix, then generate different
    continuations. Each must produce exactly what it would have produced
    alone.
    """
    torch.manual_seed(4)
    prompt = torch.randint(0, CFG.vocab_size, (1, 48))
    a_next = torch.tensor([[11]])
    b_next = torch.tensor([[99]])

    tail = torch.tensor([[5, 6, 7]])

    # Reference: each branch run independently with a contiguous cache.
    with torch.inference_mode():
        ca = DynamicCache()
        model(prompt, cache=ca)
        model(a_next, cache=ca)
        want_a, _ = model(tail, cache=ca)

        cb = DynamicCache()
        model(prompt, cache=cb)
        model(b_next, cache=cb)
        want_b, _ = model(tail, cache=cb)

    # Paged: prefill once, fork, diverge.
    pool = make_pool()
    parent = pool.allocate()
    with torch.inference_mode():
        model(prompt, cache=parent)
        child = parent.fork()
        model(a_next, cache=parent)
        model(b_next, cache=child)
        got_a, _ = model(tail, cache=parent)
        got_b, _ = model(tail, cache=child)

    assert torch.allclose(got_a, want_a, atol=2e-5), "parent diverged from its reference"
    assert torch.allclose(got_b, want_b, atol=2e-5), "child diverged from its reference"
    # Non-vacuity: the two branches must actually be distinguishable. Compared
    # on logits rather than sampled tokens -- this randomly-initialised model
    # collapses to one repeated argmax, so equal tokens would prove nothing.
    assert not torch.allclose(want_a, want_b, atol=1e-3), "branches are indistinguishable"
    assert not torch.allclose(got_a, got_b, atol=1e-3), "COW failed to separate them"


def test_cow_copies_only_the_tail_block(model):
    """Full blocks are immutable, so only the partial tail is ever copied.

    Copying the whole table on divergence would make forking cost the same as
    not forking, and prefix sharing would buy nothing.
    """
    pool = make_pool()
    parent = pool.allocate()
    with torch.inference_mode():
        model(torch.randint(0, CFG.vocab_size, (1, 40)), cache=parent)  # 3 blocks, tail partial
    child = parent.fork()
    before = pool.allocator.num_allocated

    with torch.inference_mode():
        model(torch.tensor([[7]]), cache=child)

    assert pool.allocator.num_allocated == before + 1, "exactly one block copied"
    # The two full blocks stay shared; only the tail diverged.
    assert child.block_table[:2] == parent.block_table[:2]
    assert child.block_table[2] != parent.block_table[2]
    for b in parent.block_table[:2]:
        assert pool.allocator.ref_count(b) == 2


def test_no_cow_needed_on_an_exact_block_boundary(model):
    """A sequence ending exactly on a boundary writes into a fresh block.

    Its next token opens a new block rather than touching a shared one, so
    there is nothing to copy.
    """
    pool = make_pool()
    parent = pool.allocate()
    with torch.inference_mode():
        model(torch.randint(0, CFG.vocab_size, (1, 32)), cache=parent)  # exactly 2 blocks
    child = parent.fork()
    before = pool.allocator.num_allocated

    with torch.inference_mode():
        model(torch.tensor([[7]]), cache=child)

    # One new block for the new token; no copy of an existing one.
    assert pool.allocator.num_allocated == before + 1
    assert child.block_table[:2] == parent.block_table[:2]


def test_freeing_a_fork_leaves_the_parent_intact(model):
    torch.manual_seed(5)
    prompt = torch.randint(0, CFG.vocab_size, (1, 40))
    pool = make_pool()
    parent = pool.allocate()
    with torch.inference_mode():
        model(prompt, cache=parent)

    probe = parent.fork()
    want = decode(model, probe, torch.tensor([[5]]), 10)
    probe.free()  # a fork holds blocks until freed, exactly like any sequence

    child = parent.fork()
    with torch.inference_mode():
        model(torch.tensor([[42]]), cache=child)
    child.free()

    assert pool.allocator.num_allocated == parent.num_blocks
    probe2 = parent.fork()
    assert decode(model, probe2, torch.tensor([[5]]), 10) == want
    probe2.free()


def test_many_forks_share_one_copy(model):
    """The prefix-caching win, in blocks: 8 sequences, one stored prefix."""
    pool = make_pool()
    parent = pool.allocate()
    with torch.inference_mode():
        model(torch.randint(0, CFG.vocab_size, (1, 64)), cache=parent)

    n_blocks = pool.allocator.num_allocated
    forks = [parent.fork() for _ in range(8)]

    assert pool.allocator.num_allocated == n_blocks, "9 sequences, one copy"
    assert pool.allocator.num_shared == n_blocks
    for f in forks:
        f.free()
    assert pool.allocator.num_allocated == n_blocks
    parent.free()
    assert pool.allocator.num_allocated == 0


# ---------------------------------------------------------------------------
# Accounting
# ---------------------------------------------------------------------------


def test_internal_fragmentation_is_bounded_by_block_size(model):
    """The claim step 2 could only derive, now measured."""
    pool = make_pool(num_blocks=256)
    caches = []
    total = 0
    for n in (5, 16, 17, 33, 100, 7):
        c = pool.allocate()
        with torch.inference_mode():
            model(torch.randint(0, CFG.vocab_size, (1, n)), cache=c)
        caches.append(c)
        total += n

    s = pool.stats(tokens_used=total)
    assert s.external_fragmentation == 0
    assert s.internal_fragmentation < len(caches) * pool.block_size
    assert s.internal_fragmentation == s.tokens_allocated - total


def test_waste_is_far_below_the_contiguous_design():
    """2% versus the 72% measured for contiguous slots in step 2."""
    from engine.kv_cache import slots_for_budget

    budget = 512 * 1024 * 1024
    blocks = blocks_for_budget(NANO_27M, budget)
    slots = slots_for_budget(NANO_27M, budget, max_seq_len=1024)

    # Same memory, 64x more allocation units.
    assert blocks == slots * (1024 // DEFAULT_BLOCK_SIZE)
    # A 339-token sequence (this workload's mean) wastes at most 15 tokens.
    waste = (-(-339 // DEFAULT_BLOCK_SIZE) * DEFAULT_BLOCK_SIZE) - 339
    assert 0 <= waste < DEFAULT_BLOCK_SIZE
    assert waste / 339 < 0.05


def test_stats_serialise():
    pool = make_pool(num_blocks=32)
    c = pool.allocate()
    c._ensure_capacity(20)
    d = pool.stats(tokens_used=20).to_dict()
    assert d["blocks_allocated"] == 2
    assert d["tokens_allocated"] == 32
    assert d["tokens_internal_frag"] == 12
    assert d["tokens_external_frag"] == 0
