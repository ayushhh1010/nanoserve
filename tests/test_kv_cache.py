"""The static KV cache, and the waste accounting that motivates paging.

Correctness first: a statically allocated cache must produce byte-identical
output to the growing one. It is an allocator change, not a modelling change,
and the same assertion will be made of the paged version.

Then the accounting. The claim "contiguous pre-allocation wastes 60-80% of KV
memory" is the reason step 3 exists, so it has to be a measurement rather than
a citation -- and the three kinds of waste have to be separated, because they
have different fixes.
"""

from __future__ import annotations

import pytest
import torch

from engine.kv_cache import PoolStats, StaticCachePool, slots_for_budget
from model.config import NANO_27M, NanoConfig
from model.transformer import DynamicCache, NanoForCausalLM

CFG = NanoConfig(
    vocab_size=256,
    hidden_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    intermediate_size=176,
    max_position_embeddings=128,
)


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return NanoForCausalLM(CFG).eval()


def pool(**kw):
    kw.setdefault("num_slots", 4)
    kw.setdefault("device", "cpu")
    kw.setdefault("dtype", torch.float32)
    return StaticCachePool(CFG, **kw)


# ---------------------------------------------------------------------------
# Correctness -- the allocator must not change the answer
# ---------------------------------------------------------------------------


def test_static_cache_matches_dynamic_cache_exactly(model):
    """Same invariant as the no-cache/cache pair. An allocator is not a model."""
    ids = torch.randint(0, CFG.vocab_size, (1, 24))

    with torch.inference_mode():
        dyn = DynamicCache()
        logits_d, _ = model(ids, cache=dyn)
        out_d = []
        for _ in range(20):
            nxt = logits_d[:, -1].argmax(-1, keepdim=True)
            out_d.append(int(nxt))
            logits_d, _ = model(nxt, cache=dyn)

        stat = pool().allocate()
        logits_s, _ = model(ids, cache=stat)
        out_s = []
        for _ in range(20):
            nxt = logits_s[:, -1].argmax(-1, keepdim=True)
            out_s.append(int(nxt))
            logits_s, _ = model(nxt, cache=stat)

    assert out_s == out_d


def test_static_cache_reports_its_length(model):
    """The model reads `len(cache)` to place RoPE positions.

    A cache that reports 0 makes every decode step think it is at position 0.
    The output stays fluent and is quietly wrong, which is why this is asserted
    rather than assumed.
    """
    c = pool().allocate()
    assert len(c) == 0
    with torch.inference_mode():
        model(torch.randint(0, CFG.vocab_size, (1, 11)), cache=c)
        assert len(c) == 11
        model(torch.tensor([[5]]), cache=c)
        assert len(c) == 12


def test_position_advances_once_per_forward_not_once_per_layer(model):
    """All layers of one step write at the same offset.

    Advancing per layer would scatter one step's KV across `n_layers`
    different positions. The model still runs; the output is garbage.
    """
    c = pool().allocate()
    with torch.inference_mode():
        model(torch.randint(0, CFG.vocab_size, (1, 7)), cache=c)
    assert len(c) == 7, f"expected 7 tokens cached, got {len(c)} (layers={CFG.num_hidden_layers})"


def test_chunked_prefill_through_a_static_slot(model):
    """Prefill split across calls must equal prefill in one call."""
    ids = torch.randint(0, CFG.vocab_size, (1, 30))
    with torch.inference_mode():
        whole, _ = model(ids, cache=pool().allocate())
        c = pool().allocate()
        pieces = [model(ids[:, s : s + 11], cache=c)[0] for s in range(0, 30, 11)]
    assert torch.allclose(torch.cat(pieces, dim=1), whole, atol=2e-5)


def test_overrunning_a_slot_raises(model):
    p = pool(max_seq_len=16)
    c = p.allocate()
    with torch.inference_mode(), pytest.raises(RuntimeError, match="exceeded slot capacity"):
        model(torch.randint(0, CFG.vocab_size, (1, 20)), cache=c)


# ---------------------------------------------------------------------------
# Allocation
# ---------------------------------------------------------------------------


def test_pool_hands_out_distinct_slots_and_reclaims_them():
    p = pool(num_slots=3)
    a, b, c = p.allocate(), p.allocate(), p.allocate()
    assert len({a.slot, b.slot, c.slot}) == 3
    assert p.allocate() is None, "a full pool must report full, not overcommit"

    p.free(b)
    d = p.allocate()
    assert d is not None and d.slot == b.slot


def test_full_pool_returns_none_rather_than_raising():
    """A full pool is a scheduling condition, not an error.

    Week 7's admission control is built on this signal, so it has to be a
    value the scheduler can branch on rather than an exception it must catch.
    """
    p = pool(num_slots=1)
    assert p.allocate() is not None
    assert p.allocate() is None


def test_freeing_twice_is_harmless():
    p = pool(num_slots=2)
    c = p.allocate()
    p.free(c)
    p.free(c)
    assert len(p._free) == 2


def test_sequences_do_not_see_each_others_memory(model):
    """Slot isolation. A leak here would be invisible in the output shape."""
    p = pool(num_slots=2)
    a, b = p.allocate(), p.allocate()
    with torch.inference_mode():
        model(torch.full((1, 9), 3), cache=a)
        model(torch.full((1, 5), 200), cache=b)

    assert len(a) == 9 and len(b) == 5
    # b's slot must be untouched beyond its own 5 tokens.
    assert torch.count_nonzero(p.k[0][b.slot, :, 5:]) == 0


# ---------------------------------------------------------------------------
# The waste accounting -- why step 3 exists
# ---------------------------------------------------------------------------


def test_utilization_is_used_over_allocated():
    s = PoolStats(
        slots_total=10, slots_allocated=4, tokens_per_slot=1024,
        tokens_used=300, tokens_reserved=0, bytes_per_token=4096,
    )
    assert s.tokens_allocated == 4096
    assert s.utilization == pytest.approx(300 / 4096)
    assert s.waste_fraction == pytest.approx(1 - 300 / 4096)
    assert s.occupancy == pytest.approx(300 / 10240)


def test_waste_splits_into_reserved_internal_and_external():
    """The three kinds have different fixes, so they are reported separately."""
    s = PoolStats(
        slots_total=10, slots_allocated=4, tokens_per_slot=1024,
        tokens_used=400, tokens_reserved=200, bytes_per_token=4096,
    )
    # reserved: will legitimately be filled. internal: never will be.
    assert s.internal_fragmentation == 4096 - 400 - 200
    # every unallocated slot is a full run a small request cannot subdivide
    assert s.external_fragmentation == 6 * 1024
    assert s.tokens_used + s.tokens_reserved + s.internal_fragmentation == s.tokens_allocated


def test_realistic_workload_wastes_the_published_fraction():
    """The 60-80% figure, reproduced on this project's own numbers.

    Sequences here run to a few hundred tokens against a 1,024-token
    reservation, which is exactly the regime the PagedAttention paper
    measured.
    """
    p = pool(num_slots=8, max_seq_len=1024)
    for total in (180, 320, 260, 410):  # typical prompt+output for this workload
        c = p.allocate(expected_len=total)
        c._pos = total

    s = p.stats()
    assert 0.60 < s.waste_fraction < 0.85, (
        f"waste {s.waste_fraction:.1%} outside the 60-85% band the design implies"
    )
    assert s.utilization < 0.35


def test_concurrency_is_capped_by_the_worst_case_not_the_average():
    """The core argument for paging, stated as a number.

    A pool sized for 1,024-token sequences serves the same number of
    concurrent requests whether they average 1,000 tokens or 50. Paging makes
    capacity depend on tokens actually used.
    """
    budget = 512 * 1024 * 1024  # 512 MB
    long_ctx = slots_for_budget(NANO_27M, budget, max_seq_len=1024)
    short_ctx = slots_for_budget(NANO_27M, budget, max_seq_len=128)

    assert long_ctx == 128, f"expected 128 slots at 4 KB/token x 1024, got {long_ctx}"
    assert short_ctx == 8 * long_ctx
    # 4 KB/token is the GQA figure from Phase 1; if it moves, this whole
    # phase's memory arithmetic moves with it.
    assert NANO_27M.kv_bytes_per_token == 4096


def test_stats_serialise_with_megabytes():
    p = pool(num_slots=4, max_seq_len=256)
    c = p.allocate(expected_len=100)
    c._pos = 40
    d = p.stats().to_dict()

    assert d["slots_allocated"] == 1
    assert d["tokens_used"] == 40
    assert d["tokens_reserved"] == 60
    assert d["allocated_mb"] > 0
    assert 0 <= d["utilization"] <= 1


def test_empty_pool_reports_zero_not_nan():
    s = pool().stats()
    assert s.utilization == 0.0
    assert s.occupancy == 0.0
    assert s.tokens_used == 0
