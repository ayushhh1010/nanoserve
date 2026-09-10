"""Prefix caching: correctness of reuse, key safety, and eviction.

The dangerous failure here is not a crash. It is serving one request's KV to
another whose context merely *looks* similar, producing fluent text conditioned
on the wrong prefix. So the central assertion is that a cache hit yields
byte-identical output to a cold prefill, and the key tests prove that two blocks
with the same tokens but different history never collide.

Hash chaining is what makes that safe: a block's key includes every block before
it, so `[A][B]` and `[C][B]` give different keys for the second block even
though its tokens are the same. Their KV genuinely differs -- attention inside a
block depends on the context preceding it -- and an unchained hash would happily
swap them.
"""

from __future__ import annotations

import pytest
import torch

from engine.block_manager import PagedCachePool
from engine.prefix_cache import ROOT_HASH, PrefixCache, block_hash
from engine.scheduler import ContinuousBatchingEngine
from engine.types import Request, SamplingParams
from model.config import NanoConfig
from model.transformer import NanoForCausalLM

CFG = NanoConfig(
    vocab_size=256, hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
    num_key_value_heads=2, intermediate_size=176, max_position_embeddings=512,
)
EOS = 255
BS = 16


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return NanoForCausalLM(CFG).eval()


def pool(num_blocks=256):
    return PagedCachePool(CFG, num_blocks=num_blocks, block_size=BS,
                          device="cpu", dtype=torch.float32)


def engine(model, p=None, **kw):
    kw.setdefault("reserve_blocks", 4)
    return ContinuousBatchingEngine(
        model, p or pool(), eos_token_id=EOS, device="cpu", **kw
    )


def req(prompt, max_tokens=8, rid=None):
    r = Request(prompt_token_ids=list(prompt),
                params=SamplingParams(max_tokens=max_tokens, ignore_eos=True))
    if rid:
        r.request_id = rid
    return r


# ---------------------------------------------------------------------------
# Key safety
# ---------------------------------------------------------------------------


def test_identical_tokens_with_different_history_get_different_keys():
    """The collision an unchained hash would cause.

    Block [B] following [A] and block [B] following [C] hold the same tokens
    and completely different KV. Chaining the parent hash into the key is what
    keeps them apart.
    """
    a, b, c = (1,) * BS, (2,) * BS, (3,) * BS

    after_a = block_hash(block_hash(ROOT_HASH, a), b)
    after_c = block_hash(block_hash(ROOT_HASH, c), b)

    assert after_a != after_c
    # ...and the same history does give the same key, or nothing would ever hit.
    assert after_a == block_hash(block_hash(ROOT_HASH, a), b)


def test_hash_chain_covers_only_full_blocks():
    """A partial block has no stable identity yet, so it gets no key."""
    p = pool()
    cache = PrefixCache(p)

    assert cache.hash_chain(list(range(BS - 1))) == []
    assert len(cache.hash_chain(list(range(BS)))) == 1
    assert len(cache.hash_chain(list(range(BS + 5)))) == 1
    assert len(cache.hash_chain(list(range(2 * BS)))) == 2


def test_chain_is_a_prefix_relation():
    """A longer sequence's chain must extend a shorter one's, not replace it."""
    cache = PrefixCache(pool())
    short = cache.hash_chain(list(range(2 * BS)))
    long = cache.hash_chain(list(range(5 * BS)))
    assert long[: len(short)] == short


# ---------------------------------------------------------------------------
# The headline invariant
# ---------------------------------------------------------------------------


def test_cache_hit_produces_identical_output(model):
    """A reused prefix must give exactly what a cold prefill would.

    Two requests share a 64-token prefix and differ afterwards. The second is
    served partly from cache; its tokens must match a run with caching off.
    """
    shared = [7, 3, 11, 5] * 16  # 64 tokens, 4 full blocks
    reqs = lambda: [  # noqa: E731
        req(shared + [21, 22, 23], 12, rid="a"),
        req(shared + [91, 92, 93], 12, rid="b"),
    ]

    cold = {r.request_id: r.output_token_ids
            for r in engine(model, enable_prefix_cache=False).run(reqs())}

    e = engine(model, enable_prefix_cache=True)
    warm = {r.request_id: r.output_token_ids for r in e.run(reqs())}

    assert warm == cold
    assert e.prefix_cache.stats.hit_blocks > 0, "no reuse happened; test proves nothing"


def test_repeated_identical_prompts_hit_almost_everything(model):
    """The system-prompt case: same prefix, many requests."""
    prefix = list(range(1, 97))  # 96 tokens = 6 blocks
    e = engine(model)

    e.run([req(prefix + [200 + i], 6, rid=f"r{i}") for i in range(8)])

    s = e.prefix_cache.stats
    # First request populates; the other seven should reuse nearly all of it.
    assert s.block_hit_rate > 0.7, f"hit rate only {s.block_hit_rate:.1%}"
    assert s.prefill_tokens_saved if hasattr(s, "prefill_tokens_saved") else True


def test_disjoint_prompts_never_hit(model):
    """No false sharing between unrelated requests."""
    e = engine(model)
    e.run([req([i] * 64, 4, rid=f"r{i}") for i in range(1, 6)])
    assert e.prefix_cache.stats.hit_blocks == 0


def test_partial_overlap_reuses_only_the_common_blocks(model):
    """Matching stops at the first divergent block, not before or after."""
    common = list(range(1, 33))  # 2 full blocks
    e = engine(model)

    e.run([req(common + list(range(100, 133)), 4, rid="first")])
    before = e.prefix_cache.stats.hit_blocks
    e.run([req(common + list(range(200, 233)), 4, rid="second")])
    gained = e.prefix_cache.stats.hit_blocks - before

    assert gained == 2, f"expected the 2 shared blocks, reused {gained}"


def test_caching_off_is_a_true_bypass(model):
    p = pool()
    e = engine(model, p=p, enable_prefix_cache=False)
    e.run([req(list(range(1, 65)) + [9], 5, rid=f"r{i}") for i in range(4)])

    assert e.prefix_cache.stats.hit_blocks == 0
    assert len(e.prefix_cache._entries) == 0
    assert p.allocator.num_allocated == 0, "disabled cache must hold nothing"


# ---------------------------------------------------------------------------
# References and eviction
# ---------------------------------------------------------------------------


def test_cached_blocks_outlive_the_request_that_made_them(model):
    p = pool()
    e = engine(model, p=p)
    e.run([req(list(range(1, 65)) + [9], 4, rid="only")])

    assert len(e.prefix_cache._entries) == 4
    assert p.allocator.num_allocated == 4, "cache should be holding the 4 full blocks"
    e.prefix_cache.clear()
    assert p.allocator.num_allocated == 0


def test_eviction_takes_the_coldest_first(model):
    p = pool()
    e = engine(model, p=p)

    # Three distinct prefixes, cached oldest to newest.
    for i in range(3):
        e.run([req([i + 1] * 32 + [9], 3, rid=f"p{i}")])
    assert len(e.prefix_cache._entries) == 6  # 2 blocks each

    # Touch the first prefix so it is no longer the coldest.
    e.run([req([1] * 32 + [8], 3, rid="touch")])

    e.prefix_cache.evict(2)
    remaining = {h for h in e.prefix_cache._entries}
    warm = set(e.prefix_cache.hash_chain([1] * 32))
    assert warm <= remaining, "evicted the most recently used prefix"


def test_blocks_in_use_are_never_evicted(model):
    """Eviction must not pull KV out from under a live generation."""
    p = pool()
    e = engine(model, p=p)
    e.run([req(list(range(1, 65)) + [9], 4, rid="seed")])

    # Start a request that reuses the cached prefix and leave it running.
    e.add_request(req(list(range(1, 65)) + [9], 40, rid="live"))
    e.step()
    assert e.running

    held = list(e.running[0].cache.block_table)
    freed = e.prefix_cache.evict(10)
    for block in held:
        assert p.allocator.ref_count(block) > 0, "evicted a block a sequence is reading"
    assert freed == 0, "every cached block is pinned by the running sequence"


def test_evicting_an_empty_cache_is_safe():
    p = pool()
    c = PrefixCache(p)
    assert c.evict(5) == 0
    assert c.ensure_free(0) is True


def test_cache_gives_blocks_back_under_memory_pressure(model):
    """A cached prefix is an optimisation; a request that cannot start is not.

    With a pool barely large enough, admission must be able to reclaim cache
    blocks rather than stalling.
    """
    p = pool(num_blocks=24)
    e = engine(model, p=p, reserve_blocks=2, max_batch_size=8)

    done = e.run([req(list(range(i, i + 96)) + [9], 10, rid=f"r{i}") for i in range(0, 60, 10)])

    assert len(done) == 6
    assert all(r.output_len == 10 for r in done)
    assert e.prefix_cache.stats.evictions > 0, "expected pressure to force eviction"


# ---------------------------------------------------------------------------
# Accounting
# ---------------------------------------------------------------------------


def test_hit_rate_is_measured_in_blocks_not_requests(model):
    """A request reusing 40 of 50 blocks is mostly a hit, not a miss."""
    e = engine(model)
    prefix = list(range(1, 161))  # 10 blocks
    e.run([req(prefix + [1], 3, rid="a")])
    e.run([req(prefix + [2], 3, rid="b")])

    s = e.prefix_cache.stats
    assert s.queries == 2
    assert s.hit_blocks == 10  # second request reused all ten
    assert s.query_blocks == 20
    assert s.block_hit_rate == pytest.approx(0.5)


def test_stats_serialise(model):
    e = engine(model)
    e.run([req(list(range(1, 65)) + [9], 4, rid="a")])
    d = e.prefix_cache.stats.to_dict(BS)
    assert set(d) >= {"queries", "hit_blocks", "block_hit_rate", "prefill_tokens_saved"}
    assert d["prefill_tokens_saved"] == d["hit_blocks"] * BS


def test_engine_reports_cache_stats(model):
    e = engine(model)
    prefix = list(range(1, 65))
    e.run([req(prefix + [1], 3), req(prefix + [2], 3)])
    st = e.stats()
    assert st.prefix_cache_queries > 0
    assert st.prefix_hit_rate > 0
