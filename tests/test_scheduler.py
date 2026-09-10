"""Continuous batching: correctness, scheduling behaviour, resource safety.

Two claims are being pinned.

**Batching must not change the answer.** Every sequence has to produce exactly
the tokens the sequential baseline produced, even though it now runs inside a
padded batch alongside sequences of different lengths. The failure mode is
specific and silent: without a padding mask, a short sequence attends to the
tail of a longer one and produces fluent text conditioned on somebody else's
context.

**Head-of-line blocking must actually be gone.** That is the entire point of
iteration-level scheduling, so it is asserted directly -- a short request
queued behind a long one finishes early rather than waiting for it.
"""

from __future__ import annotations

import random

import pytest
import torch

from engine.baseline import BaselineEngine
from engine.block_manager import PagedCachePool
from engine.scheduler import BatchedPagedCache, ContinuousBatchingEngine, Sequence
from engine.types import FinishReason, Request, SamplingParams
from model.config import NanoConfig
from model.transformer import NanoForCausalLM

CFG = NanoConfig(
    vocab_size=256, hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
    num_key_value_heads=2, intermediate_size=176, max_position_embeddings=512,
)
EOS = 255


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return NanoForCausalLM(CFG).eval()


def pool(num_blocks=512, block_size=16):
    return PagedCachePool(CFG, num_blocks=num_blocks, block_size=block_size,
                          device="cpu", dtype=torch.float32)


def engine(model, **kw):
    kw.setdefault("reserve_blocks", 8)
    return ContinuousBatchingEngine(
        model, kw.pop("pool", None) or pool(), eos_token_id=EOS, device="cpu", **kw
    )


def req(prompt, max_tokens, rid=None, **kw):
    kw.setdefault("ignore_eos", True)
    r = Request(
        prompt_token_ids=list(prompt),
        params=SamplingParams(max_tokens=max_tokens, **kw),
    )
    if rid:
        r.request_id = rid
    return r


def spec_workload(n=12, seed=0):
    """Fixed prompts, so two engines can be compared request by request."""
    rng = random.Random(seed)
    return [
        ([rng.randrange(1, 250) for _ in range(rng.randint(5, 60))], rng.randint(3, 40))
        for _ in range(n)
    ]


def build(specs):
    return [req(p, n, rid=f"spec-{i}") for i, (p, n) in enumerate(specs)]


# ---------------------------------------------------------------------------
# The headline invariant
# ---------------------------------------------------------------------------


def test_batched_output_is_identical_to_sequential(model):
    """Every sequence produces exactly what it would have produced alone."""
    specs = spec_workload(12)

    ref = {r.request_id: r.output_token_ids
           for r in BaselineEngine(model, eos_token_id=EOS, device="cpu").run(build(specs))}
    got = {r.request_id: r.output_token_ids for r in engine(model).run(build(specs))}

    assert set(got) == set(ref)
    mismatched = [k for k in ref if got[k] != ref[k]]
    assert not mismatched, f"{len(mismatched)} sequences differ: {mismatched[:3]}"


@pytest.mark.parametrize("batch_cap", [1, 2, 4, 32])
def test_result_does_not_depend_on_batch_size(model, batch_cap):
    """Batch size is a throughput knob, never a correctness one."""
    specs = spec_workload(8, seed=3)
    ref = {r.request_id: r.output_token_ids
           for r in BaselineEngine(model, eos_token_id=EOS, device="cpu").run(build(specs))}
    got = {r.request_id: r.output_token_ids
           for r in engine(model, max_batch_size=batch_cap).run(build(specs))}
    assert got == ref


def test_padding_mask_is_load_bearing(model):
    """Without it, short sequences read the tail of longer ones.

    Constructed to maximise the effect: one 4-token sequence batched with one
    200-token sequence, so 196 positions of padding are in play. This asserts
    the mask exists and is applied -- removing it makes the short sequence's
    output change, which is exactly the silent corruption being guarded.
    """
    short_spec = ([7] * 4, 6)
    long_spec = ([11] * 200, 6)

    alone = BaselineEngine(model, eos_token_id=EOS, device="cpu").run(
        [req(short_spec[0], short_spec[1], rid="short")]
    )[0].output_token_ids

    together = {
        r.request_id: r.output_token_ids
        for r in engine(model).run([
            req(long_spec[0], long_spec[1], rid="long"),
            req(short_spec[0], short_spec[1], rid="short"),
        ])
    }
    assert together["short"] == alone


def test_mask_marks_exactly_the_valid_positions(model):
    """The mask itself, checked directly rather than through its effect."""
    p = pool()
    seqs = []
    for n in (5, 17, 40):
        cache = p.allocate()
        with torch.inference_mode():
            model(torch.randint(0, CFG.vocab_size, (1, n)), cache=cache)
        seqs.append(Sequence(request=req([1], 10), cache=cache, next_token=1))

    batched = BatchedPagedCache(p, seqs)
    batched._prepare()

    # _prepare grows every sequence by one token.
    assert batched.mask.shape == (3, 1, 1, 41)
    for i, n in enumerate((6, 18, 41)):
        assert int(batched.mask[i, 0, 0].sum()) == n
        assert bool(batched.mask[i, 0, 0, :n].all())
        assert not bool(batched.mask[i, 0, 0, n:].any())


# ---------------------------------------------------------------------------
# Head-of-line blocking
# ---------------------------------------------------------------------------


def test_short_request_no_longer_waits_for_a_long_one(model):
    """The failure continuous batching exists to remove.

    The sequential engine makes the short request wait for all 400 tokens of
    the long one. Here they run together, so the short one finishes in its own
    time.
    """
    long_req = req([5] * 20, 400, rid="long")
    short_req = req([5] * 20, 4, rid="short")

    done = engine(model).run([long_req, short_req])
    order = [r.request_id for r in done]

    assert order[0] == "short", f"short request did not finish first: {order}"
    assert short_req.output_len == 4
    assert long_req.output_len == 400


def test_sequences_join_the_batch_mid_flight(model):
    """A request arriving during a generation starts on the next step."""
    e = engine(model)
    e.add_request(req([5] * 20, 60, rid="first"))
    e.step()
    assert len(e.running) == 1

    e.add_request(req([5] * 20, 5, rid="late"))
    e.step()
    assert len(e.running) == 2, "late arrival did not join the running batch"

    done = []
    while e.has_work():
        done.extend(e.step())
    assert [r.request_id for r in done][0] == "late"


def test_batch_shrinks_as_sequences_finish(model):
    """Slots are returned at the step the sequence ends, not at the end."""
    e = engine(model)
    for i, n in enumerate((3, 6, 9, 60)):
        e.add_request(req([5] * 10, n, rid=f"r{i}"))

    sizes = []
    while e.has_work():
        e.step()
        sizes.append(len(e.running))

    assert max(sizes) == 4
    assert sizes[-1] == 0
    # Strictly decreasing after the peak, as the short ones retire.
    assert sizes[:6] == sorted(sizes[:6], reverse=True)


# ---------------------------------------------------------------------------
# Stop conditions
# ---------------------------------------------------------------------------


def test_eos_retires_only_the_sequence_that_emitted_it(model):
    """One sequence stopping must not disturb the rest of the batch."""
    e = engine(model)
    calls = {"n": 0}
    real = e._sample_batch

    def patched(logits, batch):
        calls["n"] += 1
        out = real(logits, batch)
        if calls["n"] == 3:  # third decode step: force the first to stop
            out = out.clone()
            out[0] = EOS
        return out

    e._sample_batch = patched
    reqs = [
        req([5] * 10, 50, rid="stops", ignore_eos=False),
        req([5] * 10, 8, rid="continues"),
    ]
    done = {r.request_id: r for r in e.run(reqs)}

    assert done["stops"].finish_reason is FinishReason.EOS
    assert done["stops"].output_len < 50
    assert done["continues"].finish_reason is FinishReason.LENGTH
    assert done["continues"].output_len == 8


def test_max_tokens_is_exact_for_every_sequence(model):
    reqs = [req([5] * 10, n, rid=f"r{n}") for n in (1, 2, 7, 16, 17, 33)]
    done = {r.request_id: r for r in engine(model).run(reqs)}
    for n in (1, 2, 7, 16, 17, 33):
        assert done[f"r{n}"].output_len == n
        assert done[f"r{n}"].finish_reason is FinishReason.LENGTH


# ---------------------------------------------------------------------------
# Resource safety
# ---------------------------------------------------------------------------


def test_all_blocks_are_returned_after_a_run(model):
    p = pool()
    e = engine(model, pool=p)
    e.run(build(spec_workload(20, seed=5)))
    assert p.allocator.num_allocated == 0
    assert all(r == 0 for r in p.allocator._refs)


def test_a_small_pool_still_completes_every_request(model):
    """Admission throttles rather than failing when memory is tight."""
    p = pool(num_blocks=48)  # 768 tokens total
    e = engine(model, pool=p, reserve_blocks=4, max_batch_size=8)
    reqs = build(spec_workload(15, seed=7))
    done = e.run(reqs)

    assert len(done) == 15
    assert all(r.finish_reason is FinishReason.LENGTH for r in done)
    assert p.allocator.num_allocated == 0


def test_memory_pressure_preempts_rather_than_failing(model):
    """A pool too small for the offered load must still complete everything.

    Ten sequences of ~70 tokens need 700 tokens of KV; this pool holds 640. No
    admission policy can avoid that, because the shortfall only materialises as
    the sequences grow -- reserving for the worst case up front would put the
    engine back at the contiguous allocator's concurrency limit. The engine
    evicts the newest sequence onto the queue and re-runs it later, so every
    request still finishes.
    """
    p = pool(num_blocks=40)  # 640 tokens
    e = ContinuousBatchingEngine(
        model, p, eos_token_id=EOS, device="cpu", reserve_blocks=4, max_batch_size=16
    )
    done = e.run([req([5] * 30, 40, rid=f"r{i}") for i in range(10)])

    assert len(done) == 10
    assert all(r.finish_reason is FinishReason.LENGTH for r in done)
    assert all(r.output_len == 40 for r in done), "a preempted request lost tokens"
    assert e.preemptions > 0, "this configuration should have forced eviction"
    assert p.allocator.num_allocated == 0


def test_preempted_output_matches_an_uninterrupted_run(model):
    """Eviction must be invisible in the result, visible only in the timing.

    A preempted sequence is re-prefilled from its prompt, so it has to land on
    exactly the tokens it would have produced uninterrupted. If it does not,
    preemption is silently corrupting output under memory pressure -- the worst
    possible moment for it.

    Uses uniform long sequences rather than the varied workload: with mixed
    lengths, admission throttles enough that eviction never fires, which is the
    right behaviour but tests nothing here.
    """
    specs = [([7] * 30, 40) for _ in range(10)]
    ref = {r.request_id: r.output_token_ids
           for r in BaselineEngine(model, eos_token_id=EOS, device="cpu").run(build(specs))}

    p = pool(num_blocks=40)
    e = ContinuousBatchingEngine(
        model, p, eos_token_id=EOS, device="cpu", reserve_blocks=4, max_batch_size=16
    )
    got = {r.request_id: r.output_token_ids for r in e.run(build(specs))}

    assert e.preemptions > 0, "expected memory pressure in this configuration"
    assert got == ref
    assert p.allocator.num_allocated == 0


def test_admission_throttles_instead_of_evicting_when_it_can(model):
    """Eviction is the fallback, not the mechanism.

    A pool far too small for the whole workload should still run it to
    completion by admitting fewer sequences at a time, without a single
    preemption. Down to 8 blocks -- 128 tokens against a 539-token workload --
    the scaling reserve keeps every admitted sequence able to grow.
    """
    specs = spec_workload(10, seed=9)
    ref = {r.request_id: r.output_token_ids
           for r in BaselineEngine(model, eos_token_id=EOS, device="cpu").run(build(specs))}

    for num_blocks in (30, 12, 8):
        p = pool(num_blocks=num_blocks)
        e = ContinuousBatchingEngine(
            model, p, eos_token_id=EOS, device="cpu", reserve_blocks=1, max_batch_size=16
        )
        got = {r.request_id: r.output_token_ids for r in e.run(build(specs))}

        assert got == ref, f"wrong output at {num_blocks} blocks"
        assert e.preemptions == 0, f"evicted at {num_blocks} blocks; admission should throttle"
        assert p.allocator.num_allocated == 0


def test_engine_stats_track_the_batch(model):
    p = pool()
    e = engine(model, pool=p)
    assert e.stats().num_running == 0

    for i in range(4):
        e.add_request(req([5] * 10, 30, rid=f"r{i}"))
    e.step()

    s = e.stats()
    assert s.num_running == 4
    assert s.num_waiting == 0
    assert s.kv_blocks_used > 0
    assert 0 < s.kv_utilization <= 1
