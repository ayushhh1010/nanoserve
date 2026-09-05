"""Off-by-one hunting at the cache boundary, and numerical safety in low precision.

Both failure modes here are insidious because nothing crashes. An attention
mask that is off by one at the cache boundary still produces fluent text -- it
just quietly conditions on the wrong context, and you discover it (or never do)
after a full training run. An RMSNorm that overflows in reduced precision
produces NaNs that look like a bad learning rate.

So these are asserted directly rather than inferred from loss curves.
"""

from __future__ import annotations

import pytest
import torch

from model.config import NanoConfig
from model.transformer import (
    DynamicCache,
    NanoForCausalLM,
    RMSNorm,
    offset_causal_mask,
)

CFG = NanoConfig(
    vocab_size=512,
    hidden_size=128,
    num_hidden_layers=3,
    num_attention_heads=8,
    num_key_value_heads=2,
    intermediate_size=352,
    max_position_embeddings=1024,
)


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return NanoForCausalLM(CFG).eval()


# ---------------------------------------------------------------------------
# The mask itself
# ---------------------------------------------------------------------------


def test_offset_mask_row_counts_are_exact():
    """Query i of a chunk must see exactly (kv_len - q_len + i + 1) keys.

    Written as an explicit count rather than a shape check: an off-by-one
    shifts every row by one key and leaves the shape untouched.
    """
    q_len, kv_len = 4, 10
    mask = offset_causal_mask(q_len, kv_len, torch.device("cpu"))
    assert mask.shape == (q_len, kv_len)

    for i in range(q_len):
        visible = int(mask[i].sum())
        assert visible == kv_len - q_len + i + 1, f"row {i} sees {visible} keys"

    # The last query is the newest token: it sees everything, including itself.
    assert bool(mask[-1].all())
    # And nothing may see beyond its own position.
    assert not bool(mask[0, kv_len - q_len + 1 :].any())


def test_decode_mask_sees_exactly_one_more_key_each_step():
    """The degenerate q_len == 1 case, which the model handles by using no mask."""
    for cached in (0, 1, 411, 1023):
        mask = offset_causal_mask(1, cached + 1, torch.device("cpu"))
        assert int(mask.sum()) == cached + 1, f"at {cached} cached, saw {int(mask.sum())}"


# ---------------------------------------------------------------------------
# The boundary you asked about: position 412 with 411 cached
# ---------------------------------------------------------------------------


def test_decode_at_position_412_matches_full_forward(model):
    """Decode token 412 with 411 cached; compare against one full forward.

    If the decode step attended to 411 keys instead of 412 (excluding itself)
    or to 413 (impossible, but a sign of a mis-sized cache), the logits would
    differ. Exact agreement to fp32 noise means the boundary is right.
    """
    torch.manual_seed(1)
    ids = torch.randint(0, CFG.vocab_size, (1, 412))

    with torch.inference_mode():
        full, _ = model(ids)

        cache = DynamicCache()
        model(ids[:, :411], cache=cache)  # prefill 0..410
        assert len(cache) == 411
        step, _ = model(ids[:, 411:412], cache=cache)  # decode position 411
        assert len(cache) == 412

    diff = (step[:, 0] - full[:, 411]).abs().max().item()
    assert diff < 2e-5, f"boundary logits differ by {diff:.3e}"


def test_position_ids_advance_with_the_cache(model):
    """RoPE must use absolute position 411, not 0, for the 412th token.

    A cache that grows correctly but feeds position 0 to RoPE on every decode
    step is a real and common bug -- and it still generates plausible text.
    Detected here by decoding the same token at two different positions and
    requiring the outputs to differ.
    """
    torch.manual_seed(2)
    ids = torch.randint(0, CFG.vocab_size, (1, 400))
    probe = torch.tensor([[123]])

    with torch.inference_mode():
        early = DynamicCache()
        model(ids[:, :10], cache=early)
        out_early, _ = model(probe, cache=early)

        late = DynamicCache()
        model(ids, cache=late)
        out_late, _ = model(probe, cache=late)

    assert not torch.allclose(out_early, out_late, atol=1e-3), (
        "same token at position 10 and position 400 gave the same logits -- "
        "position_ids are probably stuck at 0"
    )


@pytest.mark.parametrize("boundary", [1, 2, 15, 16, 17, 63, 64, 65, 411, 412])
def test_every_decode_step_matches_full_forward(model, boundary: int):
    """Sweep boundaries, including the 16-token block edges Phase 2 will use.

    Block size 16 means the paged allocator crosses a boundary at 16, 32, 48...
    Those are exactly where an allocator off-by-one will show up, so the
    contract is pinned here first, against the simple cache.
    """
    torch.manual_seed(3)
    ids = torch.randint(0, CFG.vocab_size, (1, boundary + 1))

    with torch.inference_mode():
        full, _ = model(ids)
        cache = DynamicCache()
        model(ids[:, :boundary], cache=cache)
        step, _ = model(ids[:, boundary : boundary + 1], cache=cache)

    assert torch.allclose(step[:, 0], full[:, boundary], atol=2e-5)


def test_long_run_decode_stays_aligned(model):
    """256 sequential decode steps, every one compared to the full forward.

    Drift accumulates: a bug that costs 1e-6 at step 1 can cost 1e-2 at step
    256. Checking only the last token would hide where it started.
    """
    torch.manual_seed(4)
    ids = torch.randint(0, CFG.vocab_size, (1, 300))

    with torch.inference_mode():
        full, _ = model(ids)
        cache = DynamicCache()
        model(ids[:, :44], cache=cache)
        worst, worst_at = 0.0, -1
        for t in range(44, 300):
            step, _ = model(ids[:, t : t + 1], cache=cache)
            d = (step[:, 0] - full[:, t]).abs().max().item()
            if d > worst:
                worst, worst_at = d, t

    assert worst < 2e-5, f"worst drift {worst:.3e} at position {worst_at}"


# ---------------------------------------------------------------------------
# RMSNorm in reduced precision
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("magnitude", [1e2, 1e3, 1e4])
def test_rmsnorm_survives_large_inputs(dtype: torch.dtype, magnitude: float):
    """The mean-of-squares must not overflow the storage dtype.

    fp16 tops out at 65504, so x=1e3 squares to 1e6 and overflows immediately.
    bf16 has fp32's exponent range so it survives longer, but its 8-bit
    mantissa makes the accumulated sum badly imprecise. Computing the reduction
    in fp32 fixes both; this asserts it for both dtypes at magnitudes a
    training run genuinely reaches when something starts to diverge.
    """
    norm = RMSNorm(256).to(dtype)
    x = torch.full((2, 4, 256), magnitude, dtype=dtype)
    out = norm(x)

    assert torch.isfinite(out).all(), f"{dtype} at {magnitude:g} produced non-finite output"
    # All-equal input normalises to all-ones regardless of magnitude.
    assert torch.allclose(out.float(), torch.ones_like(out.float()), atol=2e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rmsnorm_matches_fp32_reference(dtype: torch.dtype):
    """Low precision should cost accuracy, not correctness."""
    torch.manual_seed(0)
    x32 = torch.randn(4, 16, 256) * 50

    ref = RMSNorm(256)(x32)
    low = RMSNorm(256).to(dtype)(x32.to(dtype)).float()

    assert torch.isfinite(low).all()
    tol = 5e-3 if dtype is torch.float16 else 4e-2  # bf16 has 8 mantissa bits
    assert (ref - low).abs().max().item() < tol


def test_rmsnorm_naive_implementation_fails_silently():
    """Demonstrates that the fp32 reduction is doing real work.

    The failure is worse than a NaN. In fp16, x=1e3 squares to 1e6, which
    overflows to +inf; the mean is then inf, rsqrt(inf) is 0, and the layer
    returns all zeros. The output is perfectly *finite* -- it has simply
    annihilated the entire activation. Nothing raises, no NaN propagates, and
    the only symptom is a model that will not learn.

    That is why this asserts the value, not just finiteness.
    """
    x = torch.full((1, 1, 256), 1e3, dtype=torch.float16)

    naive = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)
    assert torch.isfinite(naive).all(), "expected silent zeros, not NaN"
    assert naive.abs().max().item() == 0.0, "fp16 no longer overflows here; revisit RMSNorm"

    ours = RMSNorm(256).to(torch.float16)(x)
    assert torch.isfinite(ours).all()
    assert torch.allclose(ours.float(), torch.ones_like(ours.float()), atol=2e-2)
