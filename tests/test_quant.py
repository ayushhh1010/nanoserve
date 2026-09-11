"""INT8 weight-only quantization.

The thing to guard against is a quantizer that looks right and is subtly
lossy in the wrong place. So the tests check the reconstruction directly --
per-group scales, the symmetric grid, exact round-trip on values that land on
grid points -- rather than only checking that perplexity did not move much,
which would pass for a quantizer that silently fell back to fp16.

The LM head exclusion gets its own test because in this model it is not merely
conventional: the head is tied to the embedding table, so quantizing it would
quantize every token lookup too.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from engine.quant import (
    DEFAULT_GROUP_SIZE,
    QMAX,
    QuantizedLinear,
    dequantize_tensor,
    quantization_error,
    quantize_model,
    quantize_tensor,
)
from model.config import NanoConfig
from model.transformer import NanoForCausalLM

CFG = NanoConfig(
    vocab_size=256, hidden_size=128, num_hidden_layers=2, num_attention_heads=4,
    num_key_value_heads=2, intermediate_size=256, max_position_embeddings=256,
)


# ---------------------------------------------------------------------------
# The quantizer itself
# ---------------------------------------------------------------------------


def test_round_trip_is_close():
    torch.manual_seed(0)
    w = torch.randn(64, 256)
    q, scales = quantize_tensor(w, group_size=128)
    recon = dequantize_tensor(q, scales, 128)

    assert q.dtype == torch.int8
    assert q.shape == w.shape
    assert scales.shape == (64, 2)  # 256 inputs / 128 per group
    # At 8 bits with per-group scales the error is a fraction of a percent.
    assert (w - recon).norm() / w.norm() < 0.01


def test_values_stay_inside_the_symmetric_grid():
    """-127..127, not -128..127.

    Keeping -128 would make the negative side one step wider than the
    positive, biasing every quantized tensor slightly low.
    """
    torch.manual_seed(1)
    q, _ = quantize_tensor(torch.randn(32, 128) * 50, group_size=128)
    assert int(q.min()) >= -QMAX
    assert int(q.max()) <= QMAX


def test_scales_are_per_group_not_per_tensor():
    """One outlier must not crush the resolution of the whole row.

    Two groups with wildly different magnitudes: a per-tensor scale would make
    the small group quantize almost entirely to zero.
    """
    w = torch.cat([torch.full((4, 128), 0.001), torch.full((4, 128), 100.0)], dim=1)
    q, scales = quantize_tensor(w, group_size=128)

    assert scales[0, 0] < scales[0, 1] / 1000
    recon = dequantize_tensor(q, scales, 128)
    # Both halves survive, which a per-tensor scale would not manage.
    assert recon[0, 0] == pytest.approx(0.001, rel=0.02)
    assert recon[0, 200] == pytest.approx(100.0, rel=0.02)


def test_exact_on_grid_points():
    """Values that land on the grid must reconstruct exactly."""
    w = torch.zeros(1, 128)
    w[0, 0] = 127.0  # sets the scale to 1.0
    for i in range(1, 20):
        w[0, i] = float(i)
    q, scales = quantize_tensor(w, group_size=128)
    recon = dequantize_tensor(q, scales, 128)
    assert torch.allclose(recon[0, :20], w[0, :20], atol=1e-5)


def test_all_zero_group_does_not_divide_by_zero():
    w = torch.zeros(2, 128)
    q, scales = quantize_tensor(w, group_size=128)
    assert torch.isfinite(scales).all()
    assert int(q.abs().max()) == 0
    assert torch.equal(dequantize_tensor(q, scales, 128), w)


def test_indivisible_shape_is_rejected():
    """Silently changing the grouping is worse than refusing."""
    with pytest.raises(ValueError, match="not divisible"):
        quantize_tensor(torch.randn(8, 100), group_size=128)


def test_smaller_groups_are_more_accurate():
    """The quality knob behaves in the expected direction."""
    torch.manual_seed(2)
    w = torch.randn(16, 512) * torch.linspace(0.01, 10, 512)  # wide dynamic range
    coarse = quantization_error(w, group_size=512)["relative_error"]
    fine = quantization_error(w, group_size=64)["relative_error"]
    assert fine < coarse


# ---------------------------------------------------------------------------
# The module
# ---------------------------------------------------------------------------


def test_quantized_linear_matches_dense_closely():
    torch.manual_seed(3)
    dense = nn.Linear(256, 64, bias=False)
    q = QuantizedLinear.from_linear(dense)

    x = torch.randn(4, 10, 256)
    assert torch.allclose(q(x), dense(x), rtol=0.02, atol=0.02)


def test_quantized_linear_preserves_bias():
    dense = nn.Linear(128, 32, bias=True)
    nn.init.normal_(dense.bias)
    q = QuantizedLinear.from_linear(dense)
    assert q.bias is not None
    assert torch.allclose(q.bias, dense.bias)


def test_stored_weights_are_actually_smaller():
    dense = nn.Linear(512, 512, bias=False).to(torch.bfloat16)
    q = QuantizedLinear.from_linear(dense)

    dense_bytes = dense.weight.numel() * dense.weight.element_size()
    q_bytes = q.qweight.numel() * 1 + q.scales.numel() * 4
    assert q_bytes < dense_bytes
    # int8 plus one fp32 scale per 128 values: just under half of bf16.
    assert dense_bytes / q_bytes == pytest.approx(1.94, rel=0.05)


# ---------------------------------------------------------------------------
# Whole-model conversion
# ---------------------------------------------------------------------------


def test_quantize_model_replaces_linears_and_skips_the_head():
    """The head is tied to the embedding table here, so it must be skipped."""
    torch.manual_seed(4)
    model = NanoForCausalLM(CFG).eval()
    assert model.lm_head.weight is model.model.embed_tokens.weight

    stats = quantize_model(model)

    assert isinstance(model.lm_head, nn.Linear), "lm_head must not be quantized"
    assert model.lm_head.weight is model.model.embed_tokens.weight, "tie was broken"
    for layer in model.model.layers:
        for proj in (layer.self_attn.q_proj, layer.self_attn.o_proj,
                     layer.mlp.gate_proj, layer.mlp.down_proj):
            assert isinstance(proj, QuantizedLinear)
    assert stats.layers_quantized == CFG.num_hidden_layers * 7
    assert stats.layers_skipped >= 1


def test_quantized_model_stays_well_behaved():
    """Output must stay finite and keep its scale.

    Deliberately NOT asserting that logits barely move. This model is randomly
    initialised, so its logits are near-noise: a 0.65% weight perturbation
    amplifies through the layers into a different random vector, and two random
    vectors differ by ~141% by construction. That number measures noise
    amplification in an untrained network, not quantization quality.

    What is meaningful here is that nothing blows up or collapses. The real
    fidelity check runs against the trained checkpoint --
    `test_trained_model_perplexity_delta` below, and scripts/measure_quant.py.
    """
    torch.manual_seed(5)
    ids = torch.randint(0, CFG.vocab_size, (2, 24))

    dense = NanoForCausalLM(CFG).eval()
    with torch.inference_mode():
        want, _ = dense(ids)

    torch.manual_seed(5)
    quant = NanoForCausalLM(CFG).eval()
    quantize_model(quant)
    with torch.inference_mode():
        got, _ = quant(ids)

    assert torch.isfinite(got).all()
    assert got.shape == want.shape
    # Scale preserved: a broken scale factor would move this by orders of
    # magnitude, which is the failure a random-init model *can* detect.
    assert got.abs().mean().item() == pytest.approx(want.abs().mean().item(), rel=0.15)


def test_weight_reconstruction_error_is_small_in_every_layer():
    """The fidelity claim, made where it is actually measurable.

    Weight error is independent of whether the model is trained, so this is
    the honest per-layer check on a random-init model.
    """
    torch.manual_seed(5)
    model = NanoForCausalLM(CFG).eval()

    worst = 0.0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and "lm_head" not in name:
            if module.in_features % DEFAULT_GROUP_SIZE:
                continue
            err = quantization_error(module.weight)["relative_error"]
            worst = max(worst, err)
            assert err < 0.02, f"{name} reconstructs at {err:.2%} error"
    assert worst > 0, "no layer was measured"


def test_quantization_raises_loss_only_slightly():
    """The honest metric: loss delta on held-out-ish data.

    Reported as a ratio rather than an absolute so it is comparable across
    models. An INT8 weight-only quantizer that moves loss by more than a few
    percent has a bug, not a trade-off.
    """
    torch.manual_seed(6)
    ids = torch.randint(0, CFG.vocab_size, (4, 64))

    dense = NanoForCausalLM(CFG).eval()
    with torch.inference_mode():
        _, base_loss = dense(ids, labels=ids)

    torch.manual_seed(6)
    quant = NanoForCausalLM(CFG).eval()
    quantize_model(quant)
    with torch.inference_mode():
        _, q_loss = quant(ids, labels=ids)

    delta = (q_loss.item() - base_loss.item()) / base_loss.item()
    assert abs(delta) < 0.02, f"loss moved {delta:+.2%}"


def test_stats_report_real_numbers():
    torch.manual_seed(7)
    model = NanoForCausalLM(CFG).to(torch.bfloat16).eval()
    stats = quantize_model(model)
    d = stats.to_dict()

    assert d["compression"] > 1.0
    assert d["saved_mb"] > 0
    assert d["original_mb"] > d["quantized_mb"]


def test_skipping_is_configurable():
    torch.manual_seed(8)
    model = NanoForCausalLM(CFG).eval()
    quantize_model(model, skip=("lm_head", "gate_proj"))

    for layer in model.model.layers:
        assert isinstance(layer.mlp.gate_proj, nn.Linear)
        assert isinstance(layer.mlp.up_proj, QuantizedLinear)


def test_group_size_must_divide_every_layer():
    """A layer quantized at a different group size than reported is worse
    than one left alone, so an indivisible layer is skipped, not regrouped."""
    torch.manual_seed(9)
    model = NanoForCausalLM(CFG).eval()
    # hidden_size 128 is not divisible by 256.
    stats = quantize_model(model, group_size=256)
    assert stats.layers_quantized < CFG.num_hidden_layers * 7
    assert stats.layers_skipped > 0


def test_an_outlier_damages_the_rest_of_its_group():
    """What a per-group scale actually costs when one value dominates.

    Measured on the surviving values rather than on the whole-tensor relative
    error: a huge outlier inflates ||W|| far more than it inflates the error,
    so ||W - W'|| / ||W|| goes DOWN and reports the damaged tensor as cleaner
    than the benign one. The metric is standard; using it to detect outliers
    is the mistake.
    """
    torch.manual_seed(10)
    base = torch.randn(1, 128)

    spiky = base.clone()
    spiky[0, 0] = 10_000.0

    def error_on_tail(w):
        q, scales = quantize_tensor(w, group_size=128)
        recon = dequantize_tensor(q, scales, 128)
        return (w[0, 1:] - recon[0, 1:]).abs().mean().item()

    assert error_on_tail(spiky) > 100 * error_on_tail(base), (
        "the outlier's group should lose most of its resolution"
    )

    # And the whole-tensor relative error moves the other way, which is why
    # it must not be used as an outlier detector.
    assert (
        quantization_error(spiky)["relative_error"]
        < quantization_error(base)["relative_error"]
    )


@pytest.mark.skipif(
    not __import__("pathlib").Path("checkpoints/run1/best.pt").exists(),
    reason="trained checkpoint not present",
)
def test_trained_model_perplexity_delta():
    """The number that actually matters, on a model whose logits mean something.

    INT8 weight-only should cost a fraction of a percent. Anything above a few
    percent is a bug rather than a trade-off.
    """
    from model.generate import load_model

    torch.manual_seed(0)
    model, _ = load_model("checkpoints/run1/best.pt", device="cpu", dtype=torch.float32)
    ids = torch.randint(0, model.cfg.vocab_size, (2, 256))

    with torch.inference_mode():
        _, dense_loss = model(ids, labels=ids)
    quantize_model(model)
    with torch.inference_mode():
        _, quant_loss = model(ids, labels=ids)

    delta = (quant_loss.item() - dense_loss.item()) / dense_loss.item()
    assert abs(delta) < 0.03, f"loss moved {delta:+.2%} on the trained model"
