"""Structural and statistical properties of the model.

Parity (tests/test_parity.py) proves the maths matches a known-good reference.
This file proves the things a reference cannot tell you: that the model has the
size it claims, that the tied weights are actually tied, and that it starts
training from the right place.
"""

from __future__ import annotations

import math

import pytest
import torch

from model.config import NANO_27M, TINY, NanoConfig, count_parameters
from model.transformer import DynamicCache, NanoForCausalLM, RMSNorm, repeat_kv


# ---------------------------------------------------------------------------
# Shape and size
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cfg", [TINY, NANO_27M], ids=["tiny", "nano27m"])
def test_analytic_param_count_matches_reality(cfg: NanoConfig):
    """The count in config.py is used for planning; it must not drift."""
    model = NanoForCausalLM(cfg)
    assert model.num_parameters() == count_parameters(cfg)["total"]
    assert model.num_parameters(non_embedding=True) == count_parameters(cfg)["non_embedding"]


def test_nano27m_is_actually_27m():
    """The headline number in the README and on the resume."""
    total = count_parameters(NANO_27M)["total"]
    assert 26.0e6 < total < 27.5e6, f"{total:,} params is not ~27M"


def test_kv_cache_budget():
    """4 KB/token is what Phase 2's block manager gets sized against.

    A block of 16 tokens is therefore 64 KB, and a full 1024-token sequence is
    4 MB of KV. Without GQA (8 KV heads instead of 2) this would be 16 KB/token
    and every Phase 2 memory number would be four times worse.
    """
    assert NANO_27M.kv_bytes_per_token == 4096
    full_seq_mb = NANO_27M.kv_bytes_per_token * NANO_27M.max_position_embeddings / 1024**2
    assert full_seq_mb == pytest.approx(4.0)


def test_weight_tying_shares_one_tensor():
    model = NanoForCausalLM(NanoConfig(tie_word_embeddings=True, vocab_size=64, hidden_size=32))
    assert model.lm_head.weight is model.model.embed_tokens.weight

    untied = NanoForCausalLM(NanoConfig(tie_word_embeddings=False, vocab_size=64, hidden_size=32))
    assert untied.lm_head.weight is not untied.model.embed_tokens.weight


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


def test_initial_loss_is_uniform_entropy():
    """An untrained model should be exactly as confused as chance.

    Loss at step 0 must be ~ln(vocab_size). Materially below it means the head
    was initialised with structure it should not have; materially above means
    something is actively broken. This one number catches an astonishing
    number of init bugs before you burn 10 GPU-hours on them.
    """
    torch.manual_seed(0)
    cfg = NanoConfig(vocab_size=2048, hidden_size=128, num_hidden_layers=4, intermediate_size=352)
    model = NanoForCausalLM(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (4, 64))

    with torch.no_grad():
        _, loss = model(ids, labels=ids)

    expected = math.log(cfg.vocab_size)
    assert loss.item() == pytest.approx(expected, abs=0.15), (
        f"init loss {loss.item():.3f} vs ln(V)={expected:.3f}"
    )


def test_residual_projections_are_depth_scaled():
    """Deeper models must init their residual writes smaller."""
    shallow = NanoForCausalLM(NanoConfig(num_hidden_layers=2, vocab_size=64, hidden_size=32))
    deep = NanoForCausalLM(NanoConfig(num_hidden_layers=32, vocab_size=64, hidden_size=32))

    s = shallow.model.layers[0].mlp.down_proj.weight.std().item()
    d = deep.model.layers[0].mlp.down_proj.weight.std().item()
    assert d < s, "32-layer model should init down_proj smaller than a 2-layer one"
    assert d / s == pytest.approx(math.sqrt(2 / 32), rel=0.15)


def test_residual_stream_variance_stays_bounded_with_depth():
    """The property the scaling exists to produce, measured directly."""
    torch.manual_seed(0)
    ids = torch.randint(0, 512, (2, 32))
    stds = []
    for n_layers in (2, 8, 32):
        cfg = NanoConfig(
            vocab_size=512, hidden_size=128, num_hidden_layers=n_layers, intermediate_size=352
        )
        with torch.no_grad():
            hidden = NanoForCausalLM(cfg).eval().model(ids)
        stds.append(hidden.std().item())

    # Without depth scaling this ratio grows roughly like sqrt(depth) -- 4x
    # across this range. With it, the stream stays in the same ballpark.
    assert max(stds) / min(stds) < 2.0, f"residual std across depths: {stds}"


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------


def test_rmsnorm_normalises_and_survives_fp16_overflow():
    """The fp32 reduction is load-bearing, not defensive decoration."""
    norm = RMSNorm(64)
    x = torch.randn(2, 8, 64) * 100
    out = norm(x)
    assert out.pow(2).mean(-1).sqrt().allclose(torch.ones(2, 8), atol=1e-4)

    # 300^2 = 90_000, comfortably past fp16's 65_504 max. A naive
    # implementation returns NaN here.
    big = torch.full((1, 1, 64), 300.0, dtype=torch.float16)
    assert torch.isfinite(RMSNorm(64).to(torch.float16)(big)).all()


def test_repeat_kv_broadcasts_each_head_to_its_group():
    x = torch.randn(1, 2, 5, 4)  # 2 KV heads
    out = repeat_kv(x, 4)  # -> 8 query heads
    assert out.shape == (1, 8, 5, 4)
    # Query heads 0..3 all read KV head 0; heads 4..7 read KV head 1.
    for q_head in range(8):
        assert torch.equal(out[0, q_head], x[0, q_head // 4])


def test_causality_future_tokens_cannot_change_the_past():
    """Perturb the last token; every earlier logit must be bit-identical."""
    torch.manual_seed(0)
    cfg = NanoConfig(vocab_size=256, hidden_size=64, num_hidden_layers=2, intermediate_size=176)
    model = NanoForCausalLM(cfg).eval()

    a = torch.randint(0, cfg.vocab_size, (1, 10))
    b = a.clone()
    b[0, -1] = (b[0, -1] + 7) % cfg.vocab_size

    with torch.no_grad():
        la, _ = model(a)
        lb, _ = model(b)

    assert torch.equal(la[:, :-1], lb[:, :-1])


def test_cache_length_tracks_tokens_seen():
    torch.manual_seed(0)
    model = NanoForCausalLM(TINY).eval()
    cache = DynamicCache()
    assert len(cache) == 0

    with torch.no_grad():
        model(torch.randint(0, TINY.vocab_size, (1, 5)), cache=cache)
        assert len(cache) == 5
        model(torch.randint(0, TINY.vocab_size, (1, 1)), cache=cache)
        assert len(cache) == 6

    assert len(cache.keys) == TINY.num_hidden_layers
    assert cache.keys[0].shape == (1, TINY.num_key_value_heads, 6, TINY.head_dim)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"hidden_size": 100, "num_attention_heads": 8},  # not divisible
        {"num_attention_heads": 8, "num_key_value_heads": 3},  # not divisible
        {"hidden_size": 12, "num_attention_heads": 4},  # head_dim=3, odd -> no RoPE pairs
    ],
    ids=["hidden-not-divisible", "gqa-not-divisible", "head-dim-odd"],
)
def test_bad_configs_fail_loudly(kwargs):
    with pytest.raises(ValueError):
        NanoConfig(**kwargs)
