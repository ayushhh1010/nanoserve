"""Numerical parity against HuggingFace's Llama reference implementation.

This is the Phase 1 week-1 acceptance gate. Writing a transformer by hand is
only worth anything if it is *correct*, and "the loss goes down" is far too
weak a signal -- a model with a subtly wrong RoPE or a broken causal mask still
trains, just worse, and you find out three days into a run.

So: build the same architecture in both implementations, copy one set of
weights into both, and require the logits to agree to floating-point noise.

The reference is only ever used here. Nothing in `model/` imports transformers.
"""

from __future__ import annotations

import pytest
import torch

from model.config import NanoConfig
from model.transformer import DynamicCache, NanoForCausalLM

transformers = pytest.importorskip("transformers", reason="reference impl not installed")

# fp32 end-to-end, so the tolerance below is genuinely measuring "same maths"
# rather than "same rounding". Anything above ~1e-4 here is a real bug.
ATOL = 2e-5
RTOL = 1e-4


def _configs() -> tuple[NanoConfig, "transformers.LlamaConfig"]:
    """A structurally faithful miniature: GQA, SwiGLU, even head_dim, >1 layer."""
    cfg = NanoConfig(
        vocab_size=512,
        hidden_size=128,
        num_hidden_layers=3,
        num_attention_heads=8,
        num_key_value_heads=2,
        intermediate_size=352,
        max_position_embeddings=64,
        tie_word_embeddings=False,  # untied exercises lm_head as a real matrix
    )
    hf = transformers.LlamaConfig(
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.hidden_size,
        num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.num_key_value_heads,
        intermediate_size=cfg.intermediate_size,
        max_position_embeddings=cfg.max_position_embeddings,
        rms_norm_eps=cfg.rms_norm_eps,
        rope_theta=cfg.rope_theta,
        hidden_act="silu",
        attention_bias=cfg.attention_bias,
        mlp_bias=cfg.mlp_bias,
        tie_word_embeddings=cfg.tie_word_embeddings,
    )
    return cfg, hf


@pytest.fixture(scope="module")
def models():
    torch.manual_seed(0)
    cfg, hf_cfg = _configs()

    ours = NanoForCausalLM(cfg).eval().to(torch.float32)
    ref = transformers.LlamaForCausalLM(hf_cfg).eval().to(torch.float32)

    # Parameter names are identical by construction, which is the point of
    # mirroring HF's naming: this is a plain load, not a conversion.
    missing, unexpected = ref.load_state_dict(ours.state_dict(), strict=False)
    assert not unexpected, f"our state dict has keys the reference does not: {unexpected}"
    assert not [k for k in missing if "rotary" not in k and "inv_freq" not in k], (
        f"reference has weights we never produced: {missing}"
    )
    return ours, ref, cfg


def _ref_logits(ref, input_ids, **kw):
    with torch.no_grad():
        return ref(input_ids=input_ids, use_cache=False, **kw).logits


def _assert_close(got: torch.Tensor, want: torch.Tensor, what: str) -> None:
    diff = (got - want).abs().max().item()
    assert torch.allclose(got, want, atol=ATOL, rtol=RTOL), (
        f"{what}: max abs diff {diff:.3e} exceeds atol={ATOL:.1e}"
    )


def test_prefill_logits_match(models):
    """The base case: a full forward pass over a batch, no cache."""
    ours, ref, cfg = models
    torch.manual_seed(1)
    ids = torch.randint(0, cfg.vocab_size, (3, 17))

    with torch.no_grad():
        got, _ = ours(ids)
    want = _ref_logits(ref, ids)

    assert got.shape == want.shape == (3, 17, cfg.vocab_size)
    _assert_close(got, want, "prefill logits")


def test_single_token_matches_prefix_of_batch(models):
    """A length-1 sequence and a length-1 slice of a longer one agree.

    Catches the family of bugs where the causal mask silently depends on the
    sequence length rather than on position.
    """
    ours, _ref, cfg = models
    torch.manual_seed(2)
    ids = torch.randint(0, cfg.vocab_size, (1, 12))

    with torch.no_grad():
        full, _ = ours(ids)
        first, _ = ours(ids[:, :1])

    _assert_close(first[:, 0], full[:, 0], "first-token logits")


def test_incremental_decode_matches_full_forward(models):
    """Token-by-token decode with the KV cache equals one full forward pass.

    This is the invariant Phase 2's paged cache will also have to satisfy, so
    it is worth pinning down now against an implementation known to be right.
    """
    ours, _ref, cfg = models
    torch.manual_seed(3)
    ids = torch.randint(0, cfg.vocab_size, (2, 9))

    with torch.no_grad():
        full, _ = ours(ids)

        cache = DynamicCache()
        stepwise = []
        for t in range(ids.shape[1]):
            step, _ = ours(ids[:, t : t + 1], cache=cache)
            stepwise.append(step)
        got = torch.cat(stepwise, dim=1)

    assert len(cache) == ids.shape[1]
    _assert_close(got, full, "incremental decode logits")


def test_chunked_prefill_matches_full_forward(models):
    """Prefill split into chunks equals prefill in one shot.

    Exercises the offset causal mask (1 < q_len < kv_len), the regime that
    neither `is_causal=True` nor "no mask at all" handles correctly.
    """
    ours, _ref, cfg = models
    torch.manual_seed(4)
    ids = torch.randint(0, cfg.vocab_size, (1, 20))

    with torch.no_grad():
        full, _ = ours(ids)

        cache = DynamicCache()
        chunks = []
        for start in range(0, 20, 7):  # 7, 7, 6 -- deliberately uneven
            piece, _ = ours(ids[:, start : start + 7], cache=cache)
            chunks.append(piece)
        got = torch.cat(chunks, dim=1)

    _assert_close(got, full, "chunked prefill logits")


def test_greedy_generation_matches_reference(models):
    """End to end: the two implementations emit the same tokens.

    Logit parity implies this, but token parity is the claim that actually
    matters downstream -- Phase 2 asserts the optimised engine reproduces the
    baseline token for token, and this is the same assertion one level down.
    """
    ours, ref, cfg = models
    torch.manual_seed(5)
    prompt = torch.randint(0, cfg.vocab_size, (1, 6))

    ours_ids = prompt.clone()
    cache = DynamicCache()
    with torch.no_grad():
        logits, _ = ours(ours_ids, cache=cache)
        for _ in range(10):
            nxt = logits[:, -1].argmax(-1, keepdim=True)
            ours_ids = torch.cat([ours_ids, nxt], dim=1)
            logits, _ = ours(nxt, cache=cache)

        ref_ids = ref.generate(
            prompt, max_new_tokens=10, do_sample=False, pad_token_id=cfg.vocab_size - 1
        )

    assert torch.equal(ours_ids, ref_ids), (
        f"token mismatch\n  ours: {ours_ids.tolist()}\n  ref:  {ref_ids.tolist()}"
    )


def test_dtype_and_device_roundtrip(models):
    """bf16 on the GPU stays finite and stays close to the fp32 answer.

    Not a parity check -- bf16 has ~3 decimal digits, so the tolerance is loose
    on purpose. It is a smoke test that nothing overflows, which is the failure
    mode RMSNorm's fp32 reduction exists to prevent.
    """
    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    ours, _ref, cfg = models
    torch.manual_seed(6)
    ids = torch.randint(0, cfg.vocab_size, (2, 16))

    with torch.no_grad():
        fp32 = ours(ids)[0]
        gpu = ours.to("cuda", torch.bfloat16)
        bf16 = gpu(ids.to("cuda"))[0].float().cpu()
        ours.to("cpu", torch.float32)  # leave the fixture as we found it

    assert torch.isfinite(bf16).all(), "bf16 forward produced non-finite logits"
    assert (fp32 - bf16).abs().max().item() < 0.2
