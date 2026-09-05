"""A decoder-only Llama-style transformer, written by hand.

No AutoModel, no library layers. RMSNorm, RoPE, grouped-query attention and
SwiGLU are all implemented here; the only thing borrowed from torch is
`scaled_dot_product_attention`, which is a fused kernel rather than a model.

Parameter names match HuggingFace's Llama so that tests/test_parity.py can
check this implementation numerically against `LlamaForCausalLM`, and so that
publishing the trained weights to the Hub needs no key remapping.

The `KVCache` protocol in the attention path is the seam Phase 2 replaces with
a paged allocator. Everything above it stays untouched.
"""

from __future__ import annotations

import math
from typing import Protocol

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import NanoConfig

# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """Root-mean-square layer norm: no mean subtraction, no bias.

    Cheaper than LayerNorm and empirically just as good for transformers. The
    reduction runs in fp32 even under autocast -- squaring fp16 activations is
    a reliable way to produce infinities, and this is the single most common
    source of NaNs in a mixed-precision training run.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        x = x.float()
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * rms).to(dtype) * self.weight

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


# ---------------------------------------------------------------------------
# Rotary position embeddings
# ---------------------------------------------------------------------------


class RotaryEmbedding(nn.Module):
    """Precomputed RoPE cos/sin tables.

    Positions are encoded by rotating each 2D slice of the head dimension by an
    angle proportional to the position, with a per-slice frequency. Attention
    scores then depend on relative position (m - n) rather than absolute, which
    is why the same tables work for a cached decode step at position 900 as for
    a prefill token at position 3.

    Uses the half-split pairing (dims i and i + head_dim/2 form a pair) rather
    than interleaved pairing. Both are valid rotations; this one matches the
    reference implementation the parity test compares against.
    """

    def __init__(self, head_dim: int, max_positions: int, base: float = 10000.0) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_tables(max_positions)

    def _build_tables(self, max_positions: int) -> None:
        self.max_positions = max_positions
        device = self.inv_freq.device
        t = torch.arange(max_positions, dtype=torch.float32, device=device)
        freqs = torch.outer(t, self.inv_freq)  # (T, head_dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (T, head_dim)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def ensure_capacity(self, max_position: int) -> None:
        """Grow the tables to cover `max_position` (exclusive). Host-side only.

        This is deliberately NOT done inside `forward`. Deriving the needed
        size from the tensor -- `int(position_ids.max())` -- reads a GPU value
        on the host, which drains the CUDA queue. During decode that happens
        once per token and costs roughly 10x the entire forward pass. Callers
        know their sequence lengths as plain ints already, so the check belongs
        here, where it is free.
        """
        if max_position > self.max_positions:
            self._build_tables(max_position)

    @torch.no_grad()
    def forward(self, position_ids: Tensor) -> tuple[Tensor, Tensor]:
        """position_ids: (B, T) -> cos, sin each (B, T, head_dim).

        Pure indexing, no host sync. Positions must be < `max_positions`; call
        `ensure_capacity` first if that is not already guaranteed.
        """
        return self.cos_cached[position_ids], self.sin_cached[position_ids]


def rotate_half(x: Tensor) -> Tensor:
    """(x1, x2) -> (-x2, x1), the 90-degree rotation of each dimension pair."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rotary(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    """q, k: (B, H, T, D). cos, sin: (B, T, D)."""
    cos = cos.unsqueeze(1)  # (B, 1, T, D), broadcasts over heads
    sin = sin.unsqueeze(1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


# ---------------------------------------------------------------------------
# KV cache seam
# ---------------------------------------------------------------------------


class KVCache(Protocol):
    """What attention needs from a cache, and nothing more.

    Phase 2's paged allocator implements exactly this. Because the model only
    ever sees `update`, swapping a contiguous cache for a block-table one is
    invisible from here.
    """

    def update(self, layer_idx: int, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """Append this step's K/V for `layer_idx`, return the full K/V so far."""
        ...


class DynamicCache:
    """The naive cache: one growing contiguous tensor per layer.

    This is the Phase 2 baseline that gets replaced, and it is here to be
    replaced. It reallocates on every decode step and cannot share a prefix
    between two sequences -- both are reasons a paged cache exists.
    """

    def __init__(self) -> None:
        self.keys: list[Tensor | None] = []
        self.values: list[Tensor | None] = []

    def update(self, layer_idx: int, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        while len(self.keys) <= layer_idx:
            self.keys.append(None)
            self.values.append(None)
        prev_k = self.keys[layer_idx]
        if prev_k is None:
            self.keys[layer_idx], self.values[layer_idx] = k, v
        else:
            self.keys[layer_idx] = torch.cat([prev_k, k], dim=-2)
            self.values[layer_idx] = torch.cat([self.values[layer_idx], v], dim=-2)
        return self.keys[layer_idx], self.values[layer_idx]  # type: ignore[return-value]

    def __len__(self) -> int:
        """Number of tokens cached so far."""
        if not self.keys or self.keys[0] is None:
            return 0
        return self.keys[0].shape[-2]


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


def repeat_kv(x: Tensor, n_rep: int) -> Tensor:
    """(B, H_kv, T, D) -> (B, H_kv * n_rep, T, D) by repeating each KV head.

    This materialises the shared heads so SDPA sees a normal multi-head layout.
    It costs memory in the *activation*, not in the cache -- the cache still
    holds only H_kv heads, which is the entire point of GQA and the reason
    Phase 2's KV budget is 4x smaller than it would otherwise be.
    """
    if n_rep == 1:
        return x
    b, h, t, d = x.shape
    return x[:, :, None].expand(b, h, n_rep, t, d).reshape(b, h * n_rep, t, d)


def offset_causal_mask(q_len: int, kv_len: int, device: torch.device) -> Tensor:
    """Boolean mask for `q_len` queries sitting at the end of `kv_len` keys."""
    q_pos = torch.arange(kv_len - q_len, kv_len, device=device).unsqueeze(1)
    k_pos = torch.arange(kv_len, device=device).unsqueeze(0)
    return k_pos <= q_pos


class Attention(nn.Module):
    """Grouped-query causal self-attention."""

    def __init__(self, cfg: NanoConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.n_heads = cfg.num_attention_heads
        self.n_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.n_rep = cfg.n_rep
        self.scale = 1.0 / math.sqrt(self.head_dim)

        bias = cfg.attention_bias
        self.q_proj = nn.Linear(cfg.hidden_size, self.n_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(cfg.hidden_size, self.n_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(cfg.hidden_size, self.n_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, cfg.hidden_size, bias=bias)

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        cache: KVCache | None = None,
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        b, t, _ = x.shape

        q = self.q_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rotary(q, k, cos, sin)

        if cache is not None:
            k, v = cache.update(self.layer_idx, k, v)

        # Materialise the shared KV heads rather than using SDPA's `enable_gqa`.
        # Counterintuitive, and measured: the fused attention kernels require
        # query and key to have the same head count, so passing mismatched
        # heads with enable_gqa=True silently drops SDPA to its unfused MATH
        # backend. That costs ~20% (9.4 -> 11.3 ms/token here) and ~200 extra
        # kernel launches, which is far more than the copy this would avoid.
        # See scripts/profile_decode.py for the A/B.
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        # Three regimes, and conflating them is a classic silent bug:
        #   prefill (t == kv_len)        -> plain causal mask
        #   decode  (t == 1)             -> attend to everything cached, no mask
        #   chunked prefill (t < kv_len) -> causal mask offset by the cache
        kv_len = k.shape[-2]
        is_causal = False
        if attn_mask is None and t > 1:
            if t == kv_len:
                is_causal = True
            else:
                attn_mask = offset_causal_mask(t, kv_len, x.device)

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=is_causal, scale=self.scale
        )
        out = out.transpose(1, 2).reshape(b, t, self.n_heads * self.head_dim)
        return self.o_proj(out)


# ---------------------------------------------------------------------------
# Feed-forward
# ---------------------------------------------------------------------------


class SwiGLU(nn.Module):
    """SiLU-gated feed-forward: down(silu(gate(x)) * up(x)).

    Three matrices instead of two, so intermediate_size is sized down (~2.75x
    hidden rather than 4x) to keep the parameter count comparable to a plain
    two-matrix MLP.
    """

    def __init__(self, cfg: NanoConfig) -> None:
        super().__init__()
        bias = cfg.mlp_bias
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=bias)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=bias)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Block and model
# ---------------------------------------------------------------------------


class DecoderLayer(nn.Module):
    """Pre-norm residual block: x + attn(norm(x)), then x + mlp(norm(x)).

    Pre-norm (normalise the branch input, not the residual output) is what lets
    a deep stack train without a warmup-sensitive gradient explosion: the
    residual stream itself is never normalised, so gradients reach layer 0
    unattenuated.
    """

    def __init__(self, cfg: NanoConfig, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = Attention(cfg, layer_idx)
        self.mlp = SwiGLU(cfg)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        cache: KVCache | None = None,
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, cache, attn_mask)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class NanoModel(nn.Module):
    """The transformer body: embeddings, layers, final norm. No LM head."""

    def __init__(self, cfg: NanoConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(DecoderLayer(cfg, i) for i in range(cfg.num_hidden_layers))
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.rotary = RotaryEmbedding(cfg.head_dim, cfg.max_position_embeddings, cfg.rope_theta)

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor | None = None,
        cache: KVCache | None = None,
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        b, t = input_ids.shape
        if position_ids is None:
            # `len(cache)` is a shape read, not a device read -- no sync here.
            past = len(cache) if isinstance(cache, DynamicCache) else 0
            self.rotary.ensure_capacity(past + t)
            position_ids = torch.arange(past, past + t, device=input_ids.device).expand(b, t)

        x = self.embed_tokens(input_ids)
        cos, sin = self.rotary(position_ids)
        cos, sin = cos.to(x.dtype), sin.to(x.dtype)

        for layer in self.layers:
            x = layer(x, cos, sin, cache, attn_mask)
        return self.norm(x)


class NanoForCausalLM(nn.Module):
    """Transformer body plus the language-modelling head."""

    def __init__(self, cfg: NanoConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = NanoModel(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.apply(self._init_weights)
        self._scale_residual_projections()
        if cfg.tie_word_embeddings:
            # Tie after init so the shared tensor is initialised exactly once.
            self.lm_head.weight = self.model.embed_tokens.weight

    # --- initialisation ----------------------------------------------------

    def _init_weights(self, module: nn.Module) -> None:
        std = self.cfg.initializer_range
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    def _scale_residual_projections(self) -> None:
        """Shrink the two projections that write into the residual stream.

        Every layer adds to the same stream, so without this the stream's
        variance grows with depth and the late layers train against an already
        saturated signal. Scaling by 1/sqrt(2 * n_layers) -- two residual
        writes per layer -- keeps that variance roughly constant with depth.
        Standard since GPT-2, and free.
        """
        scale = 1.0 / math.sqrt(2 * self.cfg.num_hidden_layers)
        with torch.no_grad():
            for layer in self.model.layers:
                layer.self_attn.o_proj.weight.mul_(scale)
                layer.mlp.down_proj.weight.mul_(scale)

    # --- forward -----------------------------------------------------------

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor | None = None,
        cache: KVCache | None = None,
        attn_mask: Tensor | None = None,
        labels: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        hidden = self.model(input_ids, position_ids, cache, attn_mask)
        logits = self.lm_head(hidden)

        loss = None
        if labels is not None:
            # Position t predicts token t+1.
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)).float(),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return logits, loss

    # --- introspection -----------------------------------------------------

    def num_parameters(self, non_embedding: bool = False) -> int:
        seen: set[int] = set()
        total = 0
        for name, p in self.named_parameters():
            if id(p) in seen:  # tied weights must not be double-counted
                continue
            seen.add(id(p))
            if non_embedding and "embed_tokens" in name:
                continue
            total += p.numel()
        return total
