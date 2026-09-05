"""Model configuration for the Nanoserve transformer.

Deliberately a plain dataclass, not a HF PretrainedConfig: the whole point of
Phase 1 is that nothing here is inherited from a library. Field *names* mirror
HuggingFace's LlamaConfig, though, so that (a) the parity test in
tests/test_parity.py can build an equivalent reference model with one call and
(b) exporting weights to the Hub later is a rename-free operation.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class NanoConfig:
    # --- shape -------------------------------------------------------------
    vocab_size: int = 8192
    hidden_size: int = 512  # d_model
    num_hidden_layers: int = 8
    num_attention_heads: int = 8
    num_key_value_heads: int = 2  # GQA: 4 query heads share one KV head
    intermediate_size: int = 1408  # SwiGLU d_ff
    max_position_embeddings: int = 1024

    # --- numerics ----------------------------------------------------------
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    initializer_range: float = 0.02

    # --- wiring ------------------------------------------------------------
    # Tied at this scale on purpose: the untied lm_head is 4.2M params (15% of the
    # model) for a vocab of 8192, and tying is the standard choice for small
    # models. Untied would be 30.9M; tied is 26.7M, the ~27M the plan budgets for.
    tie_word_embeddings: bool = True
    attention_bias: bool = False
    mlp_bias: bool = False

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size={self.hidden_size} not divisible by "
                f"num_attention_heads={self.num_attention_heads}"
            )
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads={self.num_attention_heads} not divisible by "
                f"num_key_value_heads={self.num_key_value_heads}"
            )
        if self.head_dim % 2 != 0:
            raise ValueError(f"head_dim={self.head_dim} must be even for RoPE")

    # --- derived -----------------------------------------------------------
    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def n_rep(self) -> int:
        """How many query heads share each KV head."""
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def kv_bytes_per_token(self) -> int:
        """Bytes of KV cache one token occupies across all layers, at fp16.

        This is the number Phase 2's block manager is sized against, so it lives
        on the config rather than in the engine. GQA is what makes it small:
        with n_kv_heads=2 it is 4x smaller than with full multi-head attention.
        """
        per_layer = 2 * self.num_key_value_heads * self.head_dim * 2  # K and V, 2 bytes
        return per_layer * self.num_hidden_layers

    # --- (de)serialisation --------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> NanoConfig:
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


#: The configuration this project actually trains. ~27M parameters.
NANO_27M = NanoConfig()


#: A tiny config for unit tests, so tests stay fast and CPU-only.
TINY = NanoConfig(
    vocab_size=256,
    hidden_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    intermediate_size=176,
    max_position_embeddings=128,
)


def count_parameters(cfg: NanoConfig) -> dict[str, int]:
    """Analytic parameter count -- no torch needed, so it can run anywhere.

    Cross-checked against the real module in tests/test_transformer.py; a
    mismatch there means the model does not have the shape this claims.
    """
    h, ffn, v = cfg.hidden_size, cfg.intermediate_size, cfg.vocab_size
    kv_h = cfg.num_key_value_heads * cfg.head_dim

    embed = v * h
    attn = (h * h) + 2 * (h * kv_h) + (h * h)  # q, k, v, o
    mlp = 3 * (h * ffn)  # gate, up, down
    norms = 2 * h  # input_layernorm + post_attention_layernorm
    per_layer = attn + mlp + norms

    body = cfg.num_hidden_layers * per_layer + h  # + final norm
    head = 0 if cfg.tie_word_embeddings else v * h

    return {
        "embedding": embed,
        "per_layer": per_layer,
        "layers": cfg.num_hidden_layers * per_layer,
        "final_norm": h,
        "lm_head": head,
        "total": embed + body + head,
        "non_embedding": body,
    }


if __name__ == "__main__":
    cfg = NANO_27M
    counts = count_parameters(cfg)
    print(json.dumps(cfg.to_dict(), indent=2))
    print()
    for k, val in counts.items():
        print(f"{k:>16}: {val:>12,}")
    print(f"{'kv/token @fp16':>16}: {cfg.kv_bytes_per_token:>12,} bytes")
