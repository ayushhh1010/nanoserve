"""Autoregressive sampling from a trained checkpoint.

Deliberately simple and single-sequence: this exists to look at what the model
learned and to be the correctness reference that Phase 2's engine is checked
against. The scheduler, paged cache and continuous batching all have to
reproduce this function's output token for token under greedy decoding, so it
stays plain on purpose.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
import torch.nn.functional as F

from model.config import NanoConfig
from model.tokenizer import BPETokenizer
from model.transformer import DynamicCache, NanoForCausalLM


@dataclass
class SamplingParams:
    max_new_tokens: int = 256
    temperature: float = 0.8
    top_k: int | None = 50
    top_p: float | None = 0.95
    #: Stop as soon as the model emits EOS. Off means run to max_new_tokens,
    #: which is what a fixed-length benchmark wants.
    stop_at_eos: bool = True
    seed: int | None = None

    @property
    def greedy(self) -> bool:
        return self.temperature == 0.0


def load_model(
    checkpoint: str | Path,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[NanoForCausalLM, dict]:
    """Load weights saved by the trainer.

    Accepts a full training checkpoint (weights plus optimizer state) or a bare
    state dict, so an exported inference-only file works too.
    """
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)

    if "model" in ckpt and "config" in ckpt:
        cfg = NanoConfig(**ckpt["config"]["model"])
        weights = ckpt["model"]
        meta = {k: ckpt.get(k) for k in ("step", "tokens_seen", "best_val")}
    else:
        cfg = NanoConfig(**ckpt["config"])
        weights = ckpt["state_dict"]
        meta = ckpt.get("meta", {})

    model = NanoForCausalLM(cfg)
    model.load_state_dict(weights)
    return model.to(device=device, dtype=dtype).eval(), meta


def _filter_logits(logits: torch.Tensor, params: SamplingParams) -> torch.Tensor:
    """Apply temperature, then top-k, then top-p, in that order.

    Order matters: top-p works on the softmax of what survives top-k, so
    swapping them changes which tokens are reachable.
    """
    logits = logits / params.temperature

    if params.top_k:
        k = min(params.top_k, logits.size(-1))
        threshold = torch.topk(logits, k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < threshold, float("-inf"))

    if params.top_p is not None and params.top_p < 1.0:
        ordered, index = torch.sort(logits, descending=True, dim=-1)
        cumulative = torch.softmax(ordered, dim=-1).cumsum(dim=-1)
        # Keep the first token that crosses the threshold, so top_p is never
        # able to leave the candidate set empty.
        drop = cumulative - torch.softmax(ordered, dim=-1) > params.top_p
        logits = logits.masked_fill(drop.scatter(-1, index, drop), float("-inf"))

    return logits


@torch.inference_mode()
def generate_stream(
    model: NanoForCausalLM,
    tokenizer: BPETokenizer,
    prompt: str,
    params: SamplingParams | None = None,
    device: str = "cuda",
) -> Iterator[str]:
    """Yield decoded text as it is produced.

    Decoding is done on the running byte buffer rather than per token, because
    a multi-byte character is routinely split across two tokens and per-token
    decoding would emit replacement characters mid-word.
    """
    params = params or SamplingParams()
    gen = None
    if params.seed is not None:
        gen = torch.Generator(device=device).manual_seed(params.seed)

    ids = tokenizer.encode(prompt, allowed_special=False)
    if not ids:
        ids = [tokenizer.eos_id]
    tokens = torch.tensor([ids], device=device)

    cache = DynamicCache()
    logits, _ = model(tokens, cache=cache)

    produced: list[int] = []
    emitted = 0
    for _ in range(params.max_new_tokens):
        step = logits[:, -1]
        if params.greedy:
            nxt = step.argmax(-1, keepdim=True)
        else:
            probs = F.softmax(_filter_logits(step, params), dim=-1)
            nxt = torch.multinomial(probs, num_samples=1, generator=gen)

        token = int(nxt)
        if params.stop_at_eos and token == tokenizer.eos_id:
            break

        produced.append(token)
        text = tokenizer.decode(produced)
        # Hold back a trailing replacement char: it means the last token is
        # half of a multi-byte character and the rest is still coming.
        if text.endswith("�"):
            continue
        if len(text) > emitted:
            yield text[emitted:]
            emitted = len(text)

        logits, _ = model(nxt, cache=cache)


def generate(
    model: NanoForCausalLM,
    tokenizer: BPETokenizer,
    prompt: str,
    params: SamplingParams | None = None,
    device: str = "cuda",
) -> str:
    return "".join(generate_stream(model, tokenizer, prompt, params, device))
