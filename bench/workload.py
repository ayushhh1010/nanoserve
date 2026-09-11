"""Generating request streams that exercise the things being measured.

Three properties matter, and each rules out a simpler choice:

**Poisson arrivals, not a constant rate.** A constant arrival rate produces a
queue that is either always empty or grows without bound, so it never shows the
interesting regime in between. Real traffic is bursty; exponential inter-arrival
times are the standard model, and the bursts are what make queueing visible.

**Zipfian prefix selection.** Requests do not use uniformly random system
prompts -- a handful are used constantly and a long tail almost never. With
uniform prefixes, prefix caching looks useless and hot-spotting never appears,
so both the week-7 cache and the Phase 3 bounded-load router would be measured
against a workload that cannot show what they fix.

**Variable output lengths.** With every request generating the same number of
tokens, head-of-line blocking is invisible: no request can be stuck behind a
much longer one. The whole argument for continuous batching depends on length
variance existing.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from engine.types import Request, SamplingParams


@dataclass
class WorkloadConfig:
    num_requests: int = 200
    #: Requests per second. `None` means all requests arrive at t=0, which
    #: measures raw throughput rather than behaviour under load.
    request_rate: float | None = None

    #: Distinct shared prefixes ("system prompts") in the pool.
    num_prefixes: int = 16
    prefix_len: int = 128
    #: Zipf exponent for prefix popularity. 1.0 is the usual heavy-tailed
    #: default; 0.0 would be uniform.
    zipf_alpha: float = 1.0
    #: Fraction of requests that carry a shared prefix at all.
    prefix_fraction: float = 0.8

    #: Per-request unique continuation, sampled log-normally.
    prompt_len_mean: int = 96
    prompt_len_sigma: float = 0.6

    output_len_mean: int = 96
    output_len_sigma: float = 0.7
    output_len_min: int = 8
    output_len_max: int = 512

    #: Deadline = arrival + slo_seconds. Used for goodput.
    slo_seconds: float | None = 10.0

    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def _lognormal_lengths(
    rng: np.random.Generator, n: int, mean: int, sigma: float, lo: int, hi: int
) -> np.ndarray:
    """Log-normal lengths with the requested median, clipped to [lo, hi].

    Log-normal rather than uniform because real generation lengths are
    right-skewed: most responses are short, a few are very long, and it is
    precisely that tail that causes head-of-line blocking.
    """
    raw = rng.lognormal(mean=np.log(mean), sigma=sigma, size=n)
    return np.clip(raw, lo, hi).astype(int)


def rebase(requests: list[Request], t0: float | None = None) -> list[Request]:
    """Shift relative arrival times and deadlines onto the wall clock.

    `build_workload` emits times relative to zero so a workload is
    reproducible and comparable across runs. Nothing else in the system knows
    that: `AdmissionController` compares deadlines against `perf_counter()`,
    which on a machine that has been up for a while is a number in the
    hundreds of thousands. An un-rebased deadline is therefore ~325,000
    seconds in the past, and admission control rejects 100% of traffic while
    reporting exactly why.

    Arrival and deadline move together, so the SLO interval is preserved.
    """
    t0 = time.perf_counter() if t0 is None else t0
    for r in requests:
        # Both shift by the same offset, so the SLO interval
        # (deadline - arrival) is unchanged.
        if r.deadline is not None:
            r.deadline += t0
        r.arrival_time += t0
    return requests


def build_workload(
    cfg: WorkloadConfig,
    vocab_size: int,
    eos_token_id: int,
    prompt_pool: list[list[int]] | None = None,
) -> list[Request]:
    """Materialise a reproducible list of requests with arrival times set.

    Times are RELATIVE to zero. Call `rebase()` before handing them to an
    engine with admission control -- see that function for why.

    `prompt_pool` supplies real tokenised text; without it, token ids are drawn
    uniformly. Random ids are fine for throughput and memory measurements --
    the model does the same work either way -- but they cannot exercise prefix
    caching realistically, so the real pool is used wherever it exists.
    """
    rng = np.random.default_rng(cfg.seed)

    # -- shared prefix pool, drawn Zipfian ---------------------------------
    prefixes = [
        _take(rng, prompt_pool, cfg.prefix_len, vocab_size, eos_token_id)
        for _ in range(cfg.num_prefixes)
    ]
    ranks = np.arange(1, cfg.num_prefixes + 1)
    weights = ranks.astype(float) ** (-cfg.zipf_alpha)
    weights /= weights.sum()

    prompt_lens = _lognormal_lengths(
        rng, cfg.num_requests, cfg.prompt_len_mean, cfg.prompt_len_sigma, 4, 512
    )
    output_lens = _lognormal_lengths(
        rng, cfg.num_requests, cfg.output_len_mean, cfg.output_len_sigma,
        cfg.output_len_min, cfg.output_len_max,
    )

    # -- arrival schedule ---------------------------------------------------
    if cfg.request_rate is None:
        arrivals = np.zeros(cfg.num_requests)
    else:
        gaps = rng.exponential(1.0 / cfg.request_rate, cfg.num_requests)
        arrivals = np.cumsum(gaps)

    requests: list[Request] = []
    for i in range(cfg.num_requests):
        tokens: list[int] = []
        if rng.random() < cfg.prefix_fraction:
            tokens = list(prefixes[rng.choice(cfg.num_prefixes, p=weights)])
        tokens += _take(rng, prompt_pool, int(prompt_lens[i]), vocab_size, eos_token_id)

        requests.append(
            Request(
                prompt_token_ids=tokens,
                params=SamplingParams(
                    max_tokens=int(output_lens[i]),
                    temperature=0.0,
                    # Fixed-length generation. Whether the model happens to emit
                    # EOS early is a property of the weights, not of the
                    # scheduler, and letting it vary adds noise to every
                    # engine-vs-engine comparison.
                    ignore_eos=True,
                ),
                arrival_time=float(arrivals[i]),  # relative; the runner rebases it
                deadline=None if cfg.slo_seconds is None else float(arrivals[i]) + cfg.slo_seconds,
            )
        )
    return requests


def _take(
    rng: np.random.Generator,
    pool: list[list[int]] | None,
    n: int,
    vocab_size: int,
    eos_token_id: int,
) -> list[int]:
    """n tokens of real text if a pool was supplied, else uniform random ids."""
    if not pool:
        # Avoid EOS in synthetic prompts: an EOS mid-prompt is a different
        # thing to model than an EOS the model chose to emit.
        ids = rng.integers(0, vocab_size - 1, n)
        return [int(i) + (1 if int(i) >= eos_token_id else 0) for i in ids]

    doc = pool[int(rng.integers(0, len(pool)))]
    if len(doc) <= n:
        # Cycle the document rather than pad: padding would make the prompt
        # partly synthetic and change the attention pattern.
        return (doc * (n // len(doc) + 1))[:n]
    start = int(rng.integers(0, len(doc) - n))
    return doc[start : start + n]


def load_prompt_pool(path: Path, tokenizer, max_docs: int = 400) -> list[list[int]]:
    """Tokenise the first `max_docs` documents of a corpus file."""
    from model.tokenizer import END_OF_TEXT

    text = path.read_text(encoding="utf-8", errors="replace")
    docs = [d.strip() for d in text.split(END_OF_TEXT) if d.strip()][:max_docs]
    return [tokenizer.encode(d, allowed_special=False) for d in docs]


def summarise(requests: list[Request]) -> dict:
    """Descriptive stats, so a results file records what was actually run."""
    prompts = np.array([r.prompt_len for r in requests])
    outputs = np.array([r.params.max_tokens for r in requests])
    arrivals = np.array([r.arrival_time for r in requests])
    span = float(arrivals.max() - arrivals.min())
    return {
        "num_requests": len(requests),
        "prompt_tokens": int(prompts.sum()),
        "output_tokens": int(outputs.sum()),
        "prompt_len": {
            "mean": float(prompts.mean()), "p50": float(np.percentile(prompts, 50)),
            "p99": float(np.percentile(prompts, 99)), "max": int(prompts.max()),
        },
        "output_len": {
            "mean": float(outputs.mean()), "p50": float(np.percentile(outputs, 50)),
            "p99": float(np.percentile(outputs, 99)), "max": int(outputs.max()),
        },
        "arrival_span_s": span,
        "effective_rate": len(requests) / span if span > 0 else None,
    }


def save(cfg: WorkloadConfig, requests: list[Request], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"config": cfg.to_dict(), "summary": summarise(requests)}, indent=2) + "\n",
        encoding="utf-8",
    )
