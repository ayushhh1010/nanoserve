"""Resumable training loop.

Resumability is built in from the first commit rather than bolted on, because
the failure it protects against is not hypothetical: a 5-8 hour run on a laptop
GPU will meet a thermal event, a Windows update, or a closed lid. Bolting it on
later means discovering at hour four that the RNG state was never saved and the
resumed run silently trains on a different data order than it would have.

What a checkpoint has to contain is more than most implementations save:

  * model weights                -- obvious
  * optimizer state              -- AdamW's two moment buffers; without them a
                                    resumed run restarts momentum from zero and
                                    takes hundreds of steps to recover
  * step counter                 -- so the LR schedule continues rather than
                                    restarting its warmup
  * torch + CUDA RNG state       -- dropout and any sampling stay on the same
                                    trajectory
  * the data sampler's generator -- so the resumed run sees the batches it
                                    would have seen, not a fresh random stream

tests/test_resume.py asserts the whole thing end to end: training N steps, then
checkpointing and training N more, produces bit-identical weights to training
2N steps straight through. That is the only test that actually proves it.

No GradScaler. The RTX 3050 is Ampere, so bf16 is native and has fp32's
exponent range -- the loss-scale collapse the PRD warns about is a fp16-on-T4
problem that does not exist here.
"""

from __future__ import annotations

import contextlib
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch import nn

from model.config import NANO_27M, NanoConfig
from model.train.data import TokenDataset
from model.transformer import NanoForCausalLM


@dataclass
class TrainConfig:
    # -- data ---------------------------------------------------------------
    train_bin: Path = Path("data/train.bin")
    val_bin: Path = Path("data/val.bin")
    seq_len: int = 1024

    # -- batching -----------------------------------------------------------
    # 8 x 4 x 1024 = 32,768 tokens/step. micro_batch 8 is the measured optimum
    # on 4 GB: see scripts/bench_train_step.py. Going higher does not OOM on
    # Windows, it pages to system RAM and runs up to 12x slower.
    micro_batch: int = 8
    grad_accum: int = 4

    # -- optimiser ----------------------------------------------------------
    lr: float = 1e-3
    min_lr: float = 1e-4
    warmup_steps: int = 500
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0

    # -- schedule -----------------------------------------------------------
    # 536,174,127 tokens / 32,768 per step = 16,364 steps for one epoch.
    max_steps: int = 16_364

    # -- evaluation and checkpointing ---------------------------------------
    eval_every: int = 500
    eval_batches: int = 40
    log_every: int = 10
    ckpt_every: int = 1000
    keep_last: int = 3
    out_dir: Path = Path("checkpoints")

    # -- misc ---------------------------------------------------------------
    # torch.compile fuses the ~700 tiny kernels an eager step launches into a
    # handful of generated ones. Measured 1.61x here (29,400 -> 47,420 tok/s),
    # for a one-time ~2 min compile. Needs triton; on Windows that is the
    # separate `triton-windows` package.
    compile: bool = True
    seed: int = 1337
    device: str = "cuda"
    dtype: str = "bfloat16"
    model: NanoConfig = field(default_factory=lambda: NANO_27M)

    @property
    def tokens_per_step(self) -> int:
        return self.micro_batch * self.grad_accum * self.seq_len

    def to_dict(self) -> dict:
        d = asdict(self)
        d["train_bin"] = str(self.train_bin)
        d["val_bin"] = str(self.val_bin)
        d["out_dir"] = str(self.out_dir)
        return d


def cosine_lr(step: int, cfg: TrainConfig) -> float:
    """Linear warmup, then cosine decay from `lr` to `min_lr`.

    Warmup exists because AdamW's second-moment estimate is meaningless for the
    first few dozen steps: dividing by the square root of a near-zero running
    average produces enormous effective step sizes exactly when the weights are
    least able to absorb them.
    """
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    if step >= cfg.max_steps:
        return cfg.min_lr
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1.0 + math.cos(math.pi * progress))


def build_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.AdamW:
    """AdamW with weight decay on matrices only.

    Decaying a LayerNorm gain or a bias pulls it toward zero, which is not a
    regulariser -- it is a change to the function the layer computes. Only
    parameters with 2+ dimensions (the projections and the embedding table) are
    decayed, which is the standard split and what every reference
    implementation does.
    """
    decay, no_decay = [], []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        (decay if param.dim() >= 2 else no_decay).append(param)

    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.lr,
        betas=cfg.betas,
        fused=torch.cuda.is_available(),
    )


class Trainer:
    def __init__(self, cfg: TrainConfig) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.dtype = getattr(torch, cfg.dtype)

        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)
            # TF32 for fp32 matmuls. The master weights and the optimizer stay
            # fp32; this only affects the internal accumulation precision of
            # matmuls that autocast does not already run in bf16.
            torch.set_float32_matmul_precision("high")

        self.model = NanoForCausalLM(cfg.model).to(self.device)
        # `self.model` stays the uncompiled module and is the only thing ever
        # saved or loaded: torch.compile returns a wrapper whose state_dict
        # keys are prefixed with "_orig_mod.", which would make every
        # checkpoint incompatible with an uncompiled run (and with the HF
        # export in week 3). `self.fwd` is what actually runs.
        self.fwd = torch.compile(self.model) if cfg.compile else self.model
        self.opt = build_optimizer(self.model, cfg)

        self.train_data = TokenDataset(cfg.train_bin, cfg.seq_len)
        self.val_data = TokenDataset(cfg.val_bin, cfg.seq_len)

        # A dedicated generator for batch sampling, so its state can be saved
        # and restored independently of any other use of the global RNG.
        self.data_gen = torch.Generator().manual_seed(cfg.seed)

        self.step = 0
        self.best_val = float("inf")
        self.tokens_seen = 0

        cfg.out_dir.mkdir(parents=True, exist_ok=True)

    def _autocast(self):
        """Mixed precision, unless fp32 was asked for explicitly.

        autocast has no fp32 mode -- it only accepts bf16 or fp16 -- so a
        request for fp32 means running without it. Tests use this to get
        bit-exact determinism, which mixed precision cannot promise.
        """
        if self.dtype is torch.float32:
            return contextlib.nullcontext()
        return torch.autocast(self.device.type, dtype=self.dtype)

    # -- one optimisation step ---------------------------------------------

    def train_step(self) -> tuple[torch.Tensor, torch.Tensor]:
        """One optimiser step over `grad_accum` micro-batches.

        Returns loss and grad-norm as *device tensors*, deliberately. Calling
        `.item()` on them reads a GPU value on the host, which drains the CUDA
        queue; doing it once per micro-batch measured 36 ms each, 145 ms per
        step, ~8% of step time spent waiting for a number that is only used
        every `log_every` steps. The caller syncs when it actually logs.
        """
        cfg = self.cfg
        lr = cosine_lr(self.step, cfg)
        for group in self.opt.param_groups:
            group["lr"] = lr

        self.model.train()
        total_loss = torch.zeros((), device=self.device)

        for _ in range(cfg.grad_accum):
            tokens = self.train_data.batch(cfg.micro_batch, self.device, self.data_gen)
            with self._autocast():
                _, loss = self.fwd(tokens, labels=tokens)
            # Scale so the accumulated gradient is the mean over the full
            # batch, not the sum over micro-batches.
            (loss / cfg.grad_accum).backward()
            total_loss += loss.detach() / cfg.grad_accum

        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)

        self.step += 1
        self.tokens_seen += cfg.tokens_per_step
        return total_loss, grad_norm

    @torch.no_grad()
    def evaluate(self) -> float:
        """Mean validation loss over a fixed set of batches.

        A dedicated generator seeded identically on every call means the same
        windows are scored each time, so successive validation numbers differ
        because the model changed rather than because the sample did.
        """
        self.model.eval()
        gen = torch.Generator().manual_seed(0)
        total = 0.0
        for _ in range(self.cfg.eval_batches):
            tokens = self.val_data.batch(self.cfg.micro_batch, self.device, gen)
            with self._autocast():
                _, loss = self.fwd(tokens, labels=tokens)
            total += loss.item()
        self.model.train()
        return total / self.cfg.eval_batches

    # -- checkpointing ------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "step": self.step,
            "tokens_seen": self.tokens_seen,
            "best_val": self.best_val,
            "model": self.model.state_dict(),
            "optimizer": self.opt.state_dict(),
            "data_generator": self.data_gen.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "config": self.cfg.to_dict(),
        }

    def load_state_dict(self, ckpt: dict) -> None:
        self.step = ckpt["step"]
        self.tokens_seen = ckpt["tokens_seen"]
        self.best_val = ckpt["best_val"]
        self.model.load_state_dict(ckpt["model"])
        self.opt.load_state_dict(ckpt["optimizer"])
        self.data_gen.set_state(ckpt["data_generator"])
        torch.set_rng_state(ckpt["torch_rng"])
        if ckpt["cuda_rng"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(ckpt["cuda_rng"])

    def save(self, name: str) -> Path:
        path = self.cfg.out_dir / name
        tmp = path.with_suffix(".tmp")
        # Write then rename: a checkpoint interrupted mid-write is the one
        # thing worse than no checkpoint, because it looks resumable.
        torch.save(self.state_dict(), tmp)
        tmp.replace(path)
        return path

    def prune_checkpoints(self) -> None:
        """Keep the most recent `keep_last` step checkpoints. Never touch best."""
        ckpts = sorted(
            self.cfg.out_dir.glob("step_*.pt"),
            key=lambda p: int(p.stem.split("_")[1]),
        )
        for old in ckpts[: -self.cfg.keep_last] if self.cfg.keep_last else []:
            old.unlink()

    @classmethod
    def resume(cls, path: Path, cfg: TrainConfig | None = None) -> Trainer:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if cfg is None:
            saved = dict(ckpt["config"])
            saved["train_bin"] = Path(saved["train_bin"])
            saved["val_bin"] = Path(saved["val_bin"])
            saved["out_dir"] = Path(saved["out_dir"])
            saved["model"] = NanoConfig(**saved["model"])
            saved["betas"] = tuple(saved["betas"])
            cfg = TrainConfig(**saved)
        trainer = cls(cfg)
        trainer.load_state_dict(ckpt)
        return trainer

    @staticmethod
    def latest_checkpoint(out_dir: Path) -> Path | None:
        ckpts = list(out_dir.glob("step_*.pt"))
        if not ckpts:
            return None
        return max(ckpts, key=lambda p: int(p.stem.split("_")[1]))

    # -- the loop -----------------------------------------------------------

    def fit(self) -> None:
        cfg = self.cfg
        log_path = cfg.out_dir / "log.jsonl"
        start_step = self.step
        t0 = time.perf_counter()
        window = time.perf_counter()

        print(
            f"training {self.model.num_parameters():,} params  "
            f"{cfg.tokens_per_step:,} tokens/step  "
            f"{cfg.max_steps:,} steps  "
            f"{cfg.max_steps * cfg.tokens_per_step / 1e6:,.0f}M tokens",
            flush=True,
        )
        if start_step:
            print(f"resuming from step {start_step:,}", flush=True)

        while self.step < cfg.max_steps:
            loss_t, grad_norm_t = self.train_step()

            if self.step % cfg.log_every == 0:
                # The only host sync in the loop, and only on logging steps.
                loss, grad_norm = loss_t.item(), grad_norm_t.item()
                elapsed = time.perf_counter() - window
                window = time.perf_counter()
                tok_s = cfg.log_every * cfg.tokens_per_step / elapsed
                remaining = (cfg.max_steps - self.step) * cfg.tokens_per_step / tok_s
                lr = self.opt.param_groups[0]["lr"]
                print(
                    f"step {self.step:>6,}/{cfg.max_steps:,}  "
                    f"loss {loss:6.4f}  lr {lr:.2e}  gnorm {grad_norm:5.2f}  "
                    f"{tok_s:>7,.0f} tok/s  eta {remaining / 3600:4.1f}h",
                    flush=True,
                )
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "step": self.step,
                                "loss": round(loss, 5),
                                "lr": lr,
                                "grad_norm": round(grad_norm, 4),
                                "tokens": self.tokens_seen,
                                "tok_s": round(tok_s),
                                "wall_s": round(time.perf_counter() - t0, 1),
                            }
                        )
                        + "\n"
                    )

            if self.step % cfg.eval_every == 0:
                val = self.evaluate()
                ppl = math.exp(min(val, 20))
                marker = ""
                if val < self.best_val:
                    self.best_val = val
                    self.save("best.pt")
                    marker = "  <- best"
                print(
                    f"  eval @ {self.step:,}: val {val:.4f}  ppl {ppl:7.2f}{marker}",
                    flush=True,
                )
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(
                        json.dumps({"step": self.step, "val_loss": round(val, 5)}) + "\n"
                    )

            if self.step % cfg.ckpt_every == 0:
                self.save(f"step_{self.step:06d}.pt")
                self.prune_checkpoints()

        self.save(f"step_{self.step:06d}.pt")
        val = self.evaluate()
        # The final evaluation has to be able to win. Without this, a run whose
        # last steps improve on the previous eval leaves best.pt pointing at a
        # strictly worse model -- which is exactly what happened on run1
        # (best.pt val 1.1518 at step 16,000 vs 1.1510 at the end).
        if val < self.best_val:
            self.best_val = val
            self.save("best.pt")
        print(
            f"\ndone: {self.step:,} steps, {self.tokens_seen / 1e6:,.0f}M tokens, "
            f"{(time.perf_counter() - t0) / 3600:.2f}h, final val {val:.4f} "
            f"(ppl {math.exp(min(val, 20)):.2f})",
            flush=True,
        )
