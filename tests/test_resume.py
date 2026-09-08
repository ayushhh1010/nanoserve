"""Does an interrupted run actually resume where it left off?

The PRD rates "Kaggle session cap kills a training run" as high likelihood,
and running locally swaps that for thermal events and Windows updates. Either
way, resumability is load-bearing, and it is the kind of feature that appears
to work while being subtly wrong: a run that reloads weights but not optimizer
moments still trains, still shows a falling loss, and is simply worse than it
should have been. Nothing surfaces it except an explicit test.

`test_resume_is_bit_identical_to_uninterrupted_training` is the one that
matters. Everything else in this file localises the failure when it breaks.

Everything runs on CPU in fp32: mixed precision does not promise bit-exact
reproducibility, and this file is asking for exactly that.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from model.config import NanoConfig
from model.train.train import TrainConfig, Trainer, build_optimizer, cosine_lr

TINY_MODEL = NanoConfig(
    vocab_size=256,
    hidden_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    intermediate_size=176,
    max_position_embeddings=128,
)


@pytest.fixture
def cfg(tmp_path) -> TrainConfig:
    rng = np.random.default_rng(0)
    for name, n in (("train.bin", 200_000), ("val.bin", 20_000)):
        rng.integers(0, TINY_MODEL.vocab_size, n, dtype=np.uint16).tofile(tmp_path / name)

    return TrainConfig(
        train_bin=tmp_path / "train.bin",
        val_bin=tmp_path / "val.bin",
        out_dir=tmp_path / "ckpt",
        seq_len=32,
        micro_batch=4,
        grad_accum=2,
        max_steps=20,
        warmup_steps=4,
        eval_every=1000,
        ckpt_every=1000,
        log_every=1000,
        eval_batches=2,
        device="cpu",
        dtype="float32",
        compile=False,  # tests want fast start-up and exact fp32, not fused kernels
        model=TINY_MODEL,
    )


def weights(trainer: Trainer) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in trainer.model.state_dict().items()}


def assert_identical(a: dict, b: dict, what: str) -> None:
    assert a.keys() == b.keys()
    for key in a:
        if not torch.equal(a[key], b[key]):
            delta = (a[key].float() - b[key].float()).abs().max().item()
            pytest.fail(f"{what}: {key} differs, max |delta| = {delta:.3e}")


# ---------------------------------------------------------------------------
# The test that proves it
# ---------------------------------------------------------------------------


def test_resume_is_bit_identical_to_uninterrupted_training(cfg):
    """10 steps + checkpoint + 10 steps == 20 steps straight through.

    Bit-identical, not approximately equal. Any state left out of the
    checkpoint -- optimizer moments, the step counter feeding the LR schedule,
    the data sampler's position -- shows up here as a real difference.
    """
    straight = Trainer(cfg)
    for _ in range(20):
        straight.train_step()

    interrupted = Trainer(cfg)
    for _ in range(10):
        interrupted.train_step()
    path = interrupted.save("step_000010.pt")
    del interrupted

    resumed = Trainer.resume(path, cfg)
    assert resumed.step == 10
    for _ in range(10):
        resumed.train_step()

    assert resumed.step == straight.step == 20
    assert resumed.tokens_seen == straight.tokens_seen
    assert_identical(weights(straight), weights(resumed), "weights after resume")


def test_resume_without_optimizer_state_would_diverge(cfg):
    """Confirms the previous test is actually sensitive to what it claims.

    A test that passes for the wrong reason is worse than no test. This drops
    the optimizer moments on reload -- the single most commonly forgotten piece
    of checkpoint state -- and requires that the result differs.
    """
    straight = Trainer(cfg)
    for _ in range(20):
        straight.train_step()

    partial = Trainer(cfg)
    for _ in range(10):
        partial.train_step()
    ckpt = partial.state_dict()

    fresh = Trainer(cfg)
    fresh.step = ckpt["step"]
    fresh.model.load_state_dict(ckpt["model"])
    fresh.data_gen.set_state(ckpt["data_generator"])
    # deliberately NOT restoring fresh.opt
    for _ in range(10):
        fresh.train_step()

    same = all(
        torch.equal(a, b)
        for a, b in zip(weights(straight).values(), weights(fresh).values())
    )
    assert not same, "dropping AdamW moments changed nothing -- the test is not sensitive"


# ---------------------------------------------------------------------------
# The individual pieces
# ---------------------------------------------------------------------------


def test_data_order_continues_across_resume(cfg):
    """The resumed run must see the batches it would have seen."""
    a = Trainer(cfg)
    for _ in range(5):
        a.train_step()
    expected = a.train_data.batch(cfg.micro_batch, "cpu", a.data_gen)

    b = Trainer(cfg)
    for _ in range(5):
        b.train_step()
    path = b.save("step_000005.pt")

    resumed = Trainer.resume(path, cfg)
    assert torch.equal(resumed.train_data.batch(cfg.micro_batch, "cpu", resumed.data_gen), expected)


def test_optimizer_moments_are_restored(cfg):
    trainer = Trainer(cfg)
    for _ in range(6):
        trainer.train_step()
    path = trainer.save("step_000006.pt")

    before = trainer.opt.state_dict()["state"]
    after = Trainer.resume(path, cfg).opt.state_dict()["state"]

    assert before.keys() == after.keys()
    assert len(before) > 0, "AdamW has no state -- nothing was actually optimised"
    for key in before:
        for field in ("exp_avg", "exp_avg_sq", "step"):
            assert torch.equal(
                torch.as_tensor(before[key][field]), torch.as_tensor(after[key][field])
            ), f"{field} not restored for param {key}"


def test_step_counter_keeps_the_lr_schedule_going(cfg):
    """A resumed run must not restart its warmup."""
    trainer = Trainer(cfg)
    for _ in range(6):
        trainer.train_step()
    path = trainer.save("step_000006.pt")

    resumed = Trainer.resume(path, cfg)
    assert resumed.step == 6
    assert cosine_lr(resumed.step, cfg) == cosine_lr(6, cfg)
    assert cosine_lr(resumed.step, cfg) != cosine_lr(0, cfg)


def test_config_roundtrips_through_the_checkpoint(cfg):
    """Resuming without passing a config must reconstruct the original."""
    trainer = Trainer(cfg)
    trainer.train_step()
    path = trainer.save("step_000001.pt")

    resumed = Trainer.resume(path)  # no cfg supplied
    assert resumed.cfg.seq_len == cfg.seq_len
    assert resumed.cfg.micro_batch == cfg.micro_batch
    assert resumed.cfg.betas == cfg.betas
    assert resumed.cfg.model == cfg.model
    assert resumed.cfg.max_steps == cfg.max_steps


# ---------------------------------------------------------------------------
# Checkpoint file handling
# ---------------------------------------------------------------------------


def test_checkpoint_write_is_atomic(cfg):
    """A half-written checkpoint is worse than none: it looks resumable."""
    trainer = Trainer(cfg)
    trainer.train_step()
    path = trainer.save("step_000001.pt")

    assert path.exists()
    assert not path.with_suffix(".tmp").exists(), "temp file left behind"
    assert list(cfg.out_dir.glob("*.tmp")) == []


def test_pruning_keeps_recent_and_never_deletes_best(cfg):
    trainer = Trainer(cfg)
    trainer.save("best.pt")
    for step in (1000, 2000, 3000, 4000, 5000):
        trainer.step = step
        trainer.save(f"step_{step:06d}.pt")
        trainer.prune_checkpoints()

    kept = sorted(p.name for p in cfg.out_dir.glob("step_*.pt"))
    assert kept == ["step_003000.pt", "step_004000.pt", "step_005000.pt"]
    assert (cfg.out_dir / "best.pt").exists()


def test_latest_checkpoint_sorts_numerically_not_lexically(cfg):
    """step_000009 vs step_000010: string ordering gets this wrong."""
    trainer = Trainer(cfg)
    for step in (9, 10, 100):
        trainer.step = step
        trainer.save(f"step_{step:06d}.pt")

    latest = Trainer.latest_checkpoint(cfg.out_dir)
    assert latest is not None and latest.name == "step_000100.pt"
    assert Trainer.latest_checkpoint(cfg.out_dir / "nonexistent") is None


# ---------------------------------------------------------------------------
# Schedule and optimiser construction
# ---------------------------------------------------------------------------


def test_lr_warms_up_then_decays_to_the_floor(cfg):
    warm = [cosine_lr(s, cfg) for s in range(cfg.warmup_steps)]
    assert warm == sorted(warm) and warm[0] < warm[-1]
    assert warm[-1] == pytest.approx(cfg.lr)

    after = [cosine_lr(s, cfg) for s in range(cfg.warmup_steps, cfg.max_steps)]
    assert after == sorted(after, reverse=True)
    assert cosine_lr(cfg.max_steps, cfg) == pytest.approx(cfg.min_lr)
    assert cosine_lr(cfg.max_steps * 10, cfg) == pytest.approx(cfg.min_lr)


def test_lr_never_leaves_its_bounds(cfg):
    for step in range(0, cfg.max_steps * 2):
        assert cfg.min_lr * 0.99 <= cosine_lr(step, cfg) <= cfg.lr * 1.01


def test_weight_decay_applies_to_matrices_only(cfg):
    """Decaying a norm gain changes the function; it does not regularise it."""
    model = Trainer(cfg).model
    opt = build_optimizer(model, cfg)

    decayed, undecayed = opt.param_groups
    assert decayed["weight_decay"] == cfg.weight_decay
    assert undecayed["weight_decay"] == 0.0
    assert all(p.dim() >= 2 for p in decayed["params"])
    assert all(p.dim() == 1 for p in undecayed["params"])

    norm_params = sum(1 for n, _ in model.named_parameters() if "layernorm" in n or n.endswith("norm.weight"))
    assert len(undecayed["params"]) >= norm_params > 0


def test_tokens_per_step_matches_the_measured_configuration():
    assert TrainConfig().tokens_per_step == 32_768
    assert TrainConfig().micro_batch == 8  # the measured optimum on 4 GB
