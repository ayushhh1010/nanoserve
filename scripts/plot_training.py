"""Render the training run from checkpoints/*/log.jsonl.

Three stacked panels sharing one x-axis rather than one panel with two y-scales.
Loss, learning rate and throughput have unrelated units; overlaying them on
twin axes lets the reader infer a relationship from wherever the two curves
happen to cross, which is an artifact of the scaling choice rather than
anything in the data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Validated 2-slot categorical palette (scripts/validate_palette.js, light mode):
# adjacent CVD dE 24.7, normal-vision dE 33.6, both slots >= 3:1 on the surface.
TRAIN = "#2a78d6"
VAL = "#eb6834"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e3e2df"


def read_log(path: Path) -> tuple[dict, dict]:
    train: dict[str, list] = {"step": [], "loss": [], "lr": [], "tok_s": [], "gnorm": []}
    val: dict[str, list] = {"step": [], "loss": []}

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "val_loss" in row:
            val["step"].append(row["step"])
            val["loss"].append(row["val_loss"])
        else:
            train["step"].append(row["step"])
            train["loss"].append(row["loss"])
            train["lr"].append(row["lr"])
            train["tok_s"].append(row["tok_s"])
            train["gnorm"].append(row.get("grad_norm", float("nan")))
    return train, val


def smooth(y: list[float], window: int) -> np.ndarray:
    """Centred moving average, shrinking at the edges so the line spans the axis."""
    a = np.asarray(y, dtype=float)
    if window <= 1 or len(a) < 3:
        return a
    window = min(window, len(a))
    kernel = np.ones(window) / window
    padded = np.pad(a, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def style(ax) -> None:
    """Recessive axes: the data should be the darkest thing in the frame."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", type=Path, default=Path("checkpoints/run1/log.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("writeups/img/training.png"))
    ap.add_argument("--smooth", type=int, default=15)
    # The log records every log_every steps, so its last row is not the last
    # step of the run. The headline numbers come from the run itself.
    ap.add_argument("--steps", type=int, default=16_364)
    ap.add_argument("--tokens", type=int, default=536_174_127)
    ap.add_argument("--hours", type=float, default=3.27)
    ap.add_argument("--final-val", type=float, default=1.1485)
    args = ap.parse_args()

    train, val = read_log(args.log)
    if not train["step"]:
        raise SystemExit(f"no training rows in {args.log}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_loss, ax_lr, ax_tps) = plt.subplots(
        3, 1, figsize=(9, 8), height_ratios=[3, 1, 1], sharex=True
    )
    fig.patch.set_facecolor(SURFACE)

    # -- loss ---------------------------------------------------------------
    style(ax_loss)
    ax_loss.plot(train["step"], train["loss"], color=TRAIN, lw=0.7, alpha=0.25, zorder=2)
    ax_loss.plot(
        train["step"], smooth(train["loss"], args.smooth),
        color=TRAIN, lw=2, zorder=3, label="train",
    )
    ax_loss.plot(
        val["step"], val["loss"],
        color=VAL, lw=2, marker="o", markersize=3.5, zorder=4, label="validation",
    )

    # Log scale: the run spans 9.01 -> 1.15, and on a linear axis everything
    # after step ~1000 is squashed into the bottom tenth of the panel. The
    # interesting part of a loss curve is the long tail, not the initial drop.
    ax_loss.set_yscale("log")

    x0 = min(train["step"])
    ln_v = np.log(8192)
    ax_loss.axhline(ln_v, color=INK_MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
    ax_loss.text(
        x0, ln_v * 1.03, f"ln(V) = {ln_v:.2f} — uniform over 8,192 tokens",
        ha="left", va="bottom", fontsize=8.5, color=INK_MUTED,
    )

    target = 1.8
    ax_loss.axhline(target, color=INK_MUTED, lw=1, ls=(0, (1, 2)), zorder=1)
    # Placed mid-axis: at x0 the label sits right on top of the validation
    # curve's opening descent.
    ax_loss.text(
        x0 + (max(train["step"]) - x0) * 0.42, target * 1.04,
        "acceptance target < 1.8",
        ha="left", va="bottom", fontsize=8.5, color=INK_MUTED,
    )

    # Direct labels on the final values only -- never a number on every point.
    ax_loss.annotate(
        f"{args.final_val:.3f}",
        xy=(val["step"][-1], val["loss"][-1]),
        xytext=(8, 8), textcoords="offset points",
        fontsize=10, color=INK, fontweight="bold",
    )
    ax_loss.set_ylabel("loss (nats/token)", color=INK_MUTED, fontsize=10)
    ax_loss.set_ylim(1.0, 11.5)
    ax_loss.set_yticks([1, 1.5, 2, 3, 5, 9])
    ax_loss.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax_loss.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    # Right of the curve's tail and below the ln(V) line, so it collides with
    # neither. Checked by rendering, not assumed.
    leg = ax_loss.legend(
        loc="center right", frameon=False, fontsize=10, labelcolor=INK, handlelength=1.6
    )
    leg.set_zorder(5)
    # Room at the right edge for the direct label.
    span = max(train["step"]) - x0
    ax_loss.set_xlim(x0 - span * 0.02, max(train["step"]) + span * 0.09)

    # -- learning rate ------------------------------------------------------
    style(ax_lr)
    ax_lr.plot(train["step"], train["lr"], color=TRAIN, lw=2)
    ax_lr.set_ylabel("learning rate", color=INK_MUTED, fontsize=10)
    ax_lr.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax_lr.yaxis.get_offset_text().set_color(INK_MUTED)

    # -- throughput ---------------------------------------------------------
    style(ax_tps)
    ax_tps.plot(train["step"], np.asarray(train["tok_s"]) / 1000, color=TRAIN, lw=1.2, alpha=0.35)
    ax_tps.plot(train["step"], smooth([t / 1000 for t in train["tok_s"]], 9), color=TRAIN, lw=2)
    ax_tps.set_ylabel("throughput\n(k tokens/s)", color=INK_MUTED, fontsize=10)
    ax_tps.set_xlabel("step", color=INK_MUTED, fontsize=10)
    ax_tps.set_ylim(0, max(train["tok_s"]) / 1000 * 1.15)

    fig.suptitle(
        "Nanoserve 27M — TinyStories V2",
        x=0.075, y=0.985, ha="left", fontsize=15, color=INK, fontweight="bold",
    )
    fig.text(
        0.075, 0.950,
        f"{args.steps:,} steps · {args.tokens / 1e6:.0f}M tokens · {args.hours:.2f} h · "
        f"final validation loss {args.final_val:.4f} "
        f"(perplexity {np.exp(args.final_val):.2f})",
        ha="left", fontsize=10, color=INK_MUTED,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.925))
    fig.savefig(args.out, dpi=160, facecolor=SURFACE)
    print(f"wrote {args.out}")
    print(f"  train points {len(train['step']):,}, val points {len(val['step']):,}")
    print(f"  final: train {train['loss'][-1]:.4f}  val {val['loss'][-1]:.4f}")


if __name__ == "__main__":
    main()
