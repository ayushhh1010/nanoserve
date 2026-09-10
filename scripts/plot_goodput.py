"""The goodput curve: why a busy server can be a useless one.

Throughput counts tokens produced. Goodput counts requests that finished while
anyone still wanted them. Past capacity the two diverge completely, and only
the second one is the service.

Two panels, sharing an x-axis, because they carry opposite news at the same
load: throughput stays flat and healthy exactly where goodput reaches zero.
Plotting them on one pair of axes would invite reading a trade-off off the
crossing point, which is an artifact of the scales rather than anything real.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Validated 2-slot categorical palette (dataviz validator, light mode):
# adjacent CVD dE 24.7, normal-vision dE 33.6, both >= 3:1 on the surface.
OFF = "#eb6834"
ON = "#2a78d6"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e3e2df"


def load(pattern: str) -> dict[str, list[tuple[int, dict]]]:
    series: dict[str, list[tuple[int, dict]]] = {"off": [], "on": []}
    for f in glob.glob(pattern):
        parts = Path(f).stem.split("_")
        rate, adm = int(parts[-2]), parts[-1]
        if adm in series:
            series[adm].append((rate, json.load(open(f, encoding="utf-8"))["result"]))
    for k in series:
        series[k].sort(key=lambda r: r[0])
    return series


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pattern", default="bench/results/g2_*.json")
    ap.add_argument("--out", type=Path, default=Path("writeups/img/goodput.png"))
    ap.add_argument("--slo", type=float, default=1.0)
    args = ap.parse_args()

    s = load(args.pattern)
    if not s["off"] or not s["on"]:
        raise SystemExit(f"no results matched {args.pattern}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_g, ax_t) = plt.subplots(2, 1, figsize=(9, 7), height_ratios=[3, 2], sharex=True)
    fig.patch.set_facecolor(SURFACE)

    for key, color, label in (("off", OFF, "no admission control"), ("on", ON, "admission control")):
        rates = [r for r, _ in s[key]]
        good = [d["goodput"]["requests_per_s"] for _, d in s[key]]
        toks = [d["throughput"]["output_tokens_per_s"] for _, d in s[key]]
        ax_g.plot(rates, good, color=color, lw=2, marker="o", markersize=5, label=label)
        ax_t.plot(rates, toks, color=color, lw=2, marker="o", markersize=5, label=label)

    style(ax_g)
    ax_g.set_xscale("log")
    # Plain numbers on the ticks. "4 x 10^1" is the offered request rate, and
    # scientific notation makes a reader decode it rather than read it.
    rates_all = sorted({r for _, lst in s.items() for r, _ in lst})
    ax_g.set_xticks(rates_all)
    ax_g.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax_g.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax_g.set_ylabel(f"goodput\n(req/s finishing within {args.slo:g}s)", color=MUTED, fontsize=10)
    ax_g.legend(loc="upper left", frameon=False, fontsize=10, labelcolor=INK, handlelength=1.6)

    # Mark where the unguarded curve reaches zero -- the whole point.
    zero = [r for r, d in s["off"] if d["goodput"]["requests_per_s"] < 0.05]
    if zero:
        ax_g.axvline(min(zero), color=MUTED, lw=1, ls=(0, (3, 3)))
        ax_g.annotate(
            f"unguarded goodput reaches 0\nat {min(zero)} req/s offered",
            xy=(min(zero), 0), xytext=(8, 40), textcoords="offset points",
            fontsize=9, color=MUTED,
        )

    style(ax_t)
    ax_t.set_ylabel("throughput\n(output tokens/s)", color=MUTED, fontsize=10)
    ax_t.set_xlabel("offered load (requests/s, log scale)", color=MUTED, fontsize=10)
    ax_t.set_ylim(0, max(d["throughput"]["output_tokens_per_s"] for _, d in s["off"]) * 1.2)

    fig.suptitle(
        "Goodput collapses where throughput does not",
        x=0.075, y=0.985, ha="left", fontsize=15, color=INK, fontweight="bold",
    )
    fig.text(
        0.075, 0.945,
        "Past capacity the server stays 100% busy while nothing it produces is still wanted.",
        ha="left", fontsize=9.5, color=MUTED,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.905))
    fig.savefig(args.out, dpi=160, facecolor=SURFACE)
    print(f"wrote {args.out}")
    for key in ("off", "on"):
        pts = ", ".join(f"{r}:{d['goodput']['requests_per_s']:.1f}" for r, d in s[key])
        print(f"  {key:>3}: {pts}")


if __name__ == "__main__":
    main()
