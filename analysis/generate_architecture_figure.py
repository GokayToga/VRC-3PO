#!/usr/bin/env python3
"""Architecture progression figure with the ablation outcome.

Shows the seven candidates as a controlled progression -- each adds one temporal
mechanism to the preceding one -- alongside parameter count and mean AUC under
both evaluation protocols. The point of the figure is that the two AUC rows are
ordered differently from each other and from parameter count.

All numbers are read from ``manuscript/ieee_access/data/table_architecture_ablation.csv``
so the figure cannot drift from Table `tab:ablation`.

Note on naming: the largest candidate is labelled "VRC-3PO (full stack)", the
name used in the article for the combination of all four mechanisms. Naming it
does not privilege it: it is the hypothesis under test, it is one point on the
progression, and it does not win under participant holdout.

The output file has no number in its name. This figure sits in the Methods
section, so LaTeX numbers it before the Results figures, and the ``figN_``
filenames of the other figures no longer correspond to their printed numbers.
Reference figures by label, not by filename.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import pandas as pd

DARK = "#222222"
MID = "#6A6A6A"
GREEN = "#1B7A4B"
RED = "#C0392B"

LAYER_COLORS = {
    "input": ("#D6E4F2", "#5B8DB8"),
    "conv": ("#FADBD8", "#C0603A"),
    "tcn": ("#FDEBD0", "#D68910"),
    "gru": ("#E2DCEF", "#7D6BA8"),
    "attn": ("#FADCE8", "#C2568C"),
    "norm": ("#D8EDE1", "#3B8C63"),
    "drop": ("#E5E4E0", "#8A8880"),
    "pool": ("#D6EDE8", "#3D9B8B"),
    "dense": ("#D6E0F2", "#4A6FA5"),
}
LEGEND_ORDER = [
    ("input", "Input"), ("conv", "Conv / CNN"), ("tcn", "TCN"), ("gru", "GRU"),
    ("attn", "Attention"), ("norm", "Norm"), ("drop", "Dropout"),
    ("pool", "Pooling"), ("dense", "Dense / output"),
]

# (display name, csv key, subtitle, [(layer kind, label), ...])
COLUMNS = [
    ("MLP", "MLP", "baseline\nno temporal scale", [
        ("input", "Input 30x14"), ("dense", "Dense(64) TD"), ("norm", "Batch Norm"),
        ("drop", "Dropout(.2)"), ("dense", "Dense(32) TD"),
        ("pool", "Global Avg Pool"), ("dense", "Dense output")]),
    ("CNN", "CNN", "+ convolution\nlocal scale", [
        ("input", "Input 30x14"), ("conv", "Conv1D(64, k3)"), ("norm", "Batch Norm"),
        ("drop", "Dropout(.1)"), ("pool", "Global Avg Pool"), ("dense", "Dense output")]),
    ("CNN + GRU", "CNN+GRU", "+ recurrence\nlong-range scale", [
        ("input", "Input 30x14"), ("conv", "CNN block"), ("gru", "GRU(64) seq"),
        ("gru", "GRU(16) seq"), ("pool", "Global Avg Pool"), ("dense", "Dense output")]),
    ("CNN + TCN", "CNN+TCN", "+ dilated convolution\nmedium-range scale", [
        ("input", "Input 30x14"), ("conv", "CNN block"),
        ("tcn", "TCN(64) d 1,2,4,8"), ("drop", "Dropout(.2)"),
        ("pool", "Global Avg Pool"), ("dense", "Dense output")]),
    ("CNN + TCN\n+ GRU", "CNN+TCN+GRU", "+ combined\nmedium and long range", [
        ("input", "Input 30x14"), ("conv", "CNN block"), ("tcn", "TCN block"),
        ("norm", "BN + Dropout"), ("gru", "GRU(64) seq"), ("gru", "GRU(16) seq"),
        ("pool", "Global Avg Pool"), ("dense", "Dense output")]),
    ("VRC-3PO\n(full stack)", "VRC-3PO", "+ attention\nall scales at once", [
        ("input", "Input 30x14"), ("conv", "CNN block"), ("tcn", "TCN block"),
        ("norm", "BN + Dropout"), ("gru", "GRU(64) seq"), ("gru", "GRU(16) seq"),
        ("attn", "3-head attention"), ("norm", "Residual + LN"),
        ("pool", "Global Avg Pool"), ("dense", "Dense output")]),
    ("Lite", "Lite", "reduced capacity\nCNN + GRU", [
        ("input", "Input 30x14"), ("conv", "Conv1D(32)"), ("norm", "Batch Norm"),
        ("gru", "GRU(32) seq"), ("gru", "GRU(16)"), ("dense", "Dense(8) output")]),
]

BAR_LOW, BAR_HIGH = 0.50, 0.85


def _resolve(root: Path, *tail: str) -> Path:
    """Find a path under either the manuscript tree or the flattened archive.

    The manuscript keeps files under ``manuscript/ieee_access/``; the
    reproducibility archive flattens that to ``manuscript/``. Scripts must work
    in both, so resolve against whichever exists.
    """
    for base in (root / "manuscript" / "ieee_access", root / "manuscript"):
        candidate = base.joinpath(*tail)
        if candidate.exists() or candidate.parent.is_dir():
            return candidate
    return root.joinpath("manuscript", *tail)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path,
                        default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()

    root = args.repo_root
    table = pd.read_csv(
        _resolve(root, "data", "table_architecture_ablation.csv")
    ).set_index("variant")

    figures = _resolve(root, "figures")
    figures.mkdir(parents=True, exist_ok=True)

    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
    })

    n = len(COLUMNS)
    fig, ax = plt.subplots(figsize=(7.16, 4.75))
    ax.set_xlim(0, n)
    ax.set_ylim(0, 15.5)
    ax.axis("off")

    col_w, box_h, gap = 0.80, 0.60, 0.14
    top = 12.5

    for i, (display, key, subtitle, layers) in enumerate(COLUMNS):
        cx = i + 0.5
        ax.text(cx, 15.25, display, ha="center", va="top",
                fontsize=6.2, weight="bold", color=DARK, linespacing=1.25)
        ax.text(cx, 13.95, subtitle, ha="center", va="top",
                fontsize=5.0, style="italic", color=MID, linespacing=1.3)

        # spine behind the stack
        ax.plot([cx, cx], [top - len(layers) * (box_h + gap) - 0.05, top + 0.1],
                color="#DDDDDD", linewidth=0.7, zorder=0)

        for j, (kind, label) in enumerate(layers):
            y = top - (j + 1) * (box_h + gap)
            face, edge = LAYER_COLORS[kind]
            ax.add_patch(FancyBboxPatch(
                (cx - col_w / 2, y), col_w, box_h,
                boxstyle="round,pad=0.012,rounding_size=0.05",
                facecolor=face, edgecolor=edge, linewidth=0.8, zorder=2))
            ax.text(cx, y + box_h / 2, label, ha="center", va="center",
                    fontsize=4.5, color=DARK, zorder=3)

        row = table.loc[key]
        ax.text(cx, 2.55, f"{int(row.parameters):,}", ha="center", va="center",
                fontsize=5.4, color=DARK)

        for value, y0, color, best_key in (
            (row.auc_mean_blocked_within_participant, 1.35, GREEN,
             "auc_mean_blocked_within_participant"),
            (row.auc_mean_participant, 0.35, RED, "auc_mean_participant"),
        ):
            is_best = value == table[best_key].max()
            frac = max(0.0, min(1.0, (value - BAR_LOW) / (BAR_HIGH - BAR_LOW)))
            ax.add_patch(plt.Rectangle((cx - col_w / 2, y0), col_w, 0.42,
                                       facecolor="#EFEFEF", edgecolor="none"))
            ax.add_patch(plt.Rectangle((cx - col_w / 2, y0), col_w * frac, 0.42,
                                       facecolor=color, edgecolor="none"))
            ax.text(cx + col_w / 2 - 0.03, y0 + 0.21, f"{value:.3f}",
                    ha="right", va="center", fontsize=5.2,
                    weight="bold" if is_best else "normal", color=DARK)

    ax.text(-0.04, 2.55, "Parameters", ha="right", va="center",
            fontsize=5.4, color=MID)
    ax.text(-0.04, 1.56, "Purged blocked AUC", ha="right", va="center",
            fontsize=5.4, weight="bold", color=GREEN)
    ax.text(-0.04, 0.56, "Participant-held-out AUC", ha="right", va="center",
            fontsize=5.4, weight="bold", color=RED)

    for x0, y0 in ((0.0, 3.05),):
        ax.plot([x0, n], [y0, y0], color="#CCCCCC", linewidth=0.7)

    handles = [
        plt.Line2D([0], [0], marker="s", linestyle="none", markersize=4.6,
                   markerfacecolor=LAYER_COLORS[k][0],
                   markeredgecolor=LAYER_COLORS[k][1], label=label)
        for k, label in LEGEND_ORDER
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.10),
              ncol=9, frameon=False, fontsize=5.0, handletextpad=0.4,
              columnspacing=1.0)

    ax.text(n, -0.30, f"bars scaled {BAR_LOW:.2f} (chance) to {BAR_HIGH:.2f}; "
            "bold marks the best value in each row",
            ha="right", va="top", fontsize=4.6, style="italic", color=MID)

    fig.savefig(figures / "fig_architectures.pdf")
    fig.savefig(figures / "fig_architectures.png", dpi=300)
    print("wrote", figures / "fig_architectures.pdf")
    print("  best purged blocked:",
          table.auc_mean_blocked_within_participant.idxmax())
    print("  best participant-held-out:", table.auc_mean_participant.idxmax())


if __name__ == "__main__":
    main()
