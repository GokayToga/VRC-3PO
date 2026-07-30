#!/usr/bin/env python3
"""Figure: pooled single-task AUC across participant compositions.

Reads the per-composition records written by
``analysis.composition_robustness`` -- either the individual JSON files from a
Colab run, or the aggregated ``composition_results.csv`` shipped in the
reproducibility archive -- so the figure cannot drift from the reported
summary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE = "#2166AC"
ORANGE = "#D6604D"
DARK = "#222222"
MID = "#6A6A6A"


def _default_figure_dir() -> Path:
    """Resolve the figures directory for either the manuscript tree or the
    reproducibility archive, which flattens ``manuscript/ieee_access`` to
    ``manuscript``."""
    root = Path(__file__).resolve().parents[1]
    for candidate in (
        root / "manuscript" / "ieee_access" / "figures",
        root / "manuscript" / "figures",
    ):
        if candidate.is_dir():
            return candidate
    return root / "manuscript" / "figures"


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.35,
            "grid.color": "#D8D8D8",
            "grid.linewidth": 0.5,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    # Per-composition JSONs are written by the Colab run. The reproducibility
    # archive ships the aggregated CSV instead, which carries every field this
    # figure needs, so fall back to it when the JSONs are not present.
    records = [
        json.loads(path.read_text())
        for path in sorted(args.results_dir.glob("composition_*.json"))
        if path.name != "composition_summary.json"
    ]
    if not records:
        table = args.results_dir / "composition_results.csv"
        if not table.exists():
            raise SystemExit(
                f"found neither composition_*.json nor composition_results.csv "
                f"in {args.results_dir}"
            )
        records = pd.read_csv(table).to_dict("records")
        print(f"[fallback] read {len(records)} compositions from {table.name}")

    complete = [r for r in records if r.get("status") == "complete"]
    if not complete:
        raise SystemExit(f"no completed compositions found in {args.results_dir}")

    summary_path = args.results_dir / "composition_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    frozen = summary.get("frozen_split_auc", 0.756947866036354)

    complete.sort(key=lambda r: r["pooled_auc"])
    aucs = np.asarray([r["pooled_auc"] for r in complete])
    lower = np.asarray([r["cluster_ci_lower"] for r in complete])
    upper = np.asarray([r["cluster_ci_upper"] for r in complete])
    positions = np.arange(len(complete))

    output_dir = args.output_dir or _default_figure_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    set_style()
    fig, axes = plt.subplots(
        1, 2, figsize=(7.16, 2.9), gridspec_kw={"width_ratios": [2.3, 1.0]}
    )

    ax = axes[0]
    clears = lower > 0.5
    ax.vlines(positions, lower, upper, color=MID, linewidth=1.0, zorder=1)
    ax.scatter(
        positions[clears],
        aucs[clears],
        s=22,
        color=BLUE,
        zorder=3,
        label="Cluster interval clears 0.5",
    )
    ax.scatter(
        positions[~clears],
        aucs[~clears],
        s=22,
        facecolor="white",
        edgecolor=BLUE,
        linewidth=1.0,
        zorder=3,
        label="Interval includes 0.5",
    )
    ax.axhline(0.5, color=MID, linestyle="--", linewidth=0.9, label="Chance")
    ax.axhline(
        frozen,
        color=ORANGE,
        linewidth=1.2,
        label=f"Frozen split ({frozen:.3f})",
    )
    ax.set_xlabel("Participant composition (sorted by AUC)")
    ax.set_ylabel("Pooled single-task AUC")
    ax.set_xticks(positions)
    ax.set_xticklabels([str(r["composition_seed"]) for r in complete], fontsize=6)
    ax.legend(loc="upper left", frameon=False)
    ax.set_title("(a) Per-composition result")

    ax = axes[1]
    ax.hist(aucs, bins=8, color=BLUE, alpha=0.75, edgecolor="white")
    ax.axvline(0.5, color=MID, linestyle="--", linewidth=0.9)
    ax.axvline(frozen, color=ORANGE, linewidth=1.2)
    ax.axvline(aucs.mean(), color=DARK, linestyle=":", linewidth=1.1)
    ax.annotate(
        f"mean {aucs.mean():.3f}",
        xy=(aucs.mean(), 0.96),
        xycoords=("data", "axes fraction"),
        ha="center",
        va="top",
        fontsize=6.5,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8, "alpha": 0.9},
    )
    ax.set_xlabel("Pooled single-task AUC")
    ax.set_ylabel("Compositions")
    ax.set_title("(b) Distribution")

    fig.tight_layout()
    fig.savefig(output_dir / "fig7_composition.pdf")
    fig.savefig(output_dir / "fig7_composition.png", dpi=300)
    print("wrote", output_dir / "fig7_composition.pdf")


if __name__ == "__main__":
    main()
