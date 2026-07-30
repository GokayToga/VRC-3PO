#!/usr/bin/env python3
"""Figure for the operating-point and calibration analysis.

Reads only the frozen outputs written by ``analysis.operating_points`` so the
figure cannot drift from the reported table.
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
TEAL = "#1B9E77"
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
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()

    root = args.repo_root
    corrected = root / "results" / "corrected"
    output_dir = _default_figure_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    sweep = pd.read_csv(corrected / "operating_point_sweep.csv")
    points = pd.read_csv(corrected / "operating_points.csv")
    calibration = pd.read_csv(corrected / "calibration_bins.csv")
    summary = json.loads((corrected / "operating_point_summary.json").read_text())

    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.9))

    ax = axes[0]
    ax.plot(sweep["threshold"], sweep["sensitivity"], color=BLUE, linewidth=1.7,
            label="Sensitivity")
    ax.plot(sweep["threshold"], sweep["specificity"], color=ORANGE, linewidth=1.7,
            label="Specificity")
    ax.plot(sweep["threshold"], sweep["false_alarm_rate"], color=TEAL,
            linewidth=1.3, linestyle="--", label="False-alarm rate")

    youden = points.loc[points["name"] == "youden"].iloc[0]
    spec90 = points.loc[points["name"] == "specificity_90"].iloc[0]
    for row, label in ((youden, "Youden"), (spec90, "90 % spec.")):
        ax.axvline(row["threshold"], color=MID, linewidth=0.8, linestyle=":")
        ax.annotate(
            label,
            xy=(row["threshold"], 0.955),
            xycoords=("data", "axes fraction"),
            ha="center",
            va="top",
            fontsize=6.5,
            color=DARK,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8, "alpha": 0.9},
        )

    ax.set_xlabel("Decision threshold on ensemble score")
    ax.set_ylabel("Rate")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(sweep["threshold"].min(), sweep["threshold"].max())
    ax.legend(loc="center left", frameon=False)
    ax.set_title("(a) Threshold sweep")

    ax = axes[1]
    ax.plot([0, 1], [0, 1], color=MID, linestyle="--", linewidth=0.9,
            label="Perfect calibration")
    ax.plot(calibration["mean_score"], calibration["observed_rate"], marker="o",
            markersize=4.2, linewidth=1.5, color=BLUE, label="Ensemble (8 equal-count bins)")
    ax.axhline(summary["prevalence"], color=ORANGE, linewidth=0.9, linestyle=":",
               label=f"Prevalence = {summary['prevalence']:.3f}")

    ax.set_xlabel("Mean predicted score in bin")
    ax.set_ylabel("Observed elevated fraction")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper left", frameon=False)
    ax.set_title("(b) Reliability")

    fig.tight_layout()
    fig.savefig(output_dir / "fig6_operating_points.pdf")
    fig.savefig(output_dir / "fig6_operating_points.png", dpi=300)
    print("wrote", output_dir / "fig6_operating_points.pdf")


if __name__ == "__main__":
    main()
