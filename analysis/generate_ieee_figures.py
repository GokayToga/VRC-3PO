#!/usr/bin/env python3
"""Generate camera-ready figures from the frozen VRC-3PO result artifacts.

The script deliberately separates the two principal experiments:

* single-task binary detection from the saved five-seed CNN ensemble; and
* seven-architecture regression/thresholded-detection retraining under
  participant-held-out and purged blocked within-participant protocols.

Every plotted value is recomputed from, or read directly from, a frozen
artifact. The script fails if the single-task headline metrics do not reproduce
the values recorded in ``results/corrected/corrected_headline_metrics.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve

from analysis.corrected_evaluation import (
    ELEVATED_THRESHOLD,
    PASSIVE_SOURCES,
    average_precision,
    build_window_metadata,
    participant_split,
    roc_auc,
)


BLUE = "#2166AC"
ORANGE = "#D6604D"
TEAL = "#1B9E77"
DARK = "#222222"
MID = "#6A6A6A"
LIGHT = "#E8EDF2"
SOURCE_COLORS = {"simulations": BLUE, "terrain": ORANGE}
DISPLAY_VARIANT = {"VRC-3PO": "Full stack"}


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
            "axes.linewidth": 0.7,
            "grid.color": "#D8D8D8",
            "grid.linewidth": 0.5,
            "grid.alpha": 0.7,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: mpl.figure.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.pdf")
    fig.savefig(output_dir / f"{stem}.png", dpi=300)
    plt.close(fig)


def passive_test_data(
    dataset_path: Path, predictions_path: Path, headline_path: Path
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict[str, object]]:
    dataset = pd.read_csv(dataset_path)
    labels, metadata = build_window_metadata(dataset)
    _, _, test_participants = participant_split(labels, metadata, seed=42)
    mask = metadata["participant_id"].isin(test_participants) & metadata[
        "source_dataset"
    ].isin(PASSIVE_SOURCES)
    test_metadata = metadata.loc[mask].reset_index(drop=True)
    y_true = (labels[mask.to_numpy()] > ELEVATED_THRESHOLD).astype(int)
    scores = np.load(predictions_path).reshape(-1)
    headline = json.loads(headline_path.read_text())

    if len(scores) != len(y_true):
        raise ValueError(
            f"Saved scores contain {len(scores)} rows; expected {len(y_true)}."
        )
    checks = {
        "pooled_auc": roc_auc(y_true, scores),
        "pooled_average_precision": average_precision(y_true, scores),
        "passive_test_windows": len(y_true),
        "elevated_windows": int(y_true.sum()),
        "passive_test_participants": int(test_metadata["participant_id"].nunique()),
    }
    for key, observed in checks.items():
        expected = headline[key]
        if isinstance(observed, float):
            if not np.isclose(observed, expected, atol=1e-12):
                raise ValueError(f"{key}: reproduced {observed}, expected {expected}")
        elif observed != expected:
            raise ValueError(f"{key}: reproduced {observed}, expected {expected}")
    return test_metadata, y_true, scores, headline


def figure_study_design(output_dir: Path) -> None:
    """Study-design schematic.

    Box text is measured against the box it sits in and the font is reduced
    until it fits, so a wording change cannot silently produce overflow. The
    chosen sizes are printed for the record.
    """
    fig, ax = plt.subplots(figsize=(7.16, 2.9))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")

    TITLE_SIZE, BODY_SIZE, MIN_SIZE = 7.2, 6.0, 4.6
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    def text_width_units(text_artist) -> float:
        """Rendered width of a text artist, in axis data units."""
        bbox = text_artist.get_window_extent(renderer=renderer)
        x0, _ = ax.transData.inverted().transform((bbox.x0, bbox.y0))
        x1, _ = ax.transData.inverted().transform((bbox.x1, bbox.y1))
        return abs(x1 - x0)

    def fitted(text_artist, budget: float, start_size: float) -> float:
        """Shrink a text artist until it fits the budget; return final size."""
        size = start_size
        while size > MIN_SIZE and text_width_units(text_artist) > budget:
            size -= 0.1
            text_artist.set_fontsize(size)
        return size

    overflow_report = []

    def box(x, y, width, height, title, body, edge=BLUE, fill="white"):
        indent = 0.14
        budget = width - 2 * indent
        ax.add_patch(FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.05,rounding_size=0.08",
            linewidth=1.1, edgecolor=edge, facecolor=fill,
        ))
        title_artist = ax.text(
            x + indent, y + height - 0.22, title, va="top", ha="left",
            fontsize=TITLE_SIZE, weight="bold", color=DARK,
        )
        final_title = fitted(title_artist, budget, TITLE_SIZE)

        # Each body line is measured separately; the widest one sets the size.
        size = BODY_SIZE
        for line in body.split("\n"):
            probe = ax.text(x + indent, y - 5, line, fontsize=size, alpha=0.0)
            size = min(size, fitted(probe, budget, size))
            probe.remove()
        body_artist = ax.text(
            x + indent, y + height - 0.60, body, va="top", ha="left",
            fontsize=size, color=DARK, linespacing=1.3,
        )
        overflow_report.append((title, round(final_title, 1), round(size, 1)))
        return body_artist

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle="-|>",
            mutation_scale=10, linewidth=1.0, color=MID,
        ))

    # Four columns, widened so the longest line in each has room.
    box(0.10, 3.60, 2.70, 1.05, "Maze",
        "37 people; 2,262 windows\nReal walking + dual task", edge="#7570B3")
    box(0.10, 2.20, 2.70, 1.05, "Simulations",
        "25 people; 2,983 windows\nFive VR environments", edge=BLUE)
    box(0.10, 0.80, 2.70, 1.05, "Terrain",
        "22 people; 1,197 windows\nSeated joystick steering", edge=ORANGE)

    box(3.20, 1.50, 2.70, 2.25, "Harmonization",
        "1 Hz sampling\n14 shared eye/head signals\n30-s windows; 15-s stride\n"
        "6,442 windows; 84 identifiers",
        edge=TEAL, fill="#F4FAF8")

    box(6.30, 2.75, 2.75, 1.55, "Participant-held-out",
        "58 / 9 / 17 identifiers\nNo identifier is in two splits\nPrimary transfer estimate",
        edge=BLUE, fill="#F3F7FB")
    box(6.30, 0.70, 2.75, 1.55, "Purged blocked split",
        "84 identifiers in all sets\nChronological 60/20/20\n15-s boundary purge",
        edge=ORANGE, fill="#FCF5F3")

    box(9.45, 2.75, 2.45, 1.55, "Single-task binary",
        "9 held-out people\n768 windows; AUC 0.757\nCluster CI 0.696-0.814",
        edge=TEAL)
    box(9.45, 0.70, 2.45, 1.55, "Architecture study",
        "7 models; 2 protocols\n3 seeds = 42 runs\nRegression and AUC",
        edge="#7570B3")

    for y in (4.10, 2.70, 1.30):
        arrow(2.80, y, 3.20, 2.62)
    arrow(5.90, 2.62, 6.30, 3.52)
    arrow(5.90, 2.62, 6.30, 1.47)
    arrow(9.05, 3.52, 9.45, 3.52)
    arrow(9.05, 1.47, 9.45, 1.47)

    print("  fig1 fitted font sizes (title, body):")
    for name, t, b in overflow_report:
        flag = "  <- shrunk" if b < BODY_SIZE or t < TITLE_SIZE else ""
        print(f"    {name:24} {t:>4} {b:>4}{flag}")

    save_figure(fig, output_dir, "fig1_study_design")


def figure_detection_curves(
    output_dir: Path,
    test_metadata: pd.DataFrame,
    y_true: np.ndarray,
    scores: np.ndarray,
    headline: dict[str, object],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.85))
    fpr, tpr, _ = roc_curve(y_true, scores)
    precision, recall, _ = precision_recall_curve(y_true, scores)
    auc = float(headline["pooled_auc"])
    ap = float(headline["pooled_average_precision"])
    prevalence = float(headline["prevalence"])

    axes[0].plot(fpr, tpr, color=BLUE, linewidth=1.8, label=f"CNN ensemble (AUC={auc:.3f})")
    axes[0].plot([0, 1], [0, 1], color=MID, linestyle="--", linewidth=0.9, label="Chance")
    axes[0].fill_between(
        [0, 1],
        [0, 0],
        [0, 1],
        color=BLUE,
        alpha=0.04,
        linewidth=0,
    )
    axes[0].set(
        xlim=(0, 1),
        ylim=(0, 1.02),
        xlabel="False-positive rate",
        ylabel="True-positive rate",
        title="(a) Receiver operating characteristic",
    )
    axes[0].legend(loc="lower right", frameon=False)

    axes[1].plot(recall, precision, color=ORANGE, linewidth=1.8, label=f"CNN ensemble (AP={ap:.3f})")
    axes[1].axhline(
        prevalence,
        color=MID,
        linestyle="--",
        linewidth=0.9,
        label=f"Prevalence={prevalence:.3f}",
    )
    axes[1].set(
        xlim=(0, 1),
        ylim=(0, 1.02),
        xlabel="Recall",
        ylabel="Precision",
        title="(b) Precision–recall curve",
    )
    axes[1].legend(loc="upper right", frameon=False)
    for ax in axes:
        ax.grid(True)

    ci = headline["cluster_bootstrap_auc"]
    fig.suptitle(
        "Single-task participant-held-out detection: "
        f"{test_metadata['participant_id'].nunique()} participants, "
        f"{len(y_true)} windows; participant-cluster 95% AUC CI "
        f"{ci['lower']:.3f}–{ci['upper']:.3f}",
        y=1.01,
        fontsize=8.5,
    )
    fig.tight_layout()
    save_figure(fig, output_dir, "fig2_detection_curves")


def figure_protocol_gap(summary_path: Path, output_dir: Path) -> None:
    table = pd.read_csv(summary_path)
    order = ["MLP", "CNN", "CNN+GRU", "CNN+TCN", "CNN+TCN+GRU", "Lite", "VRC-3PO"]
    wide_auc = table.pivot(index="variant", columns="protocol", values="auc_mean").loc[order]
    wide_r2 = table.pivot(index="variant", columns="protocol", values="r2_mean").loc[order]
    color_values = mpl.colormaps["tab10"](np.linspace(0, 0.8, len(order)))

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.15))
    for ax, wide, ylabel, title, baseline in [
        (axes[0], wide_auc, "Thresholded AUC", "(a) Detection discrimination", 0.5),
        (axes[1], wide_r2, r"$R^2$", "(b) Absolute FMS regression", 0.0),
    ]:
        for variant, color in zip(order, color_values):
            values = [
                wide.loc[variant, "blocked_within_participant"],
                wide.loc[variant, "participant"],
            ]
            ax.plot([0, 1], values, marker="o", markersize=4.2, linewidth=1.0, color=color)
        ax.axhline(baseline, color=MID, linestyle="--", linewidth=0.8)
        ax.set_xticks([0, 1], ["Blocked within-\nparticipant", "Participant-\nheld-out"])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, axis="y")
        ax.set_xlim(-0.15, 1.15)
    axes[0].set_ylim(0.48, 0.82)
    axes[1].set_ylim(-0.72, 0.46)
    legend = [
        Line2D(
            [0],
            [0],
            color=color,
            marker="o",
            linewidth=1.0,
            label=DISPLAY_VARIANT.get(variant, variant),
        )
        for variant, color in zip(order, color_values)
    ]
    fig.legend(
        handles=legend,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(
        "Changing the evaluation unit alters both performance magnitude and architecture ranking",
        y=1.01,
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    save_figure(fig, output_dir, "fig3_protocol_gap")


def figure_participant_variation(per_participant_path: Path, output_dir: Path) -> None:
    table = pd.read_csv(per_participant_path)
    table["short_id"] = table["participant_id"].str.replace(
        r"^(simulations|terrain)_", "", regex=True
    )
    table["short_id"] = table.apply(
        lambda row: ("S" if row["source_dataset"] == "simulations" else "T")
        + row["short_id"].replace(".0", ""),
        axis=1,
    )
    table = table.sort_values(["source_dataset", "prevalence", "short_id"]).reset_index(drop=True)
    y = np.arange(len(table))

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.16, 3.25),
        sharey=True,
        gridspec_kw={"width_ratios": [1.0, 1.15]},
    )
    colors = table["source_dataset"].map(SOURCE_COLORS).to_numpy()
    axes[0].barh(y, table["prevalence"], color=colors, alpha=0.82, height=0.62)
    axes[0].axvline(0.14973958333333334, color=MID, linestyle="--", linewidth=0.8)
    axes[0].set(
        xlabel="Elevated-window prevalence",
        ylabel="Held-out participant",
        title="(a) Test composition",
        xlim=(0, 0.72),
        yticks=y,
        yticklabels=table["short_id"],
    )
    axes[0].grid(True, axis="x")

    eligible = table["auc"].notna()
    axes[1].scatter(
        table.loc[eligible, "auc"],
        y[eligible],
        c=colors[eligible],
        s=28,
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )
    axes[1].scatter(
        np.full((~eligible).sum(), 0.5),
        y[~eligible],
        marker="x",
        c=MID,
        s=24,
        linewidth=1.0,
        zorder=3,
    )
    axes[1].axvline(0.5, color=MID, linestyle="--", linewidth=0.8)
    axes[1].set(
        xlabel="Within-participant AUC",
        title="(b) Individual discrimination",
        xlim=(0.45, 1.02),
    )
    axes[1].grid(True, axis="x")
    axes[1].text(
        0.985,
        0.02,
        "× single-class participant",
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        color=MID,
        fontsize=6.5,
    )
    handles = [
        Line2D([0], [0], color=BLUE, linewidth=5, label="Simulations"),
        Line2D([0], [0], color=ORANGE, linewidth=5, label="Terrain"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False)
    fig.suptitle(
        "Participant heterogeneity in the single-task held-out test set",
        y=1.01,
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save_figure(fig, output_dir, "fig4_participant_variation")


def figure_paradigm_gap(
    predictions_path: Path, output_dir: Path, data_dir: Path
) -> None:
    predictions = pd.read_csv(predictions_path)
    predictions = predictions.loc[
        predictions["protocol"].eq("participant")
    ].copy()
    predictions["elevated"] = (predictions["y_true"] > ELEVATED_THRESHOLD).astype(int)
    predictions["paradigm"] = np.where(
        predictions["source_dataset"].eq("maze"),
        "Active dual-task Maze",
        "Single-task sources",
    )

    rows: list[dict[str, object]] = []
    for (variant, seed, paradigm), group in predictions.groupby(
        ["variant", "seed", "paradigm"]
    ):
        rows.append(
            {
                "variant": variant,
                "seed": int(seed),
                "paradigm": paradigm,
                "participants": int(group["participant_id"].nunique()),
                "windows": int(len(group)),
                "prevalence": float(group["elevated"].mean()),
                "auc": float(roc_auc_score(group["elevated"], group["prediction"])),
            }
        )
    seed_table = pd.DataFrame(rows)
    data_dir.mkdir(parents=True, exist_ok=True)
    seed_table.to_csv(data_dir / "table_paradigm_auc_by_seed.csv", index=False)

    order = ["MLP", "CNN", "CNN+GRU", "CNN+TCN", "CNN+TCN+GRU", "Lite", "VRC-3PO"]
    summary = (
        seed_table.groupby(["variant", "paradigm"])["auc"]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.to_csv(data_dir / "table_paradigm_auc_summary.csv", index=False)

    single_task = (
        summary.loc[summary["paradigm"].eq("Single-task sources")]
        .set_index("variant")
        .loc[order]
    )
    active = (
        summary.loc[summary["paradigm"].eq("Active dual-task Maze")]
        .set_index("variant")
        .loc[order]
    )

    # Average the three CNN regression scores for a participant-clustered
    # sensitivity analysis of the observed paradigm gap.
    cnn = predictions.loc[predictions["variant"].eq("CNN")].copy()
    key = [
        "participant_id",
        "condition",
        "source_dataset",
        "start_index",
        "end_index",
        "y_true",
        "elevated",
        "paradigm",
    ]
    cnn = cnn.groupby(key, as_index=False)["prediction"].mean()
    observed = {}
    for paradigm, group in cnn.groupby("paradigm"):
        observed[paradigm] = float(
            roc_auc_score(group["elevated"], group["prediction"])
        )

    rng = np.random.default_rng(20240724)
    differences: list[float] = []
    for _ in range(5000):
        aucs: dict[str, float] = {}
        valid = True
        for paradigm in ["Single-task sources", "Active dual-task Maze"]:
            group = cnn.loc[cnn["paradigm"].eq(paradigm)]
            participant_ids = group["participant_id"].unique()
            sampled = rng.choice(
                participant_ids, size=len(participant_ids), replace=True
            )
            chunks = [
                group.loc[group["participant_id"].eq(participant_id)]
                for participant_id in sampled
            ]
            boot = pd.concat(chunks, ignore_index=True)
            if boot["elevated"].nunique() < 2:
                valid = False
                break
            aucs[paradigm] = float(
                roc_auc_score(boot["elevated"], boot["prediction"])
            )
        if valid:
            differences.append(
                aucs["Single-task sources"]
                - aucs["Active dual-task Maze"]
            )
    lower, upper = np.quantile(differences, [0.025, 0.975])
    gap_summary = {
        "cnn_three_seed_score_ensemble_single_task_auc": observed[
            "Single-task sources"
        ],
        "cnn_three_seed_score_ensemble_active_maze_auc": observed[
            "Active dual-task Maze"
        ],
        "auc_difference_single_task_minus_active": observed[
            "Single-task sources"
        ]
        - observed["Active dual-task Maze"],
        "participant_cluster_bootstrap_valid_draws": len(differences),
        "participant_cluster_bootstrap_difference_ci_lower": float(lower),
        "participant_cluster_bootstrap_difference_ci_upper": float(upper),
    }
    (data_dir / "paradigm_gap_cluster_summary.json").write_text(
        json.dumps(gap_summary, indent=2) + "\n"
    )

    x = np.arange(len(order))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.16, 3.55))
    ax.bar(
        x - width / 2,
        single_task["mean"],
        width,
        yerr=single_task["std"],
        capsize=2.5,
        color=BLUE,
        alpha=0.86,
        label="Single-task sources",
    )
    ax.bar(
        x + width / 2,
        active["mean"],
        width,
        yerr=active["std"],
        capsize=2.5,
        color=ORANGE,
        alpha=0.86,
        label="Active walking + dual task",
    )
    ax.axhline(0.5, color=MID, linestyle="--", linewidth=0.8)
    ax.set_xticks(
        x,
        [DISPLAY_VARIANT.get(variant, variant) for variant in order],
        rotation=18,
        ha="right",
    )
    ax.set(
        ylabel="Participant-held-out AUC",
        ylim=(0.18, 0.9),
        title="The single-task-to-Maze performance gap appears in every architecture",
    )
    ax.grid(True, axis="y")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    save_figure(fig, output_dir, "fig5_paradigm_gap")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("/Users/gokaytoga/Downloads/vrc3po_master_dataset_fixed.csv"),
    )
    parser.add_argument(
        "--passive-predictions",
        type=Path,
        default=Path("results/corrected/cnn_ensemble_preds_OG.npy"),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("manuscript/ieee_access/figures"),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("manuscript/ieee_access/data"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_style()
    corrected = args.results_dir / "corrected"
    colab = args.results_dir / "colab" / "full"
    test_metadata, y_true, scores, headline = passive_test_data(
        args.dataset,
        args.passive_predictions,
        corrected / "corrected_headline_metrics.json",
    )
    figure_study_design(args.output_dir)
    figure_detection_curves(
        args.output_dir, test_metadata, y_true, scores, headline
    )
    figure_protocol_gap(colab / "architecture_summary.csv", args.output_dir)
    figure_participant_variation(
        corrected / "per_participant_metrics.csv", args.output_dir
    )
    figure_paradigm_gap(
        colab / "architecture_test_predictions.csv",
        args.output_dir,
        args.data_dir,
    )
    print(f"Generated five PDF/PNG figure pairs in {args.output_dir}")


if __name__ == "__main__":
    main()
