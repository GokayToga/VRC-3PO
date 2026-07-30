#!/usr/bin/env python3
"""Operating-point and calibration analysis for the single-task CNN ensemble.

The manuscript's Implications for Adaptive VR section previously listed a
chosen threshold, calibrated probabilities and false-alarm analysis as work
that a practical system "would need". Those quantities are computable from
artifacts that are already frozen, so this module produces them instead of
deferring them.

Inputs (both frozen, see ``results/ARTIFACT_MANIFEST.md``):

* ``results/corrected/cnn_ensemble_preds_OG.npy`` -- 768 averaged five-seed
  probabilities for the single-task held-out windows;
* ``results/colab/full/architecture_test_predictions.csv`` -- supplies the
  aligned window metadata and continuous FMS target for the same windows.

The alignment between the two files is verified against the recorded pooled
AUC in ``results/corrected/corrected_headline_metrics.json`` before any
operating point is reported. The script exits non-zero if it does not
reproduce that value, so a silent misalignment cannot reach the manuscript.

Every interval resamples participants, never individual windows, matching the
uncertainty convention used for the headline result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.corrected_evaluation import (
    ELEVATED_THRESHOLD,
    average_precision,
    roc_auc,
)

PASSIVE_SOURCES = frozenset({"simulations", "terrain"})
REFERENCE_VARIANT = "MLP"
REFERENCE_SEED = 42
AUC_TOLERANCE = 1e-9


def load_aligned_frame(
    predictions_path: Path,
    architecture_predictions_path: Path,
    participant_metrics_path: Path,
) -> pd.DataFrame:
    """Return one row per single-task held-out window with score and binary label.

    ``cnn_ensemble_preds_OG.npy`` stores scores without metadata. The
    architecture prediction export lists the same windows, in the same order,
    for every variant/seed pair under the participant-held-out protocol, so any
    single variant supplies the metadata and the continuous target. The choice
    of variant does not affect the metadata columns.
    """
    scores = np.load(predictions_path)
    architecture = pd.read_csv(architecture_predictions_path)
    participants = pd.read_csv(participant_metrics_path)

    seated_participants = set(participants["participant_id"])
    frame = architecture[
        (architecture["protocol"] == "participant")
        & (architecture["variant"] == REFERENCE_VARIANT)
        & (architecture["seed"] == REFERENCE_SEED)
        & (architecture["participant_id"].isin(seated_participants))
    ].copy()

    if len(frame) != len(scores):
        raise SystemExit(
            f"window count mismatch: {len(frame)} metadata rows against "
            f"{len(scores)} saved scores"
        )

    frame = frame.reset_index(drop=True)
    frame["score"] = scores
    frame["label"] = (frame["y_true"] > ELEVATED_THRESHOLD).astype(int)
    frame = frame[
        [
            "participant_id",
            "source_dataset",
            "condition",
            "start_index",
            "end_index",
            "y_true",
            "label",
            "score",
        ]
    ]
    return frame


def verify_alignment(frame: pd.DataFrame, headline_path: Path) -> None:
    """Fail loudly if the score/label pairing does not reproduce the headline."""
    headline = json.loads(headline_path.read_text())
    observed_auc = roc_auc(frame["label"].to_numpy(), frame["score"].to_numpy())
    expected_auc = headline["pooled_auc"]
    if abs(observed_auc - expected_auc) > AUC_TOLERANCE:
        raise SystemExit(
            "alignment check failed: recomputed pooled AUC "
            f"{observed_auc!r} does not match recorded {expected_auc!r}"
        )
    for key, observed in (
        ("passive_test_windows", len(frame)),
        ("elevated_windows", int(frame["label"].sum())),
        ("passive_test_participants", frame["participant_id"].nunique()),
    ):
        if headline[key] != observed:
            raise SystemExit(
                f"alignment check failed on {key}: {observed} against "
                f"recorded {headline[key]}"
            )


def confusion_at(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    predicted = scores >= threshold
    positive = labels == 1
    true_positive = int(np.sum(predicted & positive))
    false_positive = int(np.sum(predicted & ~positive))
    false_negative = int(np.sum(~predicted & positive))
    true_negative = int(np.sum(~predicted & ~positive))

    def ratio(numerator: int, denominator: int) -> float:
        return float(numerator / denominator) if denominator else float("nan")

    return {
        "threshold": float(threshold),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "sensitivity": ratio(true_positive, true_positive + false_negative),
        "specificity": ratio(true_negative, true_negative + false_positive),
        "false_alarm_rate": ratio(false_positive, false_positive + true_negative),
        "precision": ratio(true_positive, true_positive + false_positive),
        "negative_predictive_value": ratio(
            true_negative, true_negative + false_negative
        ),
        "alert_rate": ratio(true_positive + false_positive, len(labels)),
        "balanced_accuracy": 0.5
        * (
            ratio(true_positive, true_positive + false_negative)
            + ratio(true_negative, true_negative + false_positive)
        ),
        "youden_j": ratio(true_positive, true_positive + false_negative)
        + ratio(true_negative, true_negative + false_positive)
        - 1.0,
    }


def threshold_sweep(frame: pd.DataFrame) -> pd.DataFrame:
    labels = frame["label"].to_numpy()
    scores = frame["score"].to_numpy()
    candidates = np.unique(scores)
    rows = [confusion_at(labels, scores, threshold) for threshold in candidates]
    return pd.DataFrame(rows)


def cluster_bootstrap_operating_point(
    frame: pd.DataFrame,
    threshold: float,
    replicates: int = 5000,
    seed: int = 42,
) -> dict:
    """Participant-cluster interval for sensitivity, specificity and alarm rate."""
    rng = np.random.RandomState(seed)
    groups = [group for _, group in frame.groupby("participant_id", sort=True)]
    sensitivity: list[float] = []
    specificity: list[float] = []
    false_alarm: list[float] = []

    for _ in range(replicates):
        chosen = rng.randint(0, len(groups), size=len(groups))
        sample = pd.concat([groups[index] for index in chosen], ignore_index=True)
        labels = sample["label"].to_numpy()
        if labels.sum() == 0 or labels.sum() == len(labels):
            continue
        point = confusion_at(labels, sample["score"].to_numpy(), threshold)
        sensitivity.append(point["sensitivity"])
        specificity.append(point["specificity"])
        false_alarm.append(point["false_alarm_rate"])

    def interval(values: list[float]) -> dict:
        array = np.asarray(values, dtype=float)
        return {
            "lower": float(np.percentile(array, 2.5)),
            "upper": float(np.percentile(array, 97.5)),
        }

    return {
        "threshold": float(threshold),
        "bootstrap_replicates": len(sensitivity),
        "sensitivity": interval(sensitivity),
        "specificity": interval(specificity),
        "false_alarm_rate": interval(false_alarm),
    }


def select_operating_points(sweep: pd.DataFrame) -> dict:
    """Pick reference thresholds a deployment discussion would actually use."""
    selected: dict[str, float] = {}

    selected["default_half"] = 0.5
    selected["youden"] = float(sweep.loc[sweep["youden_j"].idxmax(), "threshold"])

    low_alarm = sweep[sweep["specificity"] >= 0.90]
    if len(low_alarm):
        selected["specificity_90"] = float(low_alarm["threshold"].min())

    high_recall = sweep[sweep["sensitivity"] >= 0.80]
    if len(high_recall):
        selected["sensitivity_80"] = float(high_recall["threshold"].max())

    return selected


def calibration_table(frame: pd.DataFrame, bins: int = 8) -> pd.DataFrame:
    """Reliability table over equal-count score bins."""
    scores = frame["score"].to_numpy()
    labels = frame["label"].to_numpy()
    edges = np.quantile(scores, np.linspace(0.0, 1.0, bins + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    assignment = np.digitize(scores, edges[1:-1], right=True)

    rows = []
    for index in range(bins):
        mask = assignment == index
        if not mask.any():
            continue
        rows.append(
            {
                "bin": index,
                "lower_edge": float(edges[index]),
                "upper_edge": float(edges[index + 1]),
                "n_windows": int(mask.sum()),
                "mean_score": float(scores[mask].mean()),
                "observed_rate": float(labels[mask].mean()),
            }
        )
    return pd.DataFrame(rows)


def calibration_summary(frame: pd.DataFrame, table: pd.DataFrame) -> dict:
    scores = frame["score"].to_numpy()
    labels = frame["label"].to_numpy()
    weights = table["n_windows"].to_numpy() / table["n_windows"].sum()
    gaps = np.abs(table["mean_score"].to_numpy() - table["observed_rate"].to_numpy())
    return {
        "brier_score": float(np.mean((scores - labels) ** 2)),
        "mean_predicted_probability": float(scores.mean()),
        "observed_prevalence": float(labels.mean()),
        "expected_calibration_error": float(np.sum(weights * gaps)),
        "score_minimum": float(scores.min()),
        "score_maximum": float(scores.max()),
    }


def participant_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for participant, group in frame.groupby("participant_id", sort=True):
        labels = group["label"].to_numpy()
        scores = group["score"].to_numpy()
        both = 0 < labels.sum() < len(labels)
        rows.append(
            {
                "participant_id": participant,
                "source_dataset": group["source_dataset"].iloc[0],
                "n_windows": len(group),
                "elevated_windows": int(labels.sum()),
                "prevalence": float(labels.mean()),
                "within_participant_auc": roc_auc(labels, scores) if both else np.nan,
                "average_precision": average_precision(labels, scores)
                if labels.sum()
                else np.nan,
                "mean_score": float(scores.mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--replicates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    root = args.repo_root
    output_dir = args.output_dir or (root / "results" / "corrected")
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = load_aligned_frame(
        root / "results" / "corrected" / "cnn_ensemble_preds_OG.npy",
        root / "results" / "colab" / "full" / "architecture_test_predictions.csv",
        root / "results" / "corrected" / "per_participant_metrics.csv",
    )
    verify_alignment(frame, root / "results" / "corrected" / "corrected_headline_metrics.json")

    sweep = threshold_sweep(frame)
    sweep.to_csv(output_dir / "operating_point_sweep.csv", index=False)

    selected = select_operating_points(sweep)
    labels = frame["label"].to_numpy()
    scores = frame["score"].to_numpy()

    points = []
    for name, threshold in selected.items():
        point = confusion_at(labels, scores, threshold)
        point["name"] = name
        point.update(
            {
                "interval": cluster_bootstrap_operating_point(
                    frame, threshold, args.replicates, args.seed
                )
            }
        )
        points.append(point)

    flat = []
    for point in points:
        interval = point.pop("interval")
        row = dict(point)
        row["sensitivity_lower"] = interval["sensitivity"]["lower"]
        row["sensitivity_upper"] = interval["sensitivity"]["upper"]
        row["specificity_lower"] = interval["specificity"]["lower"]
        row["specificity_upper"] = interval["specificity"]["upper"]
        row["false_alarm_lower"] = interval["false_alarm_rate"]["lower"]
        row["false_alarm_upper"] = interval["false_alarm_rate"]["upper"]
        row["bootstrap_replicates"] = interval["bootstrap_replicates"]
        flat.append(row)
    operating = pd.DataFrame(flat)
    ordered = ["name", "threshold", "sensitivity", "specificity", "false_alarm_rate"]
    operating = operating[ordered + [c for c in operating.columns if c not in ordered]]
    operating.to_csv(output_dir / "operating_points.csv", index=False)

    calibration = calibration_table(frame)
    calibration.to_csv(output_dir / "calibration_bins.csv", index=False)

    participants = participant_table(frame)
    participants.to_csv(output_dir / "participant_diagnostics.csv", index=False)

    summary = {
        "n_windows": int(len(frame)),
        "n_participants": int(frame["participant_id"].nunique()),
        "elevated_windows": int(labels.sum()),
        "prevalence": float(labels.mean()),
        "pooled_auc": roc_auc(labels, scores),
        "average_precision": average_precision(labels, scores),
        "calibration": calibration_summary(frame, calibration),
        "operating_points": {
            row["name"]: {
                "threshold": row["threshold"],
                "sensitivity": row["sensitivity"],
                "specificity": row["specificity"],
                "false_alarm_rate": row["false_alarm_rate"],
                "precision": row["precision"],
                "alert_rate": row["alert_rate"],
                "sensitivity_ci": [row["sensitivity_lower"], row["sensitivity_upper"]],
                "specificity_ci": [row["specificity_lower"], row["specificity_upper"]],
                "false_alarm_ci": [row["false_alarm_lower"], row["false_alarm_upper"]],
            }
            for _, row in operating.iterrows()
        },
        "bootstrap_seed": args.seed,
    }
    (output_dir / "operating_point_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
