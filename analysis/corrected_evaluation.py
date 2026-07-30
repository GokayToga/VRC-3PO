#!/usr/bin/env python3
"""Recompute clustered headline metrics for the VRC-3PO manuscript.

This module intentionally evaluates saved predictions without retraining.  It:

1. recreates the manuscript's participant split;
2. verifies that the prediction vector has the expected length;
3. reports the actual number of single-task test participants;
4. computes pooled, participant-equal, and participant-macro metrics; and
5. uses participants, rather than overlapping windows, as bootstrap clusters.

The output files are suitable for direct use when rebuilding the manuscript.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


WINDOW_LENGTH = 30
WINDOW_STRIDE = 15
ELEVATED_THRESHOLD = 2.0
PASSIVE_SOURCES = frozenset({"simulations", "terrain"})


@dataclass(frozen=True)
class Interval:
    estimate: float
    lower: float
    upper: float
    bootstrap_replicates: int


def roc_auc(y_true: np.ndarray, scores: np.ndarray, weights: np.ndarray | None = None) -> float:
    """Weighted ROC AUC with tie handling and no sklearn dependency."""
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    w = np.ones(len(y), dtype=float) if weights is None else np.asarray(weights, dtype=float)
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    if not len(positive) or not len(negative):
        return float("nan")

    order = np.argsort(s, kind="mergesort")
    sorted_scores = s[order]
    sorted_y = y[order]
    sorted_w = w[order]

    concordant_weight = 0.0
    negative_weight_before = 0.0
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        group_positive = sorted_w[i:j][sorted_y[i:j] == 1].sum()
        group_negative = sorted_w[i:j][sorted_y[i:j] == 0].sum()
        concordant_weight += group_positive * (
            negative_weight_before + 0.5 * group_negative
        )
        negative_weight_before += group_negative
        i = j

    return float(concordant_weight / (w[positive].sum() * w[negative].sum()))


def average_precision(
    y_true: np.ndarray, scores: np.ndarray, weights: np.ndarray | None = None
) -> float:
    """Weighted non-interpolated average precision, matching sklearn semantics."""
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    w = np.ones(len(y), dtype=float) if weights is None else np.asarray(weights, dtype=float)
    total_positive = w[y == 1].sum()
    if total_positive == 0:
        return float("nan")

    order = np.argsort(-s, kind="mergesort")
    sorted_scores = s[order]
    sorted_y = y[order]
    sorted_w = w[order]

    true_positive = 0.0
    false_positive = 0.0
    result = 0.0
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        added_positive = sorted_w[i:j][sorted_y[i:j] == 1].sum()
        added_negative = sorted_w[i:j][sorted_y[i:j] == 0].sum()
        true_positive += added_positive
        false_positive += added_negative
        precision = true_positive / (true_positive + false_positive)
        result += precision * (added_positive / total_positive)
        i = j
    return float(result)


def _approximate_mode(
    class_counts: np.ndarray, draws: int, rng: np.random.RandomState
) -> np.ndarray:
    """Port of sklearn's allocation helper used by StratifiedShuffleSplit."""
    continuous = class_counts / class_counts.sum() * draws
    allocated = np.floor(continuous).astype(int)
    need = int(draws - allocated.sum())
    if need:
        remainder = continuous - allocated
        for value in np.sort(np.unique(remainder))[::-1]:
            candidates = np.flatnonzero(remainder == value)
            take = min(len(candidates), need)
            if take:
                allocated[rng.choice(candidates, size=take, replace=False)] += 1
                need -= take
            if need == 0:
                break
    return allocated


def stratified_train_test_indices(
    labels: Iterable[str], test_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce sklearn's one-split stratified shuffle behavior."""
    labels_array = np.asarray(list(labels))
    indices = np.arange(len(labels_array))
    n_test = int(np.ceil(test_fraction * len(indices)))
    n_train = len(indices) - n_test
    rng = np.random.RandomState(seed)

    classes, inverse = np.unique(labels_array, return_inverse=True)
    counts = np.bincount(inverse)
    class_indices = np.split(
        np.argsort(inverse, kind="mergesort"), np.cumsum(counts)[:-1]
    )
    train_counts = _approximate_mode(counts, n_train, rng)
    test_counts = _approximate_mode(counts - train_counts, n_test, rng)

    train: list[int] = []
    test: list[int] = []
    for class_index in range(len(classes)):
        permuted = class_indices[class_index][rng.permutation(counts[class_index])]
        train.extend(permuted[: train_counts[class_index]])
        start = train_counts[class_index]
        test.extend(permuted[start : start + test_counts[class_index]])

    return (
        indices[rng.permutation(train)],
        indices[rng.permutation(test)],
    )


def build_window_metadata(df: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """Build labels and metadata in the same order as the manuscript notebook."""
    labels: list[float] = []
    rows: list[tuple[str, str, str, int, int]] = []
    grouped = df.groupby(["global_participant_id", "condition"], sort=False)
    for (participant, condition), session in grouped:
        session = session.sort_values("elapsed_s").reset_index(drop=True)
        fms = session["fms"].to_numpy()
        source = str(session["source_dataset"].iloc[0])
        for start in range(0, len(session) - WINDOW_LENGTH + 1, WINDOW_STRIDE):
            end = start + WINDOW_LENGTH
            labels.append(float(fms[start:end].mean()))
            rows.append((str(participant), str(condition), source, start, end))
    metadata = pd.DataFrame(
        rows,
        columns=["participant_id", "condition", "source_dataset", "start_index", "end_index"],
    )
    return np.asarray(labels), metadata


def participant_split(
    labels: np.ndarray, metadata: pd.DataFrame, seed: int = 42
) -> tuple[set[str], set[str], set[str]]:
    elevated = (labels > ELEVATED_THRESHOLD).astype(int)
    severity = (
        pd.DataFrame(
            {
                "participant_id": metadata["participant_id"],
                "elevated": elevated,
            }
        )
        .groupby("participant_id")["elevated"]
        .mean()
        .reset_index(name="elevated_rate")
    )
    participant_info = (
        metadata.groupby("participant_id")["source_dataset"]
        .first()
        .reset_index()
        .merge(severity, on="participant_id")
    )
    participant_info["rank"] = participant_info.groupby("source_dataset")[
        "elevated_rate"
    ].rank(method="first")
    participant_info["severity_tier"] = participant_info.groupby("source_dataset")[
        "rank"
    ].transform(lambda values: pd.qcut(values, 2, labels=["low", "high"]))
    participant_info["stratum"] = (
        participant_info["source_dataset"]
        + "_"
        + participant_info["severity_tier"].astype(str)
    )

    train_index, test_index = stratified_train_test_indices(
        participant_info["stratum"], test_fraction=0.20, seed=seed
    )
    inner_train, validation_relative = stratified_train_test_indices(
        participant_info.iloc[train_index]["stratum"],
        test_fraction=0.125,
        seed=seed,
    )
    validation_index = train_index[validation_relative]
    train_index = train_index[inner_train]
    return (
        set(participant_info.iloc[train_index]["participant_id"]),
        set(participant_info.iloc[validation_index]["participant_id"]),
        set(participant_info.iloc[test_index]["participant_id"]),
    )


def percentile_interval(values: list[float]) -> tuple[float, float]:
    return tuple(float(v) for v in np.percentile(values, [2.5, 97.5]))


def cluster_bootstrap_auc(
    y_true: np.ndarray,
    scores: np.ndarray,
    participant_ids: np.ndarray,
    replicates: int,
    seed: int,
) -> Interval:
    """Bootstrap participants and retain every window within selected clusters."""
    rng = np.random.default_rng(seed)
    participants = np.unique(participant_ids)
    participant_rows = {
        participant: np.flatnonzero(participant_ids == participant)
        for participant in participants
    }
    estimates: list[float] = []
    for _ in range(replicates):
        sampled = rng.choice(participants, size=len(participants), replace=True)
        rows = np.concatenate([participant_rows[p] for p in sampled])
        estimate = roc_auc(y_true[rows], scores[rows])
        if np.isfinite(estimate):
            estimates.append(estimate)
    lower, upper = percentile_interval(estimates)
    return Interval(
        estimate=roc_auc(y_true, scores),
        lower=lower,
        upper=upper,
        bootstrap_replicates=len(estimates),
    )


def macro_auc(
    y_true: np.ndarray, scores: np.ndarray, participant_ids: np.ndarray
) -> tuple[float, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for participant in np.unique(participant_ids):
        index = np.flatnonzero(participant_ids == participant)
        rows.append(
            {
                "participant_id": participant,
                "windows": len(index),
                "elevated_windows": int(y_true[index].sum()),
                "prevalence": float(y_true[index].mean()),
                "auc": roc_auc(y_true[index], scores[index]),
                "average_precision": average_precision(y_true[index], scores[index]),
            }
        )
    table = pd.DataFrame(rows)
    eligible = table["auc"].dropna()
    return float(eligible.mean()), table


def cluster_bootstrap_macro_auc(
    per_participant: pd.DataFrame, replicates: int, seed: int
) -> Interval:
    eligible = per_participant.dropna(subset=["auc"])["auc"].to_numpy()
    if not len(eligible):
        return Interval(float("nan"), float("nan"), float("nan"), 0)
    rng = np.random.default_rng(seed)
    draws = rng.choice(eligible, size=(replicates, len(eligible)), replace=True)
    values = draws.mean(axis=1)
    lower, upper = percentile_interval(values.tolist())
    return Interval(
        estimate=float(eligible.mean()),
        lower=lower,
        upper=upper,
        bootstrap_replicates=replicates,
    )


def evaluate(
    dataset_path: Path,
    predictions_path: Path,
    output_dir: Path,
    bootstrap_replicates: int,
    seed: int,
) -> dict[str, object]:
    df = pd.read_csv(dataset_path)
    labels, metadata = build_window_metadata(df)
    train_participants, validation_participants, test_participants = participant_split(
        labels, metadata, seed=42
    )
    passive_test = metadata["participant_id"].isin(test_participants) & metadata[
        "source_dataset"
    ].isin(PASSIVE_SOURCES)
    test_metadata = metadata.loc[passive_test].reset_index(drop=True)
    y_true = (labels[passive_test.to_numpy()] > ELEVATED_THRESHOLD).astype(int)
    scores = np.load(predictions_path).reshape(-1)
    if len(scores) != len(y_true):
        raise ValueError(
            f"Prediction length {len(scores)} does not match single-task test windows "
            f"{len(y_true)}. Check the split seed and prediction artifact."
        )

    participant_ids = test_metadata["participant_id"].to_numpy()
    participant_counts = test_metadata["participant_id"].value_counts()
    equal_participant_weights = test_metadata["participant_id"].map(
        lambda participant: 1.0 / participant_counts[participant]
    ).to_numpy()

    clustered = cluster_bootstrap_auc(
        y_true, scores, participant_ids, bootstrap_replicates, seed
    )
    macro_estimate, per_participant = macro_auc(y_true, scores, participant_ids)
    macro_clustered = cluster_bootstrap_macro_auc(
        per_participant, bootstrap_replicates, seed
    )
    per_participant = per_participant.merge(
        test_metadata.groupby("participant_id")["source_dataset"].first(),
        on="participant_id",
    )

    source_rows: list[dict[str, object]] = []
    for source, group in test_metadata.groupby("source_dataset"):
        index = group.index.to_numpy()
        source_rows.append(
            {
                "source_dataset": source,
                "participants": int(group["participant_id"].nunique()),
                "windows": len(index),
                "elevated_windows": int(y_true[index].sum()),
                "prevalence": float(y_true[index].mean()),
                "auc": roc_auc(y_true[index], scores[index]),
                "average_precision": average_precision(y_true[index], scores[index]),
            }
        )
    per_source = pd.DataFrame(source_rows)

    all_participants = (
        metadata.groupby("participant_id")["source_dataset"].first().reset_index()
    )
    all_participants["split"] = all_participants["participant_id"].map(
        lambda participant: (
            "train"
            if participant in train_participants
            else "validation"
            if participant in validation_participants
            else "test"
        )
    )
    all_participants["included_in_passive_headline"] = (
        all_participants["participant_id"].isin(test_metadata["participant_id"])
        & all_participants["source_dataset"].isin(PASSIVE_SOURCES)
    )

    summary: dict[str, object] = {
        "dataset_rows": len(df),
        "all_windows": len(labels),
        "all_test_participants": len(test_participants),
        "passive_test_participants": int(test_metadata["participant_id"].nunique()),
        "passive_test_windows": len(y_true),
        "elevated_windows": int(y_true.sum()),
        "prevalence": float(y_true.mean()),
        "pooled_auc": roc_auc(y_true, scores),
        "pooled_average_precision": average_precision(y_true, scores),
        "participant_equal_auc": roc_auc(
            y_true, scores, equal_participant_weights
        ),
        "participant_equal_average_precision": average_precision(
            y_true, scores, equal_participant_weights
        ),
        "cluster_bootstrap_auc": asdict(clustered),
        "participants_with_both_classes": int(per_participant["auc"].notna().sum()),
        "participant_macro_auc": macro_estimate,
        "cluster_bootstrap_macro_auc": asdict(macro_clustered),
        "split_seed": 42,
        "bootstrap_seed": seed,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    per_participant.to_csv(output_dir / "per_participant_metrics.csv", index=False)
    per_source.to_csv(output_dir / "per_source_metrics.csv", index=False)
    all_participants.to_csv(output_dir / "split_manifest.csv", index=False)
    pd.DataFrame(
        [
            {
                "model": "CNN ensemble (5 seeds)",
                "protocol": "participant-level, single-task sources",
                "participants": summary["passive_test_participants"],
                "windows": summary["passive_test_windows"],
                "elevated_windows": summary["elevated_windows"],
                "auc": summary["pooled_auc"],
                "auc_ci_lower": clustered.lower,
                "auc_ci_upper": clustered.upper,
                "ci_unit": "participant cluster",
                "average_precision": summary["pooled_average_precision"],
            }
        ]
    ).to_csv(output_dir / "table3_headline_corrected.csv", index=False)
    with (output_dir / "corrected_headline_metrics.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    replacement = f"""# Manuscript-ready corrections

## Headline result

The five-model CNN ensemble achieved a pooled AUC of {summary['pooled_auc']:.3f}
on {summary['passive_test_windows']} windows from
{summary['passive_test_participants']} unseen participants in the two passive
paradigms. Because windows overlapped and were clustered within participants,
uncertainty was estimated by resampling participants rather than individual
windows. The participant-cluster bootstrap 95% interval was
[{clustered.lower:.3f}, {clustered.upper:.3f}].

Among the {summary['participants_with_both_classes']} test participants who
contributed both elevated and non-elevated windows, mean participant-level AUC
was {macro_estimate:.3f} (participant-bootstrap 95% interval
[{macro_clustered.lower:.3f}, {macro_clustered.upper:.3f}]). The remaining
participants contributed only one outcome class and therefore had no defined
within-participant AUC.

## Required caption correction

Replace "17 unseen participants" in the single-task-only result and Figure 2
caption with "{summary['passive_test_participants']} unseen participants".
Seventeen is the size of the complete all-source test split, not the single-task
subset evaluated by the CNN ensemble.

## Interpretation constraint

The pooled result supports cross-participant discrimination in this test split.
The participant-level analysis should be reported alongside it because a pooled
window AUC alone can mix within-participant change with between-participant
differences.
"""
    (output_dir / "manuscript_replacements.md").write_text(replacement)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/corrected"))
    parser.add_argument("--bootstrap-replicates", type=int, default=5_000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = evaluate(
        dataset_path=args.dataset,
        predictions_path=args.predictions,
        output_dir=args.output_dir,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
