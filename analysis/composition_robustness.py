#!/usr/bin/env python3
"""Composition-robustness re-run for the single-task binary endpoint.

WHY THIS EXISTS
---------------
The single-task headline (pooled AUC 0.757) rests on one participant assignment,
produced by seed 42. A single split cannot say whether that number is
representative or fortunate. This module repeats the entire pipeline --- fresh
participant assignment, fresh standardization fitted on that assignment's own
training split, retrained five-CNN ensemble, re-evaluation on the single-task part
of that assignment's own test set --- across many compositions, and records the
resulting distribution.

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
The compositions redraw from one shared pool of 84 participants, so any two of
them overlap heavily in training data and may share test participants. Their
AUC values are therefore dependent. This module reports the observed
distribution and refuses to compute a one-sample t-test against 0.5. The
accompanying manuscript text must not call the compositions independent. See
``manuscript/revision_map.md``.

USAGE
-----
    python -m analysis.composition_robustness \\
        --dataset /path/to/vrc3po_master_dataset_fixed.csv \\
        --output-dir /path/to/vrc3po_composition_robustness

Each composition is written to its own JSON as soon as it finishes, so an
interrupted run resumes without repeating completed work.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.corrected_evaluation import (
    ELEVATED_THRESHOLD,
    PASSIVE_SOURCES,
    average_precision,
    roc_auc,
)
from analysis.retrain_camera_ready import FEATURE_COLUMNS, WINDOW_LENGTH, WINDOW_STRIDE

ENSEMBLE_SEEDS = (42, 123, 456, 789, 2024)
DEFAULT_COMPOSITIONS = 20
DEFAULT_BOOTSTRAP = 5000
FROZEN_POOLED_AUC = 0.756947866036354

# Bump whenever the training protocol changes in a way that makes previously
# written composition files incomparable. The resume logic refuses to reuse a
# file written under a different value, so a protocol change cannot silently
# survive inside a half-finished output directory.
PROTOCOL_VERSION = 2  # v2: single-task-only train/validation/test, Masking layer restored


# ---------------------------------------------------------------------------
# Windowing and splitting
# ---------------------------------------------------------------------------
def build_windows(df: pd.DataFrame):
    """Same windowing rule as the frozen pipeline: 30 s windows, 15 s stride."""
    features, labels, rows = [], [], []
    grouped = df.groupby(["global_participant_id", "condition"], sort=False)
    for (participant, condition), session in grouped:
        session = session.sort_values("elapsed_s").reset_index(drop=True)
        session_features = session[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        session_fms = session["fms"].to_numpy(dtype=np.float32)
        source = str(session["source_dataset"].iloc[0])
        for start in range(0, len(session) - WINDOW_LENGTH + 1, WINDOW_STRIDE):
            end = start + WINDOW_LENGTH
            features.append(session_features[start:end])
            labels.append(float(session_fms[start:end].mean()))
            rows.append((str(participant), str(condition), source, start, end))
    metadata = pd.DataFrame(
        rows,
        columns=[
            "participant_id",
            "condition",
            "source_dataset",
            "start_index",
            "end_index",
        ],
    )
    return (
        np.stack(features).astype(np.float32),
        np.asarray(labels, dtype=np.float32),
        metadata,
    )


def participant_strata(labels: np.ndarray, metadata: pd.DataFrame) -> pd.DataFrame:
    """Source folder crossed with a within-folder severity tier."""
    elevated = (labels > ELEVATED_THRESHOLD).astype(int)
    severity = (
        pd.DataFrame(
            {"participant_id": metadata["participant_id"], "elevated": elevated}
        )
        .groupby("participant_id")["elevated"]
        .mean()
        .reset_index(name="elevated_rate")
    )
    info = (
        metadata.groupby("participant_id")["source_dataset"]
        .first()
        .reset_index()
        .merge(severity, on="participant_id")
    )
    info["rank"] = info.groupby("source_dataset")["elevated_rate"].rank(method="first")
    info["severity_tier"] = info.groupby("source_dataset")["rank"].transform(
        lambda values: pd.qcut(values, 2, labels=["low", "high"])
    )
    info["stratum"] = info["source_dataset"] + "_" + info["severity_tier"].astype(str)
    return info


def stratified_participant_split(info: pd.DataFrame, seed: int):
    """Draw one composition: 20 % test, then 12.5 % of the remainder to validation.

    Proportions match the frozen split. Stratification is by source folder
    crossed with severity tier, so every composition keeps all three settings
    represented in each partition.
    """
    rng = np.random.default_rng(seed)
    train_ids, validation_ids, test_ids = [], [], []
    for _, group in info.groupby("stratum", sort=True):
        members = group["participant_id"].to_numpy()
        members = members[rng.permutation(len(members))]
        n_test = max(1, int(round(0.20 * len(members))))
        n_validation = max(1, int(round(0.125 * (len(members) - n_test))))
        test_ids.extend(members[:n_test])
        validation_ids.extend(members[n_test : n_test + n_validation])
        train_ids.extend(members[n_test + n_validation :])
    return set(train_ids), set(validation_ids), set(test_ids)


def standardize(x: np.ndarray, train_index: np.ndarray, other_indices: list):
    train_flat = x[train_index].reshape(-1, x.shape[-1]).astype(np.float64)
    mean = train_flat.mean(axis=0)
    scale = train_flat.std(axis=0)
    scale[scale == 0] = 1.0

    def transform(index: np.ndarray) -> np.ndarray:
        return ((x[index].astype(np.float64) - mean) / scale).astype(np.float32)

    return transform(train_index), [transform(i) for i in other_indices]


# ---------------------------------------------------------------------------
# Model and uncertainty
# ---------------------------------------------------------------------------
def build_cnn_classifier(tf):
    """The CNN used as the base of the frozen five-model single-task ensemble.

    This mirrors ``build_cnn_binary`` from the original notebook exactly,
    including the leading Masking layer. The mask is very nearly inert on
    standardized inputs, but it is kept so that this re-run differs from the
    frozen endpoint in participant composition alone.
    """
    layers = tf.keras.layers
    inp = layers.Input(shape=(WINDOW_LENGTH, len(FEATURE_COLUMNS)))
    x = layers.Masking(mask_value=0.0)(inp)
    x = layers.Conv1D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(8, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inp, out)


def balanced_class_weights(binary_labels: np.ndarray) -> dict:
    counts = np.bincount(binary_labels, minlength=2).astype(float)
    counts[counts == 0] = 1.0
    total = float(len(binary_labels))
    return {0: total / (2.0 * counts[0]), 1: total / (2.0 * counts[1])}


def _fast_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Vectorized ROC AUC via the rank identity, with midranks for ties.

    Mathematically identical to ``analysis.corrected_evaluation.roc_auc`` --
    both give tied pairs half credit -- but without the Python loop over tie
    groups, which dominates bootstrap cost. Equality with the reference
    implementation is asserted in ``tests/test_composition_robustness.py``.
    """
    n_positive = int(y_true.sum())
    n_negative = len(y_true) - n_positive
    if n_positive == 0 or n_negative == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    boundaries = np.flatnonzero(np.diff(sorted_scores)) + 1
    for end in np.append(boundaries, len(scores)):
        ranks[order[start:end]] = 0.5 * (start + end + 1)
        start = end
    return float(
        (ranks[y_true == 1].sum() - n_positive * (n_positive + 1) / 2.0)
        / (n_positive * n_negative)
    )


def cluster_bootstrap_auc(y_true, scores, participant_ids, replicates, seed):
    """Resample whole participants, keeping every window of each drawn person."""
    rng = np.random.default_rng(seed)
    participants = np.unique(participant_ids)
    rows_for = {p: np.flatnonzero(participant_ids == p) for p in participants}
    estimates = []
    for _ in range(replicates):
        sampled = rng.choice(participants, size=len(participants), replace=True)
        rows = np.concatenate([rows_for[p] for p in sampled])
        estimate = _fast_auc(y_true[rows], scores[rows])
        if np.isfinite(estimate):
            estimates.append(estimate)
    if not estimates:
        return {"lower": float("nan"), "upper": float("nan"), "replicates": 0}
    return {
        "lower": float(np.percentile(estimates, 2.5)),
        "upper": float(np.percentile(estimates, 97.5)),
        "replicates": len(estimates),
    }


# ---------------------------------------------------------------------------
# One composition
# ---------------------------------------------------------------------------
def run_one_composition(tf, x, labels, metadata, seed, epochs=100, bootstrap=DEFAULT_BOOTSTRAP):
    info = participant_strata(labels, metadata)
    train_ids, validation_ids, test_ids = stratified_participant_split(info, seed)

    participant = metadata["participant_id"]
    is_seated = metadata["source_dataset"].isin(PASSIVE_SOURCES)

    # The frozen single-task endpoint trains, validates and tests on single-task windows
    # only; Maze is excluded from all three partitions. Every partition is
    # masked the same way here so that this re-run differs from the frozen
    # result in participant composition alone. Training on all three settings
    # would confound composition variance with the task-context effect that
    # the paper reports separately.
    train_index = np.flatnonzero(participant.isin(train_ids) & is_seated)
    validation_index = np.flatnonzero(participant.isin(validation_ids) & is_seated)
    test_index = np.flatnonzero(participant.isin(test_ids) & is_seated)

    binary = (labels > ELEVATED_THRESHOLD).astype(int)
    y_test = binary[test_index]
    if len(test_index) == 0 or y_test.sum() == 0 or y_test.sum() == len(y_test):
        return {
            "composition_seed": int(seed),
            "protocol_version": PROTOCOL_VERSION,
            "status": "skipped_single_class_test",
            "test_windows": int(len(test_index)),
            "test_participants": int(
                metadata.iloc[test_index]["participant_id"].nunique()
            ),
        }

    x_train, (x_validation, x_test) = standardize(
        x, train_index, [validation_index, test_index]
    )
    y_train = binary[train_index]
    y_validation = binary[validation_index]
    weights = balanced_class_weights(y_train)

    member_scores = []
    for member_seed in ENSEMBLE_SEEDS:
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(member_seed)
        model = build_cnn_classifier(tf)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4, clipnorm=1.0),
            loss="binary_crossentropy",
            metrics=[tf.keras.metrics.AUC(name="auc")],
        )
        model.fit(
            x_train,
            y_train,
            validation_data=(x_validation, y_validation),
            class_weight=weights,
            epochs=epochs,
            batch_size=32,
            verbose=0,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_auc",
                    mode="max",
                    patience=12,
                    min_delta=1e-4,
                    restore_best_weights=True,
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_auc", mode="max", factor=0.5, patience=5, min_lr=1e-6
                ),
                tf.keras.callbacks.TerminateOnNaN(),
            ],
        )
        member_scores.append(model.predict(x_test, verbose=0).reshape(-1))

    ensemble = np.mean(np.stack(member_scores), axis=0)
    participant_ids = metadata.iloc[test_index]["participant_id"].to_numpy()
    interval = cluster_bootstrap_auc(y_test, ensemble, participant_ids, bootstrap, seed)

    per_participant = []
    for pid in np.unique(participant_ids):
        rows = participant_ids == pid
        per_participant.append(
            {
                "participant_id": str(pid),
                "windows": int(rows.sum()),
                "elevated": int(y_test[rows].sum()),
                "auc": roc_auc(y_test[rows], ensemble[rows]),
            }
        )

    return {
        "composition_seed": int(seed),
        "protocol_version": PROTOCOL_VERSION,
        "status": "complete",
        "train_windows": int(len(train_index)),
        "validation_windows": int(len(validation_index)),
        "test_participants": int(len(np.unique(participant_ids))),
        "test_windows": int(len(test_index)),
        "elevated_windows": int(y_test.sum()),
        "prevalence": float(y_test.mean()),
        "pooled_auc": roc_auc(y_test, ensemble),
        "average_precision": average_precision(y_test, ensemble),
        "cluster_ci_lower": interval["lower"],
        "cluster_ci_upper": interval["upper"],
        "cluster_replicates": interval["replicates"],
        "member_aucs": [roc_auc(y_test, s) for s in member_scores],
        "per_participant": per_participant,
        "train_participants": int(len(train_ids)),
        "validation_participants": int(len(validation_ids)),
    }


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def summarize(records: list) -> dict:
    """Distributional summary only. No significance test: see module docstring."""
    complete = [r for r in records if r.get("status") == "complete"]
    if not complete:
        return {"n_compositions_attempted": len(records), "n_compositions_complete": 0}
    aucs = np.asarray([r["pooled_auc"] for r in complete], dtype=float)
    below = int(np.sum(aucs < FROZEN_POOLED_AUC))
    return {
        "n_compositions_attempted": len(records),
        "n_compositions_complete": len(complete),
        "mean_auc": float(aucs.mean()),
        "sd_auc": float(aucs.std(ddof=1)) if len(aucs) > 1 else float("nan"),
        "median_auc": float(np.median(aucs)),
        "min_auc": float(aucs.min()),
        "max_auc": float(aucs.max()),
        "range_auc": float(aucs.max() - aucs.min()),
        "n_with_ci_above_chance": int(
            sum(1 for r in complete if r["cluster_ci_lower"] > 0.5)
        ),
        "n_with_point_estimate_above_chance": int(np.sum(aucs > 0.5)),
        "frozen_split_auc": FROZEN_POOLED_AUC,
        "frozen_split_percentile": float(100.0 * below / len(aucs)),
        "n_compositions_below_frozen": below,
        "mean_test_participants": float(
            np.mean([r["test_participants"] for r in complete])
        ),
        "note": (
            "Compositions share one pool of 84 participants and are therefore "
            "dependent. Report the distribution; do not apply a one-sample "
            "test against 0.5 and do not describe the compositions as "
            "independent."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--compositions", type=int, default=DEFAULT_COMPOSITIONS)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP)
    args = parser.parse_args()

    import tensorflow as tf

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Rebuilding windows from the 31 MB CSV costs far more than one composition
    # fit. Cache them next to the results so an interrupted run resumes cheaply.
    cache = args.output_dir / "window_cache.npz"
    if cache.exists():
        blob = np.load(cache, allow_pickle=True)
        x = blob["x"]
        labels = blob["labels"]
        metadata = pd.DataFrame(blob["metadata"], columns=list(blob["columns"]))
        metadata["start_index"] = metadata["start_index"].astype(int)
        metadata["end_index"] = metadata["end_index"].astype(int)
        print(f"[cache] loaded windows from {cache}", flush=True)
    else:
        frame = pd.read_csv(args.dataset)
        x, labels, metadata = build_windows(frame)
        np.savez_compressed(
            cache,
            x=x,
            labels=labels,
            metadata=metadata.to_numpy(dtype=object),
            columns=np.asarray(metadata.columns),
        )
        print(f"[cache] wrote {cache}", flush=True)

    print(
        f"windows={len(labels)} "
        f"participants={metadata['participant_id'].nunique()} "
        f"elevated={float((labels > ELEVATED_THRESHOLD).mean()):.4f}",
        flush=True,
    )

    seeds = range(args.start_seed, args.start_seed + args.compositions)
    records = []
    for seed in seeds:
        path = args.output_dir / f"composition_{seed:03d}.json"
        if path.exists():
            existing = json.loads(path.read_text())
            if existing.get("protocol_version") != PROTOCOL_VERSION:
                raise SystemExit(
                    f"{path} was written under protocol version "
                    f"{existing.get('protocol_version')}, but this code is "
                    f"version {PROTOCOL_VERSION}. Those results are not "
                    "comparable. Delete the output directory and start again "
                    "rather than mixing them."
                )
            print(f"[resume] composition {seed} already complete", flush=True)
            records.append(existing)
            continue
        print(f"[run] composition {seed}", flush=True)
        record = run_one_composition(
            tf, x, labels, metadata, seed, args.epochs, args.bootstrap
        )
        path.write_text(json.dumps(record, indent=2) + "\n")
        records.append(record)
        print(
            f"      {record['status']} auc={record.get('pooled_auc')} "
            f"test_participants={record.get('test_participants')}",
            flush=True,
        )

    summary = summarize(records)
    (args.output_dir / "composition_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    pd.DataFrame(
        [
            {k: v for k, v in r.items() if k not in ("per_participant", "member_aucs")}
            for r in records
        ]
    ).to_csv(args.output_dir / "composition_results.csv", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
