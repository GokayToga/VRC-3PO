#!/usr/bin/env python3
"""Does the residual sentinel artefact change the single-task headline result?

The published table retains 438 rows with negative pupil values, an artefact of
an invalid-marker test applied after one-second averaging. 79 of the 768 single-task
held-out test windows contain at least one such row, so the artefact does enter
the reported single-task AUC. This script measures how much.

It retrains the five-model single-task CNN ensemble on the *frozen* participant split
(seed 42) and reports pooled AUC with a participant-cluster interval. Run it on
two tables built from the same pooled source -- one with the published masking
rule, one with ``--strict-sentinel`` -- so the only difference between the runs
is the masking rule. Comparing a locally trained run against the published
0.757 would confound the artefact with library and hardware differences;
comparing two local runs does not.

    python -m analysis.sentinel_sensitivity --dataset rebuilt_fixed.csv --label default
    python -m analysis.sentinel_sensitivity --dataset strict_fixed.csv  --label strict
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.composition_robustness import (
    ENSEMBLE_SEEDS,
    balanced_class_weights,
    build_cnn_classifier,
    build_windows,
    cluster_bootstrap_auc,
    standardize,
)
from analysis.corrected_evaluation import (
    ELEVATED_THRESHOLD,
    PASSIVE_SOURCES,
    average_precision,
    participant_split,
    roc_auc,
)


def run(dataset: Path, epochs: int, bootstrap: int, seed: int) -> dict:
    import tensorflow as tf

    frame = pd.read_csv(dataset)
    x, labels, metadata = build_windows(frame)

    # The frozen split, not a redrawn composition.
    train_ids, validation_ids, test_ids = participant_split(labels, metadata, seed=42)
    participant = metadata["participant_id"]
    is_seated = metadata["source_dataset"].isin(PASSIVE_SOURCES)

    train_index = np.flatnonzero(participant.isin(train_ids) & is_seated)
    validation_index = np.flatnonzero(participant.isin(validation_ids) & is_seated)
    test_index = np.flatnonzero(participant.isin(test_ids) & is_seated)

    binary = (labels > ELEVATED_THRESHOLD).astype(int)
    x_train, (x_validation, x_test) = standardize(
        x, train_index, [validation_index, test_index]
    )
    y_train = binary[train_index]
    y_validation = binary[validation_index]
    y_test = binary[test_index]
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
            x_train, y_train,
            validation_data=(x_validation, y_validation),
            class_weight=weights, epochs=epochs, batch_size=32, verbose=0,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_auc", mode="max", patience=12,
                    min_delta=1e-4, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_auc", mode="max", factor=0.5, patience=5, min_lr=1e-6),
                tf.keras.callbacks.TerminateOnNaN(),
            ],
        )
        member_scores.append(model.predict(x_test, verbose=0).reshape(-1))

    ensemble = np.mean(np.stack(member_scores), axis=0)
    participant_ids = metadata.iloc[test_index]["participant_id"].to_numpy()
    interval = cluster_bootstrap_auc(y_test, ensemble, participant_ids, bootstrap, seed)

    negative = (
        (frame["pupil_diam_L"] < 0) | (frame["pupil_diam_R"] < 0)
    ).sum()

    return {
        "dataset": str(dataset),
        "negative_pupil_rows": int(negative),
        "train_windows": int(len(train_index)),
        "test_windows": int(len(test_index)),
        "test_participants": int(len(np.unique(participant_ids))),
        "elevated_windows": int(y_test.sum()),
        "pooled_auc": roc_auc(y_test, ensemble),
        "average_precision": average_precision(y_test, ensemble),
        "cluster_ci_lower": interval["lower"],
        "cluster_ci_upper": interval["upper"],
        "member_aucs": [roc_auc(y_test, s) for s in member_scores],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/sentinel_check"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    record = run(args.dataset, args.epochs, args.bootstrap, args.seed)
    record["label"] = args.label
    destination = args.output_dir / f"{args.label}.json"
    destination.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
