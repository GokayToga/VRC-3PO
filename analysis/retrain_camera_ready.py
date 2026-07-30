#!/usr/bin/env python3
"""Retrain the seven manuscript architectures from one frozen protocol.

This script is designed for the existing Colab/TensorFlow environment. It
generates seed-level metrics and predictions first, then derives Tables 4 and 5
from those files. The VRC-3PO row is therefore necessarily identical wherever
the same protocol is reported.

Example:

    python -m analysis.retrain_camera_ready \
      --dataset /content/drive/MyDrive/vrc3po_master_dataset_fixed.csv \
      --output-dir /content/drive/MyDrive/vrc3po_camera_ready
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from analysis.corrected_evaluation import (
    ELEVATED_THRESHOLD,
    participant_split,
)
from analysis.split_protocols import (
    assert_no_interval_overlap,
    chronological_purged_split,
)


FEATURE_COLUMNS = [
    "pupil_diam_L",
    "pupil_diam_R",
    "eye_open_L",
    "eye_open_R",
    "gaze_dir_world_X",
    "gaze_dir_world_Y",
    "gaze_dir_world_Z",
    "gaze_origin_world_X",
    "gaze_origin_world_Y",
    "gaze_origin_world_Z",
    "head_quat_X",
    "head_quat_Y",
    "head_quat_Z",
    "head_quat_W",
]
WINDOW_LENGTH = 30
WINDOW_STRIDE = 15
DEFAULT_SEEDS = (42, 123, 456)


def build_windows(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    features: list[np.ndarray] = []
    labels: list[float] = []
    rows: list[tuple[str, str, str, int, int]] = []
    grouped = df.groupby(["global_participant_id", "condition"], sort=False)
    for (participant, condition), session in grouped:
        session = session.sort_values("elapsed_s").reset_index(drop=True)
        session_features = session[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        session_fms = session["fms"].to_numpy(dtype=np.float32)
        source = str(session["source_dataset"].iloc[0])
        for start in range(
            0, len(session) - WINDOW_LENGTH + 1, WINDOW_STRIDE
        ):
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


def protocol_indices(
    labels: np.ndarray, metadata: pd.DataFrame
) -> dict[str, dict[str, np.ndarray]]:
    train_ids, validation_ids, test_ids = participant_split(
        labels, metadata, seed=42
    )
    participant_protocol = {
        "train": np.flatnonzero(metadata["participant_id"].isin(train_ids)),
        "validation": np.flatnonzero(
            metadata["participant_id"].isin(validation_ids)
        ),
        "test": np.flatnonzero(metadata["participant_id"].isin(test_ids)),
    }

    blocked_assignment = chronological_purged_split(
        metadata,
        train_fraction=0.60,
        validation_fraction=0.20,
        purge_samples=15,
    )
    assert_no_interval_overlap(metadata, blocked_assignment)
    blocked_protocol = {
        split: np.flatnonzero(blocked_assignment.to_numpy() == split)
        for split in ("train", "validation", "test")
    }
    return {
        "participant": participant_protocol,
        "blocked_within_participant": blocked_protocol,
    }


def standardize(
    x: np.ndarray, train_index: np.ndarray, other_indices: list[np.ndarray]
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, np.ndarray]:
    train_flat = x[train_index].reshape(-1, x.shape[-1]).astype(np.float64)
    mean = train_flat.mean(axis=0)
    scale = train_flat.std(axis=0)
    scale[scale == 0] = 1.0

    def transform(index: np.ndarray) -> np.ndarray:
        values = x[index].astype(np.float64)
        return ((values - mean) / scale).astype(np.float32)

    return transform(train_index), [transform(i) for i in other_indices], mean, scale


def balanced_regression_weights(y: np.ndarray) -> np.ndarray:
    classes = np.rint(y).astype(int)
    unique, counts = np.unique(classes, return_counts=True)
    weights = {
        value: len(classes) / (len(unique) * count)
        for value, count in zip(unique, counts)
    }
    return np.asarray([weights[value] for value in classes], dtype=np.float32)


def import_ml_dependencies():
    try:
        import tensorflow as tf
        from scipy.stats import spearmanr
        from sklearn.metrics import (
            cohen_kappa_score,
            mean_squared_error,
            r2_score,
            roc_auc_score,
        )
        from tcn import TCN
    except ImportError as error:
        raise SystemExit(
            "This training script requires TensorFlow, scipy, scikit-learn, "
            "and keras-tcn. Run it in the existing Colab environment after "
            "`pip install keras-tcn`."
        ) from error
    return (
        tf,
        TCN,
        spearmanr,
        cohen_kappa_score,
        mean_squared_error,
        r2_score,
        roc_auc_score,
    )


def architecture_builders(tf, TCN):
    layers = tf.keras.layers

    def head(sequence):
        x = layers.GlobalAveragePooling1D()(sequence)
        x = layers.Dense(8, activation="relu")(x)
        return layers.Dense(1, activation="linear")(x)

    def input_layer():
        return layers.Input(shape=(WINDOW_LENGTH, len(FEATURE_COLUMNS)))

    def mlp():
        inp = input_layer()
        x = layers.TimeDistributed(layers.Dense(64, activation="relu"))(inp)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.TimeDistributed(layers.Dense(32, activation="relu"))(x)
        return tf.keras.Model(inp, head(x), name="MLP")

    def cnn():
        inp = input_layer()
        x = layers.Conv1D(64, 3, padding="same", activation="relu")(inp)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.1)(x)
        return tf.keras.Model(inp, head(x), name="CNN")

    def cnn_gru():
        inp = input_layer()
        x = layers.Conv1D(64, 3, padding="same", activation="relu")(inp)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.1)(x)
        x = layers.GRU(64, return_sequences=True)(x)
        x = layers.Dropout(0.2)(x)
        x = layers.GRU(16, return_sequences=True)(x)
        x = layers.Dropout(0.1)(x)
        return tf.keras.Model(inp, head(x), name="CNN_GRU")

    def cnn_tcn():
        inp = input_layer()
        x = layers.Conv1D(64, 3, padding="same", activation="relu")(inp)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.1)(x)
        x = TCN(
            nb_filters=64,
            kernel_size=3,
            dilations=[1, 2, 4, 8],
            return_sequences=True,
            dropout_rate=0.2,
        )(x)
        x = layers.Dropout(0.2)(x)
        return tf.keras.Model(inp, head(x), name="CNN_TCN")

    def cnn_tcn_gru():
        inp = input_layer()
        x = layers.Conv1D(64, 3, padding="same", activation="relu")(inp)
        x = TCN(
            nb_filters=64,
            kernel_size=3,
            dilations=[1, 2, 4, 8],
            return_sequences=True,
            dropout_rate=0.2,
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.1)(x)
        x = layers.GRU(64, return_sequences=True)(x)
        x = layers.Dropout(0.2)(x)
        x = layers.GRU(16, return_sequences=True)(x)
        x = layers.Dropout(0.1)(x)
        return tf.keras.Model(inp, head(x), name="CNN_TCN_GRU")

    def vrc3po():
        inp = input_layer()
        x = layers.Conv1D(64, 3, padding="same", activation="relu")(inp)
        x = TCN(
            nb_filters=64,
            kernel_size=3,
            dilations=[1, 2, 4, 8],
            return_sequences=True,
            dropout_rate=0.2,
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.1)(x)
        x = layers.GRU(64, return_sequences=True)(x)
        x = layers.Dropout(0.2)(x)
        x = layers.GRU(16, return_sequences=True)(x)
        x = layers.Dropout(0.1)(x)
        attention = layers.MultiHeadAttention(num_heads=3, key_dim=32)(
            query=x, value=x, key=x
        )
        x = layers.Add()([x, attention])
        x = layers.LayerNormalization()(x)
        return tf.keras.Model(inp, head(x), name="VRC_3PO")

    def lite():
        inp = input_layer()
        x = layers.Conv1D(32, 3, padding="same", activation="relu")(inp)
        x = layers.BatchNormalization()(x)
        x = layers.GRU(32, return_sequences=True)(x)
        x = layers.GRU(16)(x)
        x = layers.Dense(8, activation="relu")(x)
        out = layers.Dense(1, activation="linear")(x)
        return tf.keras.Model(inp, out, name="Lite")

    return {
        "MLP": mlp,
        "CNN": cnn,
        "CNN+GRU": cnn_gru,
        "CNN+TCN": cnn_tcn,
        "CNN+TCN+GRU": cnn_tcn_gru,
        "VRC-3PO": vrc3po,
        "Lite": lite,
    }


def metric_row(
    y_true,
    prediction,
    protocol,
    variant,
    seed,
    parameter_count,
    spearmanr,
    cohen_kappa_score,
    mean_squared_error,
    r2_score,
    roc_auc_score,
):
    binary = (y_true > ELEVATED_THRESHOLD).astype(int)
    rounded_true = np.rint(y_true).astype(int)
    # Clip to the prespecified FMS range rather than to the observed test-label
    # range. Using rounded_true.min()/max() would let the test labels determine
    # prediction post-processing before the metric is computed.
    rounded_prediction = np.clip(
        np.rint(prediction), FMS_SCALE_MIN, FMS_SCALE_MAX
    ).astype(int)
    rho = spearmanr(y_true, prediction).statistic
    return {
        "protocol": protocol,
        "variant": variant,
        "seed": seed,
        "parameters": parameter_count,
        "test_windows": len(y_true),
        "r2": r2_score(y_true, prediction),
        "mse": mean_squared_error(y_true, prediction),
        "spearman": rho,
        "qwk": cohen_kappa_score(
            rounded_true, rounded_prediction, weights="quadratic"
        ),
        "auc": roc_auc_score(binary, prediction),
    }


def generate_tables(metrics: pd.DataFrame, output_dir: Path) -> None:
    grouped = (
        metrics.groupby(["protocol", "variant"])
        .agg(
            parameters=("parameters", "first"),
            r2_mean=("r2", "mean"),
            r2_sd=("r2", "std"),
            mse_mean=("mse", "mean"),
            mse_sd=("mse", "std"),
            spearman_mean=("spearman", "mean"),
            spearman_sd=("spearman", "std"),
            qwk_mean=("qwk", "mean"),
            qwk_sd=("qwk", "std"),
            auc_mean=("auc", "mean"),
            auc_sd=("auc", "std"),
        )
        .reset_index()
    )
    grouped.to_csv(output_dir / "architecture_summary.csv", index=False)

    vrc = grouped[grouped["variant"] == "VRC-3PO"].copy()
    vrc.to_csv(output_dir / "table4_protocol_comparison.csv", index=False)

    table5 = grouped.pivot(
        index=["variant", "parameters"],
        columns="protocol",
        values=["r2_mean", "r2_sd", "auc_mean", "auc_sd"],
    )
    table5.columns = ["_".join(column) for column in table5.columns]
    table5.reset_index().to_csv(
        output_dir / "table5_architecture_ablation.csv", index=False
    )


def _run_slug(protocol: str, variant: str, seed: int) -> str:
    safe_variant = re.sub(r"[^A-Za-z0-9]+", "_", variant).strip("_")
    return f"{protocol}__{safe_variant}__seed_{seed}"


def run(
    dataset: Path,
    output_dir: Path,
    seeds: tuple[int, ...],
    selected_variants: tuple[str, ...] | None = None,
    selected_protocols: tuple[str, ...] | None = None,
    epochs: int = 100,
) -> None:
    (
        tf,
        TCN,
        spearmanr,
        cohen_kappa_score,
        mean_squared_error,
        r2_score,
        roc_auc_score,
    ) = import_ml_dependencies()

    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(dataset)
    x, y, metadata = build_windows(df)
    protocols = protocol_indices(y, metadata)
    builders = architecture_builders(tf, TCN)
    if selected_variants:
        unknown = sorted(set(selected_variants) - set(builders))
        if unknown:
            raise ValueError(
                f"Unknown variants: {unknown}. Available: {sorted(builders)}"
            )
        builders = {
            name: builder
            for name, builder in builders.items()
            if name in selected_variants
        }
    if selected_protocols:
        unknown = sorted(set(selected_protocols) - set(protocols))
        if unknown:
            raise ValueError(
                f"Unknown protocols: {unknown}. Available: {sorted(protocols)}"
            )
        protocols = {
            name: indices
            for name, indices in protocols.items()
            if name in selected_protocols
        }

    metric_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    protocol_manifest: dict[str, object] = {}
    run_dir = output_dir / "completed_runs"
    run_dir.mkdir(parents=True, exist_ok=True)

    for protocol, indices in protocols.items():
        train_index = indices["train"]
        validation_index = indices["validation"]
        test_index = indices["test"]
        x_train, transformed, mean, scale = standardize(
            x, train_index, [validation_index, test_index]
        )
        x_validation, x_test = transformed
        y_train = y[train_index]
        y_validation = y[validation_index]
        y_test = y[test_index]
        sample_weight = balanced_regression_weights(y_train)

        protocol_manifest[protocol] = {
            "train_windows": len(train_index),
            "validation_windows": len(validation_index),
            "test_windows": len(test_index),
            "train_participants": int(
                metadata.iloc[train_index]["participant_id"].nunique()
            ),
            "validation_participants": int(
                metadata.iloc[validation_index]["participant_id"].nunique()
            ),
            "test_participants": int(
                metadata.iloc[test_index]["participant_id"].nunique()
            ),
            "feature_mean": mean.tolist(),
            "feature_scale": scale.tolist(),
        }

        for variant, build_model in builders.items():
            for seed in seeds:
                slug = _run_slug(protocol, variant, seed)
                metric_path = run_dir / f"{slug}.metrics.json"
                prediction_path = run_dir / f"{slug}.predictions.csv"
                if metric_path.exists() and prediction_path.exists():
                    print(
                        f"\n[resume] {protocol} / {variant} / seed {seed} "
                        "already complete"
                    )
                    with metric_path.open() as handle:
                        metric_rows.append(json.load(handle))
                    prediction_frames.append(pd.read_csv(prediction_path))
                    continue

                print(
                    f"\n[train] protocol={protocol} variant={variant} seed={seed}"
                )
                tf.keras.backend.clear_session()
                tf.keras.utils.set_random_seed(seed)
                model = build_model()
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(
                        learning_rate=5e-4, clipnorm=1.0
                    ),
                    loss="mse",
                    metrics=["mae"],
                )
                callbacks = [
                    tf.keras.callbacks.EarlyStopping(
                        monitor="val_loss",
                        patience=12,
                        min_delta=1e-4,
                        restore_best_weights=True,
                    ),
                    tf.keras.callbacks.ReduceLROnPlateau(
                        monitor="val_loss",
                        factor=0.5,
                        patience=5,
                        min_lr=1e-6,
                    ),
                    tf.keras.callbacks.TerminateOnNaN(),
                ]
                model.fit(
                    x_train,
                    y_train,
                    sample_weight=sample_weight,
                    validation_data=(x_validation, y_validation),
                    epochs=epochs,
                    batch_size=32,
                    callbacks=callbacks,
                    verbose=2,
                )
                prediction = model.predict(x_test, verbose=0).reshape(-1)
                completed_metric = metric_row(
                    y_test,
                    prediction,
                    protocol,
                    variant,
                    seed,
                    model.count_params(),
                    spearmanr,
                    cohen_kappa_score,
                    mean_squared_error,
                    r2_score,
                    roc_auc_score,
                )
                metric_rows.append(completed_metric)
                with metric_path.open("w") as handle:
                    json.dump(completed_metric, handle, indent=2)
                    handle.write("\n")
                prediction_frame = metadata.iloc[test_index].reset_index(drop=True)
                prediction_frame = prediction_frame.assign(
                    protocol=protocol,
                    variant=variant,
                    seed=seed,
                    y_true=y_test,
                    prediction=prediction,
                )
                prediction_frames.append(prediction_frame)
                prediction_frame.to_csv(prediction_path, index=False)

                pd.DataFrame(metric_rows).to_csv(
                    output_dir / "architecture_seed_metrics.partial.csv",
                    index=False,
                )
                pd.concat(prediction_frames, ignore_index=True).to_csv(
                    output_dir / "architecture_test_predictions.partial.csv",
                    index=False,
                )

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    metrics.to_csv(output_dir / "architecture_seed_metrics.csv", index=False)
    predictions.to_csv(output_dir / "architecture_test_predictions.csv", index=False)
    with (output_dir / "protocol_manifest.json").open("w") as handle:
        json.dump(protocol_manifest, handle, indent=2)
        handle.write("\n")
    generate_tables(metrics, output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        help=(
            "Optional subset: MLP CNN 'CNN+GRU' 'CNN+TCN' "
            "'CNN+TCN+GRU' VRC-3PO Lite"
        ),
    )
    parser.add_argument(
        "--protocols",
        nargs="+",
        choices=["participant", "blocked_within_participant"],
    )
    parser.add_argument("--epochs", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        args.dataset,
        args.output_dir,
        tuple(args.seeds),
        tuple(args.variants) if args.variants else None,
        tuple(args.protocols) if args.protocols else None,
        args.epochs,
    )


if __name__ == "__main__":
    main()
