"""Re-implementations of published cybersickness detectors (Reviewer 1, Comment 3).

LiteVR (Kundu et al., 2023, arXiv:2302.03037) trained LSTM, GRU and MLP
detectors on the Islam et al. eye- and head-tracking recordings, which are the
Simulations source folder of this benchmark. Its reduced ("lite") models are
specified layer by layer in its Tables 8-10; they are rebuilt here with the
same layer types, widths and dropout rates. TruVR (Kundu et al., 2022,
arXiv:2209.05257) used an Explainable Boosting Machine (EBM) on tabular
features; it is rebuilt on the benchmark's window statistics.

Adaptations, stated in the manuscript:
  * input: the benchmark's 30 x 14 window at 1 Hz (LiteVR: 60 timesteps at the
    native rate, 18 selected of 43 features including physiological signals);
  * output: one sigmoid unit for the binary endpoint (LiteVR: four-class
    softmax over severity levels);
  * the MLP flattens the window (LiteVR's MLP sees one timestep's features);
  * training uses the benchmark's shared recipe (Adam 5e-4, batch 32, early
    stopping on validation AUC, balanced class weights) instead of LiteVR's
    (Adam 1e-3, batch 256, patience 30), so every compared model is trained
    identically.

Registered in ``revision_cv`` as ``litevr:LSTM``, ``litevr:GRU``,
``litevr:MLP`` and ``feat:ebm``; the FMS regression versions (LiteVR also
regressed FMS) as ``litevr_reg:<Variant>``. ``python -m analysis.litevr
regression --dataset ... --output-dir ...`` summarizes R^2 for every
regression run (``litevr_reg:*`` and ``reg:*``) in a run store.
"""

from __future__ import annotations

import numpy as np

LITEVR_VARIANTS = ("LSTM", "GRU", "MLP")


def build_litevr(tf, variant: str, n_timesteps: int = 30, n_features: int = 14,
                 output: str = "sigmoid"):
    """LiteVR reduced architectures (Tables 8, 9, 10) with a binary (sigmoid)
    or regression (linear) output."""
    layers = tf.keras.layers
    inp = layers.Input(shape=(n_timesteps, n_features))
    if variant == "LSTM":            # Table 8: LSTM 64 (0.15) -> Dense 32 ReLU -> output
        x = layers.LSTM(64, dropout=0.15)(inp)
        x = layers.Dropout(0.15)(x)
        x = layers.Dense(32, activation="relu")(x)
    elif variant == "GRU":           # Table 9: GRU 32 (0.2) -> GRU 64 (0.2) -> Dense 32 ReLU -> output
        x = layers.GRU(32, dropout=0.2, return_sequences=True)(inp)
        x = layers.Dropout(0.2)(x)
        x = layers.GRU(64, dropout=0.2)(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(32, activation="relu")(x)
    elif variant == "MLP":           # Table 10: Dense 64 (0.25) -> 32 (0.25) -> 16 -> output
        x = layers.Flatten()(inp)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(0.25)(x)
        x = layers.Dense(32, activation="relu")(x)
        x = layers.Dropout(0.25)(x)
        x = layers.Dense(16, activation="relu")(x)
    else:
        raise ValueError(f"Unknown LiteVR variant {variant!r}; choose from {LITEVR_VARIANTS}")
    out = layers.Dense(1, activation=output)(x)
    return tf.keras.Model(inp, out, name=f"litevr_{variant.lower()}")


def model_litevr(tf, variant, x_tr, y_tr, x_va, y_va, x_te, epochs, seeds=(42,), **_):
    """Train one LiteVR variant per seed and average the seeds' probabilities."""
    from analysis.revision_cv import _fit_binary

    shape = x_tr.shape[1:]
    build = lambda: build_litevr(tf, variant, *shape)  # noqa: E731
    va, te, info = [], [], {"members": []}
    for seed in seeds:
        model, n, monitor = _fit_binary(tf, build, x_tr, y_tr, x_va, y_va, seed, epochs)
        va.append(model.predict(x_va, verbose=0).reshape(-1))
        te.append(model.predict(x_te, verbose=0).reshape(-1))
        info["members"].append({"seed": seed, "epochs": n, "monitor": monitor})
    info["parameters"] = int(model.count_params())
    return np.mean(va, axis=0), np.mean(te, axis=0), info


def make_ebm():
    """TruVR's EBM settings: learning rate 0.001, early-stopping patience 30,
    library defaults otherwise. Pairwise interactions are switched off to bound
    the runtime of 60+ fits; the main-effect model is what TruVR explains."""
    from interpret.glassbox import ExplainableBoostingClassifier

    return ExplainableBoostingClassifier(learning_rate=0.001, early_stopping_rounds=30,
                                         interactions=0, random_state=42, n_jobs=-1)


def balanced_sample_weight(y: np.ndarray) -> np.ndarray:
    counts = np.bincount(y, minlength=2).astype(float)
    counts[counts == 0] = 1.0
    return np.where(y == 1, len(y) / (2 * counts[1]), len(y) / (2 * counts[0]))


def window_folds(y: np.ndarray, n_folds: int, n_repeats: int, seed: int,
                 validation_fraction: float = 0.1):
    """Sample-level K-fold over windows (the protocol of LiteVR and TruVR).

    Windows of one participant, and overlapping neighbouring windows, can fall
    on both sides of a split. Validation windows for early stopping are drawn
    from the training windows. Returns (repeat, fold, train, validation, test)
    index arrays; every window is tested exactly once per repeat.
    """
    from sklearn.model_selection import StratifiedKFold, train_test_split

    out = []
    for repeat in range(n_repeats):
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed + repeat)
        for fold, (train, test) in enumerate(splitter.split(np.zeros(len(y)), y)):
            train, validation = train_test_split(
                train, test_size=validation_fraction, stratify=y[train],
                random_state=seed + 1000 * repeat + fold)
            out.append((repeat, fold, np.sort(train), np.sort(validation), np.sort(test)))
    return out


def model_litevr_regression(tf, variant, x_tr, fms_tr, x_va, fms_va, x_te, epochs, seeds=(42,), **_):
    """LiteVR architecture regressing window-mean FMS, trained exactly like the
    benchmark's regression models (MSE, balanced regression weights, Adam)."""
    from analysis.retrain_camera_ready import balanced_regression_weights
    from analysis.revision_cv import _callbacks

    shape = x_tr.shape[1:]
    va, te, info = [], [], {"members": []}
    for seed in seeds:
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(seed)
        model = build_litevr(tf, variant, *shape, output="linear")
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4, clipnorm=1.0),
                      loss="mse")
        history = model.fit(x_tr, fms_tr, sample_weight=balanced_regression_weights(fms_tr),
                            validation_data=(x_va, fms_va), epochs=epochs, batch_size=32,
                            verbose=0, callbacks=_callbacks(tf, "val_loss", "min"))
        va.append(model.predict(x_va, verbose=0).reshape(-1))
        te.append(model.predict(x_te, verbose=0).reshape(-1))
        info["members"].append({"seed": seed, "epochs": len(history.history["loss"])})
    info["parameters"] = int(model.count_params())
    return np.mean(va, axis=0), np.mean(te, axis=0), info


def weighted_r2(fms, pred, w=None):
    w = np.ones(len(fms)) if w is None else w
    mean = (w * fms).sum() / w.sum()
    total = (w * (fms - mean) ** 2).sum()
    return 1.0 - (w * (fms - pred) ** 2).sum() / total if total > 0 else float("nan")


def summarize_regression(dataset, output_dir, bootstrap=2000):
    """R^2, RMSE and Spearman rho of every regression run, per scope and protocol.

    Out-of-fold predictions of each complete repeat are pooled; R^2 is computed
    against the pooled test FMS and averaged over repeats. Intervals resample
    participants. Writes ``summary/regression.csv``.
    """
    from pathlib import Path

    import pandas as pd
    from scipy.stats import spearmanr

    from analysis.revision_common import ParticipantBootstrap, percentile_ci
    from analysis.revision_cv import load_runs, oof_matrix, prepare

    out = Path(output_dir)
    runs = load_runs(out)
    runs = runs[runs["model"].str.startswith(("litevr_reg:", "reg:"))]
    if runs.empty:
        raise SystemExit("No regression runs found.")
    rows = []
    for (label, scope, protocol), block in runs.groupby(["label", "scope", "protocol"]):
        _, fms, _, meta = prepare(dataset, scope, label, out / "cache")
        pid = meta["participant_id"].to_numpy()
        boot = ParticipantBootstrap(pid, bootstrap, seed=3)
        for model, group in block.groupby("model"):
            M, _, reps = oof_matrix(group, len(fms))
            if not len(reps):
                continue
            draws = boot.run(lambda w: np.mean([weighted_r2(fms, m, w) for m in M]))
            lo, hi = percentile_ci(draws)
            rows.append({"label": label, "scope": scope, "protocol": protocol, "model": model,
                         "repeats": len(reps), "windows": len(fms), "participants": len(set(pid)),
                         "r2": np.mean([weighted_r2(fms, m) for m in M]),
                         "r2_ci_low": lo, "r2_ci_high": hi,
                         "rmse": np.mean([np.sqrt(np.mean((fms - m) ** 2)) for m in M]),
                         "spearman": np.mean([spearmanr(fms, m)[0] for m in M])})
    table = pd.DataFrame(rows)
    (out / "summary").mkdir(exist_ok=True)
    table.to_csv(out / "summary" / "regression.csv", index=False)
    print(table.round(3).to_string(index=False))
    return table


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Regression summary (R^2) for a run store.")
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("regression")
    r.add_argument("--dataset", type=Path, required=True)
    r.add_argument("--output-dir", type=Path, required=True)
    r.add_argument("--bootstrap", type=int, default=2000)
    a = parser.parse_args()
    summarize_regression(a.dataset, a.output_dir, a.bootstrap)
