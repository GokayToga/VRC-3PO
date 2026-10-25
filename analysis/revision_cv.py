#!/usr/bin/env python3
"""Revision experiments: repeated participant-level CV for the seated endpoint.

One engine serves every revision analysis that needs training, so that all of
them share the same windows, folds, standardization and uncertainty method:

  * primary estimate          Reviewer 2, Comment 1   model ``cnn_ensemble``
  * environment/source control Reviewer 2, Comment 2  ``nuisance:condition``,
                                                       ``nuisance:source`` and
                                                       the within-condition AUC
  * threshold sensitivity     Reviewer 2, Comment 3   ``--labels`` variants
  * equal-budget tuning       Reviewer 2, Comment 5   ``tuned:<Variant>``
  * architecture x composition Reviewer 2, Comment 6  ``clf:<Variant>``
  * comparison with existing  Reviewer 1, Comment 3   ``feat:logreg|rf|hgb``
    method families
  * published detectors       Reviewer 1, Comment 3   ``litevr:LSTM|GRU|MLP``,
    re-implemented                                     ``feat:ebm``, ``litevr_reg:*`` (analysis/litevr.py),
                                                       also under ``--protocol windows``
                                                       (their sample-level K-fold)
  * validation-selected operating points  Reviewer 2 (Table 5 note)

Usage
-----
Train (GPU for the neural models; resumable, one file per model/label/fold)::

    python -m analysis.revision_cv run --dataset v1_1.csv --output-dir OUT \\
        --models cnn_ensemble --labels fms_gt2 --repeats 5

Summarize (CPU, reads whatever has finished)::

    python -m analysis.revision_cv summarize --dataset v1_1.csv --output-dir OUT

Folds are a deterministic function of (dataset, scope, label, repeats, folds,
seed), so separate ``run`` invocations with different ``--models`` evaluate
every model on identical participant partitions and can be compared pairwise.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.retrain_camera_ready import FEATURE_COLUMNS
from analysis.revision_common import (
    CHANNEL_SETS,
    LABEL_DEFINITIONS,
    calibrate_per_recording,
    parse_scope,
    ParticipantBootstrap,
    WeightedAUC,
    SCOPES,
    Fold,
    auc,
    average_precision,
    build_windows,
    make_labels,
    participant_bootstrap,
    participant_table,
    percentile_ci,
    repeated_participant_folds,
    standardize,
    window_features,
    within_group_auc,
)

ENSEMBLE_SEEDS = (42, 123, 456, 789, 2024)
ARCH_SEEDS = (42,)
ENGINE_VERSION = 1

# Equal-budget search space for ``tuned:<Variant>`` (Reviewer 2, Comment 5).
# Only training hyperparameters are searched so that every architecture keeps
# the exact layer stack reported in the architecture table.
TUNING_SPACE = {
    "learning_rate": [1e-4, 3e-4, 1e-3, 3e-3],
    "batch_size": [16, 32, 64],
    "weight_decay": [0.0, 1e-4, 1e-3],
}


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
def prepare(dataset: Path, scope: str, label: str, cache_dir: Path | None = None):
    """Windows for ``scope`` (see ``parse_scope``) with binary labels for ``label``."""
    spec = parse_scope(scope)
    cache = None
    if cache_dir is not None:
        digest = hashlib.sha1(
            f"{Path(dataset).resolve()}|{Path(dataset).stat().st_size}|{spec['calibration']}".encode()
        ).hexdigest()[:10]
        cache = Path(cache_dir) / f"windows_{digest}.npz"
    if cache is not None and cache.exists():
        blob = np.load(cache, allow_pickle=True)
        x, fms = blob["x"], blob["fms"]
        metadata = pd.DataFrame(blob["metadata"], columns=list(blob["columns"]))
        for column in ("start_index", "end_index"):
            metadata[column] = metadata[column].astype(int)
        metadata["session_baseline_fms"] = metadata["session_baseline_fms"].astype(float)
    else:
        frame = pd.read_csv(dataset)
        if spec["calibration"] == "cal60":
            frame = calibrate_per_recording(frame)
        x, fms, metadata = build_windows(frame)
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cache, x=x, fms=fms,
                metadata=metadata.to_numpy(dtype=object),
                columns=np.asarray(metadata.columns),
            )
    if "-" in scope:
        # Canonical window order for variant scopes, so raw and calibrated inputs
        # index identical windows and can be compared pairwise. (Plain scopes keep
        # the original order, which finished runs were stored against.)
        order = np.lexsort((metadata["start_index"].to_numpy(), metadata["recording_id"].astype(str).to_numpy()))
        x, fms = x[order], fms[order]
        metadata = metadata.iloc[order].reset_index(drop=True)
    keep = (metadata["source_dataset"].isin(spec["sources"])
            & (metadata["start_index"] >= spec["min_start"])).to_numpy()
    channels = [FEATURE_COLUMNS.index(c) for c in CHANNEL_SETS[spec["channels"]]]
    x, fms = x[keep][:, :, channels], fms[keep]
    metadata = metadata[keep].reset_index(drop=True)
    y = make_labels(fms, metadata, label)
    return x, fms, y, metadata


def make_folds(y, metadata, n_folds, n_repeats, seed, protocol="cv", all_meta=None, all_fms=None):
    if protocol == "cv":
        info = participant_table(y, metadata)
        return repeated_participant_folds(info, n_folds=n_folds, n_repeats=n_repeats, seed=seed)
    if protocol == "reference":
        # The frozen seed-42 assignment over all 84 identifiers, restricted to scope.
        from analysis.corrected_evaluation import participant_split

        train, validation, test = participant_split(all_fms, all_meta, seed=42)
        in_scope = set(metadata["participant_id"])
        return [Fold(0, 0, frozenset(train & in_scope), frozenset(validation & in_scope),
                     frozenset(test & in_scope))]
    raise ValueError(protocol)


# ---------------------------------------------------------------------------
# Models: each returns (validation_scores, test_scores, info)
# ---------------------------------------------------------------------------
def _callbacks(tf, monitor, mode):
    return [
        tf.keras.callbacks.EarlyStopping(monitor=monitor, mode=mode, patience=12,
                                         min_delta=1e-4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor=monitor, mode=mode, factor=0.5,
                                             patience=5, min_lr=1e-6),
        tf.keras.callbacks.TerminateOnNaN(),
    ]


def _class_weights(y):
    counts = np.bincount(y, minlength=2).astype(float)
    counts[counts == 0] = 1.0
    return {0: len(y) / (2 * counts[0]), 1: len(y) / (2 * counts[1])}


def _fit_binary(tf, build, x_tr, y_tr, x_va, y_va, seed, epochs,
                learning_rate=5e-4, batch_size=32, weight_decay=0.0):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    model = build()
    if weight_decay:
        optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate,
                                              weight_decay=weight_decay, clipnorm=1.0)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss="binary_crossentropy",
                  metrics=[tf.keras.metrics.AUC(name="auc")])
    two_class_val = 0 < y_va.sum() < len(y_va)
    monitor, mode = ("val_auc", "max") if two_class_val else ("val_loss", "min")
    history = model.fit(x_tr, y_tr, validation_data=(x_va, y_va),
                        class_weight=_class_weights(y_tr), epochs=epochs,
                        batch_size=batch_size, verbose=0,
                        callbacks=_callbacks(tf, monitor, mode))
    epochs_run = len(history.history.get("loss", []))
    return model, epochs_run, monitor


def _classifier_builder(tf, variant):
    """Architecture body from the manuscript with a sigmoid output."""
    if variant == "CNN_frozen":
        from analysis.composition_robustness import build_cnn_classifier
        return lambda: build_cnn_classifier(tf)
    from analysis.retrain_camera_ready import architecture_builders
    from tcn import TCN

    builders = architecture_builders(tf, TCN)
    if variant not in builders:
        raise ValueError(f"Unknown architecture {variant!r}; choose from {sorted(builders)}")

    def build():
        base = builders[variant]()
        out = tf.keras.layers.Activation("sigmoid")(base.output)
        return tf.keras.Model(base.input, out, name=f"{base.name}_clf")

    return build


def build_frozen_cnn(tf, n_features):
    """The frozen seated-ensemble CNN (Masking, Conv1D-64, BN, Dropout, GAP, 8, 1)
    for any number of input channels; identical to ``build_cnn_classifier`` at 14."""
    layers = tf.keras.layers
    inp = layers.Input(shape=(30, n_features))
    x = layers.Masking(mask_value=0.0)(inp)
    x = layers.Conv1D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(8, activation="relu")(x)
    return tf.keras.Model(inp, layers.Dense(1, activation="sigmoid")(x))


def model_cnn_ensemble(tf, x_tr, y_tr, x_va, y_va, x_te, epochs, members=ENSEMBLE_SEEDS, **_):
    n_features = x_tr.shape[-1]
    build = lambda: build_frozen_cnn(tf, n_features)  # noqa: E731
    va, te, info = [], [], {"members": []}
    for seed in members:
        model, n, monitor = _fit_binary(tf, build, x_tr, y_tr, x_va, y_va, seed, epochs)
        va.append(model.predict(x_va, verbose=0).reshape(-1))
        te.append(model.predict(x_te, verbose=0).reshape(-1))
        info["members"].append({"seed": seed, "epochs": n, "monitor": monitor})
    info["parameters"] = int(model.count_params())
    return np.mean(va, axis=0), np.mean(te, axis=0), info


def model_classifier(tf, variant, x_tr, y_tr, x_va, y_va, x_te, epochs, seeds=ARCH_SEEDS, **_):
    build = _classifier_builder(tf, variant)
    va, te, info = [], [], {"members": []}
    for seed in seeds:
        model, n, monitor = _fit_binary(tf, build, x_tr, y_tr, x_va, y_va, seed, epochs)
        va.append(model.predict(x_va, verbose=0).reshape(-1))
        te.append(model.predict(x_te, verbose=0).reshape(-1))
        info["members"].append({"seed": seed, "epochs": n, "monitor": monitor})
    info["parameters"] = int(model.count_params())
    return np.mean(va, axis=0), np.mean(te, axis=0), info


def model_tuned(tf, variant, x_tr, y_tr, x_va, y_va, x_te, epochs, trials=8, fold_seed=0, **_):
    """Random search with an identical trial budget for every architecture.

    Trials are ranked by validation-participant AUC only; the winning trial's
    model is the one scored on the test participants, once.
    """
    build = _classifier_builder(tf, variant)
    grid = list(itertools.product(*TUNING_SPACE.values()))
    rng = np.random.default_rng(fold_seed)
    chosen = [grid[i] for i in rng.choice(len(grid), size=min(trials, len(grid)), replace=False)]
    best = None
    log = []
    for lr, bs, wd in chosen:
        model, n, monitor = _fit_binary(tf, build, x_tr, y_tr, x_va, y_va, 42, epochs,
                                        learning_rate=lr, batch_size=bs, weight_decay=wd)
        va = model.predict(x_va, verbose=0).reshape(-1)
        score = auc(y_va, va) if 0 < y_va.sum() < len(y_va) else -float(
            np.mean((va - y_va) ** 2))
        log.append({"learning_rate": lr, "batch_size": bs, "weight_decay": wd,
                    "validation_score": float(score), "epochs": n})
        if best is None or score > best[0]:
            best = (score, va, model.predict(x_te, verbose=0).reshape(-1), log[-1])
    return best[1], best[2], {"trials": log, "selected": best[3],
                              "parameters": int(model.count_params())}


def model_regression(tf, variant, x_tr, fms_tr, x_va, fms_va, x_te, epochs, seeds=ARCH_SEEDS, **_):
    """Regression-trained architecture as in the manuscript's architecture table."""
    from analysis.retrain_camera_ready import architecture_builders, balanced_regression_weights
    from tcn import TCN

    build = architecture_builders(tf, TCN)[variant]
    va, te, info = [], [], {"members": []}
    for seed in seeds:
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(seed)
        model = build()
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


def model_features(kind, x_tr_raw, y_tr, x_va_raw, x_te_raw, **_):
    """Feature-based baselines on hand-crafted window statistics."""
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    f_tr, f_va, f_te = (window_features(a) for a in (x_tr_raw, x_va_raw, x_te_raw))
    if kind == "logreg":
        model = make_pipeline(StandardScaler(), LogisticRegression(
            C=0.1, class_weight="balanced", max_iter=5000))
    elif kind == "rf":
        model = RandomForestClassifier(n_estimators=500, min_samples_leaf=5,
                                       class_weight="balanced_subsample", n_jobs=-1,
                                       random_state=42)
    elif kind == "hgb":
        model = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300,
                                               class_weight="balanced", random_state=42)
    elif kind == "ebm":
        from analysis.litevr import balanced_sample_weight, make_ebm

        model = make_ebm()
        model.fit(f_tr, y_tr, sample_weight=balanced_sample_weight(y_tr))
        return (model.predict_proba(f_va)[:, 1], model.predict_proba(f_te)[:, 1],
                {"n_features": int(f_tr.shape[1])})
    else:
        raise ValueError(kind)
    model.fit(f_tr, y_tr)
    return (model.predict_proba(f_va)[:, 1], model.predict_proba(f_te)[:, 1],
            {"n_features": int(f_tr.shape[1])})


def model_nuisance(spec, meta_tr, y_tr, meta_va, meta_te, **_):
    """Logistic regression on nuisance variables only (no physiology).

    ``condition`` / ``source``: environment identity. ``clock``: time since the
    recording started (sickness accumulates with exposure). ``condition_clock``:
    both, i.e. everything an application knows without any sensor.
    """
    from sklearn.linear_model import LogisticRegression

    parts = {"condition_clock": ["condition", "clock"]}.get(spec, [spec])
    cats = [{"source": "source_dataset"}.get(p, p) for p in parts if p != "clock"]
    levels = {c: sorted(set(meta_tr[c]) | set(meta_va[c]) | set(meta_te[c])) for c in cats}

    def design(meta):
        cols = [(meta[c].to_numpy()[:, None] == np.asarray(levels[c])[None, :]).astype(float) for c in cats]
        if "clock" in parts:
            t = meta["start_index"].to_numpy(dtype=float)[:, None] / 600.0
            cols += [t, t ** 2]
        return np.hstack(cols)

    model = LogisticRegression(C=1e3, max_iter=5000, class_weight="balanced")
    model.fit(design(meta_tr), y_tr)
    return (model.predict_proba(design(meta_va))[:, 1], model.predict_proba(design(meta_te))[:, 1],
            {"levels": levels, "parts": parts})


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")


def needs_tf(model: str) -> bool:
    return model == "cnn_ensemble" or model.split(":")[0] in ("clf", "reg", "tuned", "debias", "litevr", "litevr_reg")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
def run(args) -> None:
    out = Path(args.output_dir)
    runs = out / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    tf = None
    if any(needs_tf(m) for m in args.models):
        import tensorflow as tf  # noqa: F811

    all_x, all_fms, all_y, all_meta = prepare(args.dataset, "all", "fms_gt2", out / "cache")
    for label in args.labels:
        x, fms, y, meta = prepare(args.dataset, args.scope, label, out / "cache")
        pid = meta["participant_id"].to_numpy()
        if args.protocol == "windows":
            from analysis.litevr import window_folds

            splits = window_folds(y, args.folds, args.repeats, args.seed)
        else:
            folds = make_folds(y, meta, args.folds, args.repeats, args.seed, args.protocol,
                               all_meta=all_meta, all_fms=all_fms)
            splits = [(f.repeat, f.fold,
                       np.flatnonzero(np.isin(pid, list(f.train))),
                       np.flatnonzero(np.isin(pid, list(f.validation))),
                       np.flatnonzero(np.isin(pid, list(f.test)))) for f in folds]
        print(f"[{label}] scope={args.scope} protocol={args.protocol} windows={len(y)} "
              f"participants={len(set(pid))} prevalence={y.mean():.3f} folds={len(splits)}", flush=True)
        for model_name, (repeat, fold_no, tr, va, te) in itertools.product(args.models, splits):
            tag = f"{slug(model_name)}__{label}__{args.scope}__{args.protocol}__r{repeat:02d}_f{fold_no}"
            path = runs / f"{tag}.npz"
            if path.exists():
                print(f"  [resume] {tag}", flush=True)
                continue
            if y[te].sum() in (0, len(te)):
                print(f"  [note] {tag}: single-class test fold; stored, excluded from fold AUC")
            x_tr, (x_va, x_te) = standardize(x, tr, [va, te])
            kind, _, variant = model_name.partition(":")
            common = dict(epochs=args.epochs)
            if kind in ("clf", "tuned", "reg") and x.shape[-1] != len(FEATURE_COLUMNS):
                raise SystemExit(f"{model_name} needs all 14 channels; use cnn_ensemble or feat:* "
                                 f"for scope {args.scope}")
            if model_name == "cnn_ensemble":
                s_va, s_te, info = model_cnn_ensemble(tf, x_tr, y[tr], x_va, y[va], x_te,
                                                      members=tuple(args.members), **common)
            elif kind == "clf":
                s_va, s_te, info = model_classifier(tf, variant, x_tr, y[tr], x_va, y[va], x_te,
                                                    seeds=tuple(args.arch_seeds), **common)
            elif kind == "tuned":
                s_va, s_te, info = model_tuned(tf, variant, x_tr, y[tr], x_va, y[va], x_te,
                                               trials=args.trials,
                                               fold_seed=1000 * (repeat + 1) + fold_no, **common)
            elif kind == "reg":
                s_va, s_te, info = model_regression(tf, variant, x_tr, fms[tr], x_va, fms[va], x_te,
                                                    seeds=tuple(args.arch_seeds), **common)
            elif kind == "feat":
                s_va, s_te, info = model_features(variant, x[tr], y[tr], x[va], x[te])
            elif kind == "nuisance":
                s_va, s_te, info = model_nuisance(variant, meta.iloc[tr], y[tr],
                                                  meta.iloc[va], meta.iloc[te])
            elif kind == "debias":
                from analysis.revision_debias import model_debiased
                s_va, s_te, info = model_debiased(tf, variant, x_tr, y[tr], x_va, y[va], x_te,
                                                  meta.iloc[tr], meta.iloc[va],
                                                  members=tuple(args.members), **common)
            elif kind == "litevr_reg":
                from analysis.litevr import model_litevr_regression
                s_va, s_te, info = model_litevr_regression(tf, variant, x_tr, fms[tr], x_va, fms[va],
                                                           x_te, seeds=tuple(args.arch_seeds), **common)
            elif kind == "litevr":
                from analysis.litevr import model_litevr
                s_va, s_te, info = model_litevr(tf, variant, x_tr, y[tr], x_va, y[va], x_te,
                                                seeds=tuple(args.arch_seeds), **common)
            else:
                raise ValueError(f"Unknown model {model_name!r}")
            info.update({
                "engine_version": ENGINE_VERSION, "model": model_name, "label": label,
                "scope": args.scope, "protocol": args.protocol, "repeat": repeat,
                "fold": fold_no, "train_participants": len(set(pid[tr])),
                "validation_participants": len(set(pid[va])),
                "test_participants": len(set(pid[te])), "train_windows": int(len(tr)),
                "test_windows": int(len(te)), "test_auc": auc(y[te], s_te),
            })
            np.savez_compressed(path, val_index=va, val_scores=s_va.astype(np.float64),
                                test_index=te, test_scores=s_te.astype(np.float64),
                                info=json.dumps(info))
            print(f"  [done] {tag}  test AUC={info['test_auc']:.3f}", flush=True)


# ---------------------------------------------------------------------------
# Summaries (CPU only)
# ---------------------------------------------------------------------------
def load_runs(out: Path) -> pd.DataFrame:
    records = []
    for path in sorted((out / "runs").glob("*.npz")):
        blob = np.load(path, allow_pickle=False)
        info = json.loads(str(blob["info"]))
        info.update({k: blob[k] for k in ("val_index", "val_scores", "test_index", "test_scores")})
        records.append(info)
    return pd.DataFrame(records)


def oof_matrix(group: pd.DataFrame, n_windows: int):
    """Scores and fold ids (repeats x windows) for each complete repeat."""
    repeats = sorted(group["repeat"].unique())
    matrix, folds, complete = [], [], []
    for r in repeats:
        scores = np.full(n_windows, np.nan)
        fold_id = np.full(n_windows, -1)
        for _, row in group[group["repeat"] == r].iterrows():
            scores[row["test_index"]] = row["test_scores"]
            fold_id[row["test_index"]] = row["fold"]
        if np.isfinite(scores).all():
            matrix.append(scores)
            folds.append(fold_id)
            complete.append(r)
    return np.asarray(matrix), np.asarray(folds), complete


def pick_thresholds(y_val, s_val):
    """Operating points chosen on validation participants only."""
    thresholds = np.unique(s_val)
    pos, neg = max(int(y_val.sum()), 1), max(int((1 - y_val).sum()), 1)
    sens = np.array([(s_val[y_val == 1] >= t).sum() / pos for t in thresholds])
    spec = np.array([(s_val[y_val == 0] < t).sum() / neg for t in thresholds])
    chosen = {"default_0.5": 0.5,
              "max_youden": float(thresholds[np.argmax(sens + spec - 1)])}
    ok = np.flatnonzero(sens >= 0.80)
    chosen["sensitivity_ge_0.80"] = float(thresholds[ok[-1]]) if len(ok) else float(thresholds[0])
    ok = np.flatnonzero(spec >= 0.90)
    chosen["specificity_ge_0.90"] = float(thresholds[ok[0]]) if len(ok) else float(thresholds[-1])
    return chosen


def confusion(y, s, t):
    alert = s >= t
    tp, fp = int((alert & (y == 1)).sum()), int((alert & (y == 0)).sum())
    fn, tn = int((~alert & (y == 1)).sum()), int((~alert & (y == 0)).sum())
    return {"sensitivity": tp / max(tp + fn, 1), "specificity": tn / max(tn + fp, 1),
            "alert_rate": float(alert.mean()), "ppv": tp / max(tp + fp, 1)}


def variance_components(fold_auc: pd.DataFrame) -> dict:
    """Two-way random-effects decomposition of fold AUC: composition x model."""
    table = fold_auc.pivot_table(index="composition", columns="model", values="auc").dropna()
    if table.shape[0] < 2 or table.shape[1] < 2:
        return {}
    a = table.to_numpy()
    n_c, n_m = a.shape
    grand = a.mean()
    ss_c = n_m * ((a.mean(axis=1) - grand) ** 2).sum()
    ss_m = n_c * ((a.mean(axis=0) - grand) ** 2).sum()
    ss_r = ((a - a.mean(axis=1, keepdims=True) - a.mean(axis=0, keepdims=True) + grand) ** 2).sum()
    ms_c, ms_m = ss_c / (n_c - 1), ss_m / (n_m - 1)
    ms_r = ss_r / ((n_c - 1) * (n_m - 1))
    var_c, var_m = max((ms_c - ms_r) / n_m, 0.0), max((ms_m - ms_r) / n_c, 0.0)
    total = var_c + var_m + ms_r
    return {
        "n_compositions": int(n_c), "n_models": int(n_m), "models": list(table.columns),
        "var_composition": var_c, "var_model": var_m, "var_residual": ms_r,
        "share_composition": var_c / total if total else float("nan"),
        "share_model": var_m / total if total else float("nan"),
        "share_residual": ms_r / total if total else float("nan"),
        "range_across_compositions_mean": float(np.ptp(a.mean(axis=1))),
        "range_across_models_mean": float(np.ptp(a.mean(axis=0))),
        "model_means": dict(zip(table.columns, a.mean(axis=0).round(4).tolist())),
    }


def stacked_increment(y, scores, onehot, participants, n_folds=5, seed=0):
    """Out-of-fold AUC of [environment] vs [environment + model score].

    Second-stage logistic regressions are fitted with participant-grouped CV
    on the out-of-fold scores, so neither stage ever sees its own test people.
    """
    from sklearn.linear_model import LogisticRegression

    uniq = np.unique(participants)
    rng = np.random.default_rng(seed)
    fold_of = dict(zip(uniq[rng.permutation(len(uniq))], np.arange(len(uniq)) % n_folds))
    fid = np.array([fold_of[p] for p in participants])
    logit = np.log(np.clip(scores, 1e-6, 1 - 1e-6) / (1 - np.clip(scores, 1e-6, 1 - 1e-6)))
    if scores.min() < 0 or scores.max() > 1:           # regression scores
        logit = (scores - scores.mean()) / (scores.std() + 1e-9)
    env_only, env_plus = np.zeros(len(y)), np.zeros(len(y))
    for f in range(n_folds):
        tr, te = fid != f, fid == f
        for design, target in ((onehot, env_only), (np.column_stack([onehot, logit]), env_plus)):
            m = LogisticRegression(C=1e3, max_iter=5000).fit(design[tr], y[tr])
            target[te] = m.predict_proba(design[te])[:, 1]
    return env_only, env_plus


def _groups(cond, fold_row):
    return np.char.add(cond.astype(str), np.char.add("|", fold_row.astype(str)))


def summarize(args) -> None:
    """Tables for the manuscript. All intervals resample participants.

    Every statistic of one evaluation set is computed on the same bootstrap
    draws, so paired differences are exact paired bootstrap differences.
    """
    import time

    out = Path(args.output_dir)
    runs = load_runs(out)
    if runs.empty:
        raise SystemExit("No finished runs found.")
    summary_dir = out / "summary"
    summary_dir.mkdir(exist_ok=True)
    B = args.bootstrap
    t0 = time.time()
    primary_rows, fold_rows, op_rows, cmp_rows, nuis_rows, cond_rows = [], [], [], [], [], []
    variant_store = {}   # (label, sources, min_start) -> {"scopes": {scope: {model: (M, F)}}, ...}

    def log(msg):
        print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)

    for (label, scope, protocol), block in runs.groupby(["label", "scope", "protocol"]):
        x, fms, y, meta = prepare(args.dataset, scope, label, out / "cache")
        pid = meta["participant_id"].to_numpy()
        cond = meta["condition"].to_numpy()
        src = meta["source_dataset"].to_numpy()
        minute = np.minimum(meta["start_index"].to_numpy() // 60, 8).astype(str)
        clock = meta["start_index"].to_numpy(dtype=float) / 600.0
        boot = ParticipantBootstrap(pid, B, seed=1)
        spec = parse_scope(scope)
        vkey = (label, spec["sources"], spec["min_start"])
        matrices, fold_ids = {}, {}
        log(f"{label} / {scope} / {protocol}: {block['model'].nunique()} models, {len(y)} windows")

        for model, group in block.groupby("model"):
            for _, row in group.iterrows():
                ti = row["test_index"]
                fold_rows.append({"label": label, "scope": scope, "protocol": protocol,
                                  "model": model, "repeat": row["repeat"], "fold": row["fold"],
                                  "composition": f"r{row['repeat']}_f{row['fold']}",
                                  "test_participants": row["test_participants"],
                                  "test_windows": len(ti), "elevated": int(y[ti].sum()),
                                  "auc": auc(y[ti], row["test_scores"])})
            if protocol == "reference":
                row = group.iloc[0]
                ti, s = row["test_index"], row["test_scores"]
                ref_boot = ParticipantBootstrap(pid[ti], B, seed=1)
                lo, hi = percentile_ci(ref_boot.run(WeightedAUC(y[ti], s)))
                primary_rows.append({"label": label, "scope": scope, "protocol": protocol,
                                     "model": model, "repeats": 1, "windows": len(ti),
                                     "participants": len(set(pid[ti])), "prevalence": y[ti].mean(),
                                     "auc_mean": auc(y[ti], s), "auc_ci_low": lo, "auc_ci_high": hi,
                                     "ap_mean": average_precision(y[ti], s),
                                     "within_condition_auc": within_group_auc(y[ti], s, cond[ti])})
                continue

            M, F, reps = oof_matrix(group, len(y))
            if len(reps) == 0:
                log(f"  {model}: no complete repeat yet, skipped")
                continue
            matrices[model], fold_ids[model] = M, F
            pooled = [WeightedAUC(y, m) for m in M]
            # Within-condition pairs restricted to the same CV fold, so that
            # fold-to-fold score offsets cannot pose as discrimination; an
            # environment-only score is then exactly 0.5.
            within = [WeightedAUC(y, m, _groups(cond, f)) for m, f in zip(M, F)]
            # ... and additionally within the same minute of exposure, which removes
            # anything that merely tracks time on task.
            within_t = [WeightedAUC(y, m, np.char.add(_groups(cond, f), np.char.add("|", minute)))
                        for m, f in zip(M, F)]
            per_rep_auc = np.array([f() for f in pooled])
            per_rep_wc = np.array([f() for f in within])
            per_rep_wct = np.array([f() for f in within_t])
            wt_draws = boot.run(lambda w: np.mean([f(w) for f in within_t]))
            wtlo, wthi = percentile_ci(wt_draws)
            wtlo98, wthi98 = np.nanpercentile(wt_draws, [0.625, 99.375])   # Bonferroni, 4 methods
            if protocol == "cv":
                variant_store.setdefault(vkey, {"scopes": {}, "y": y, "pid": pid, "cond": cond,
                                                "minute": minute})["scopes"].setdefault(scope, {})[model] = (M, F)
            draws = boot.run(lambda w: np.mean([f(w) for f in pooled]))
            wdraws = boot.run(lambda w: np.mean([f(w) for f in within]))
            lo, hi = percentile_ci(draws)
            wlo, whi = percentile_ci(wdraws)
            macro = []
            for m in M:
                vals = [auc(y[pid == p], m[pid == p]) for p in np.unique(pid)
                        if 0 < y[pid == p].sum() < (pid == p).sum()]
                macro.append(np.mean(vals))
            primary_rows.append({
                "label": label, "scope": scope, "protocol": protocol, "model": model,
                "repeats": len(reps), "windows": len(y), "participants": len(set(pid)),
                "prevalence": y.mean(),
                "auc_mean": per_rep_auc.mean(),
                "auc_sd_repeats": per_rep_auc.std(ddof=1) if len(reps) > 1 else np.nan,
                "auc_min_repeat": per_rep_auc.min(), "auc_max_repeat": per_rep_auc.max(),
                "auc_ci_low": lo, "auc_ci_high": hi,
                "ap_mean": np.mean([average_precision(y, m) for m in M]),
                "participant_macro_auc": np.mean(macro),
                "within_condition_auc": per_rep_wc.mean(), "within_condition_ci_low": wlo,
                "within_condition_ci_high": whi,
                "within_condition_minute_auc": per_rep_wct.mean(),
                "within_condition_minute_ci_low": wtlo, "within_condition_minute_ci_high": wthi,
                "within_condition_minute_ci98_low": wtlo98, "within_condition_minute_ci98_high": wthi98,
                **{f"auc_{s_}": np.mean([auc(y[src == s_], m[src == s_]) for m in M])
                   for s_ in np.unique(src)},
                "mean_score": float(M.mean()),
            })
            for c in np.unique(cond):
                r = cond == c
                cond_rows.append({"label": label, "scope": scope, "protocol": protocol,
                                  "model": model, "condition": c,
                                  "source": src[r][0], "windows": int(r.sum()),
                                  "elevated": int(y[r].sum()), "participants": len(set(pid[r])),
                                  "auc_pooled": np.mean([auc(y[r], m[r]) for m in M]),
                                  "auc_within_fold": np.mean([within_group_auc(y[r], m[r], f[r])
                                                              for m, f in zip(M, F)])})

            # Operating points chosen on validation participants only (Table 5 note)
            alerts_by_point = {}
            for r_ in reps:
                decided = {}
                for _, row in group[group["repeat"] == r_].iterrows():
                    for name, t in pick_thresholds(y[row["val_index"]], row["val_scores"]).items():
                        decided.setdefault(name, np.zeros(len(y), bool))
                        decided[name][row["test_index"]] = row["test_scores"] >= t
                for name, a in decided.items():
                    alerts_by_point.setdefault(name, []).append(a)
            yp, yn = (y == 1).astype(float), (y == 0).astype(float)
            for name, alerts in alerts_by_point.items():
                A = np.asarray(alerts, dtype=float)

                def op_stat(w, A=A):
                    return np.array([[(a * yp * w).sum() / max((yp * w).sum(), 1e-12),
                                      ((1 - a) * yn * w).sum() / max((yn * w).sum(), 1e-12),
                                      (a * w).sum() / w.sum()] for a in A]).mean(axis=0)

                point = op_stat(np.ones(len(y)))
                d = np.asarray([op_stat(w) for w in boot])
                lo_, hi_ = percentile_ci(d)
                op_rows.append({"label": label, "scope": scope, "protocol": protocol,
                                "model": model, "operating_point": name,
                                "sensitivity": point[0], "sens_ci": f"{lo_[0]:.3f}-{hi_[0]:.3f}",
                                "specificity": point[1], "spec_ci": f"{lo_[1]:.3f}-{hi_[1]:.3f}",
                                "alert_rate": point[2], "alert_ci": f"{lo_[2]:.3f}-{hi_[2]:.3f}"})
            log(f"  {model}: AUC {per_rep_auc.mean():.3f} [{lo:.3f}, {hi:.3f}] "
                f"within-condition {per_rep_wc.mean():.3f}")

        # Paired comparisons against the reference model on shared repeats
        ref = args.reference_model
        if ref in matrices:
            for model, M in matrices.items():
                if model == ref:
                    continue
                k = min(len(M), len(matrices[ref]))
                G = [_groups(cond, f) for f in fold_ids[ref][:k]]
                fa = [WeightedAUC(y, a) for a in M[:k]]
                fr = [WeightedAUC(y, r) for r in matrices[ref][:k]]
                ga = [WeightedAUC(y, a, g) for a, g in zip(M[:k], G)]
                gr = [WeightedAUC(y, r, g) for r, g in zip(matrices[ref][:k], G)]

                def delta(w=None, fa=fa, fr=fr):
                    return np.mean([a(w) - r(w) for a, r in zip(fa, fr)])

                def delta_wc(w=None, ga=ga, gr=gr):
                    return np.mean([a(w) - r(w) for a, r in zip(ga, gr)])

                lo, hi = percentile_ci(boot.run(delta))
                wlo, whi = percentile_ci(boot.run(delta_wc))
                cmp_rows.append({"label": label, "scope": scope, "protocol": protocol,
                                 "model": model, "reference": ref,
                                 "shared_repeats": k, "delta_auc": delta(),
                                 "delta_ci_low": lo, "delta_ci_high": hi,
                                 "delta_within_condition_auc": delta_wc(),
                                 "delta_wc_ci_low": wlo, "delta_wc_ci_high": whi})
            log(f"  paired comparisons vs {ref} done")

        # Environment-controlled increment (Reviewer 2, Comment 2)
        levels = np.unique(cond)
        onehot = (cond[:, None] == levels[None, :]).astype(float)
        for model, M in matrices.items():
            if model.startswith("nuisance:"):
                continue
            pairs = [stacked_increment(y, m, onehot, pid, seed=i) for i, m in enumerate(M)]
            fe = [WeightedAUC(y, e) for e, _ in pairs]
            fp = [WeightedAUC(y, p_) for _, p_ in pairs]
            base_tc = np.column_stack([onehot, clock, clock ** 2])
            pairs_t = [stacked_increment(y, m, base_tc, pid, seed=i) for i, m in enumerate(M)]
            fe_t = [WeightedAUC(y, e) for e, _ in pairs_t]
            fp_t = [WeightedAUC(y, p_) for _, p_ in pairs_t]

            def inc(w=None, fe=fe, fp=fp):
                return np.mean([p_(w) - e(w) for e, p_ in zip(fe, fp)])

            def inc_t(w=None, fe=fe_t, fp=fp_t):
                return np.mean([p_(w) - e(w) for e, p_ in zip(fe, fp)])

            lo, hi = percentile_ci(boot.run(inc))
            t_draws = boot.run(inc_t)
            tlo, thi = percentile_ci(t_draws)
            tlo98, thi98 = np.nanpercentile(t_draws, [0.625, 99.375])
            nuis_rows.append({"label": label, "scope": scope, "protocol": protocol, "model": model,
                              "auc_environment_only": np.mean([e() for e in fe]),
                              "auc_environment_plus_model": np.mean([p_() for p_ in fp]),
                              "increment": inc(), "increment_ci_low": lo, "increment_ci_high": hi,
                              "auc_environment_clock": np.mean([e() for e in fe_t]),
                              "auc_environment_clock_plus_model": np.mean([p_() for p_ in fp_t]),
                              "increment_over_env_clock": inc_t(),
                              "increment_env_clock_ci_low": tlo, "increment_env_clock_ci_high": thi,
                              "increment_env_clock_ci98_low": tlo98, "increment_env_clock_ci98_high": thi98,
                              "within_condition_auc": np.mean([
                                  within_group_auc(y, m, _groups(cond, f))
                                  for m, f in zip(M, fold_ids[model])])})
        log("  environment control done")

    # Input variants (channel set x calibration) on identical windows and folds,
    # paired against the published input (all channels, raw) of the same window set.
    var_rows = []
    for (label, sources, min_start), store in variant_store.items():
        scopes = store["scopes"]
        ref_scope = next((sc for sc in scopes if parse_scope(sc)["channels"] == "all"
                          and parse_scope(sc)["calibration"] == "raw"), None)
        if ref_scope is None or len(scopes) < 2:
            continue
        y, pid, cond, minute = store["y"], store["pid"], store["cond"], store["minute"]
        vboot = ParticipantBootstrap(pid, B, seed=7)
        for sc, models in scopes.items():
            for model, (M, F) in models.items():
                if model.startswith("nuisance:") or model not in scopes[ref_scope]:
                    continue
                R, RF = scopes[ref_scope][model]
                k = min(len(M), len(R))
                G = [_groups(cond, f) for f in RF[:k]]
                GT = [np.char.add(g, np.char.add("|", minute)) for g in G]
                stats = {}
                for name, groups in (("pooled", [None] * k), ("within_condition", G),
                                     ("within_condition_minute", GT)):
                    fa = [WeightedAUC(y, a, g) for a, g in zip(M[:k], groups)]
                    fr = [WeightedAUC(y, r, g) for r, g in zip(R[:k], groups)]
                    point = np.mean([a() for a in fa])
                    delta = lambda w=None, fa=fa, fr=fr: np.mean([a(w) - r(w) for a, r in zip(fa, fr)])  # noqa: E731
                    lo, hi = percentile_ci(vboot.run(delta)) if sc != ref_scope else (np.nan, np.nan)
                    stats.update({f"{name}_auc": point, f"{name}_delta": delta(),
                                  f"{name}_delta_ci_low": lo, f"{name}_delta_ci_high": hi})
                var_rows.append({"label": label, "min_start": min_start, "model": model,
                                 "scope": sc, "channels": parse_scope(sc)["channels"],
                                 "calibration": parse_scope(sc)["calibration"],
                                 "reference_scope": ref_scope, "repeats": k, **stats})
        log(f"input-variant comparison for {label} (start >= {min_start}s) done")
    pd.DataFrame(var_rows).to_csv(summary_dir / "input_variants.csv", index=False)

    primary = pd.DataFrame(primary_rows)
    folds = pd.DataFrame(fold_rows)
    primary.to_csv(summary_dir / "primary_estimates.csv", index=False)
    folds.to_csv(summary_dir / "fold_auc.csv", index=False)
    pd.DataFrame(cond_rows).to_csv(summary_dir / "per_condition_auc.csv", index=False)
    pd.DataFrame(op_rows).to_csv(summary_dir / "operating_points_validation_selected.csv", index=False)
    pd.DataFrame(cmp_rows).to_csv(summary_dir / "paired_comparisons.csv", index=False)
    pd.DataFrame(nuis_rows).to_csv(summary_dir / "environment_control.csv", index=False)

    vc = {}
    if len(folds):
        cv = folds[(folds["protocol"] == "cv")]
        cv = cv[cv["elevated"].between(1, cv["test_windows"] - 1)]
        for label, block in cv.groupby("label"):
            for family, prefix in (("classifier_architectures", ("clf:", "cnn_ensemble")),
                                   ("tuned_architectures", ("tuned:",))):
                res = variance_components(block[block["model"].str.startswith(prefix)])
                if res:
                    vc[f"{label}__{family}"] = res
    (summary_dir / "variance_components.json").write_text(json.dumps(vc, indent=2, default=float) + "\n")
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(primary.round(3).to_string(index=False))
    log(f"wrote summaries to {summary_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "summarize"):
        p = sub.add_parser(name)
        p.add_argument("--dataset", type=Path, required=True)
        p.add_argument("--output-dir", type=Path, required=True)
    r = sub.choices["run"]
    r.add_argument("--models", nargs="+", default=["cnn_ensemble"])
    r.add_argument("--labels", nargs="+", default=["fms_gt2"], choices=sorted(LABEL_DEFINITIONS))
    r.add_argument("--scope", default="seated",
                   help="seated | all | <base>-<channels>-<raw|cal60>-s<min_start>, "
                        f"channels in {sorted(CHANNEL_SETS)}")
    r.add_argument("--members", type=int, nargs="+", default=list(ENSEMBLE_SEEDS))
    r.add_argument("--protocol", default="cv", choices=["cv", "reference", "windows"],
                   help="cv: repeated participant K-fold; reference: frozen seed-42 split; "
                        "windows: sample-level K-fold over windows (LiteVR/TruVR protocol)")
    r.add_argument("--repeats", type=int, default=5)
    r.add_argument("--folds", type=int, default=5)
    r.add_argument("--seed", type=int, default=2026)
    r.add_argument("--epochs", type=int, default=100)
    r.add_argument("--arch-seeds", type=int, nargs="+", default=list(ARCH_SEEDS))
    r.add_argument("--trials", type=int, default=8)
    s = sub.choices["summarize"]
    s.add_argument("--bootstrap", type=int, default=2000)
    s.add_argument("--reference-model", default="cnn_ensemble")
    args = parser.parse_args()
    run(args) if args.command == "run" else summarize(args)


if __name__ == "__main__":
    main()
