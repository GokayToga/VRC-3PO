#!/usr/bin/env python3
"""Shared building blocks for the first-round revision analyses (IEEE Access).

Everything here is deterministic, CPU-only and unit-tested; the GPU training
loop lives in ``analysis.revision_cv``.

What changed relative to the frozen submission pipeline
-------------------------------------------------------
1. **Recording-aware windowing.** Benchmark v1.1 carries a ``recording_id``
   column. Windows are cut within a recording, so two Simulations folders filed
   under the same participant and environment can no longer be interleaved.
   A table without ``recording_id`` (v1.0) falls back to the published
   participant x condition sessions, so the published window set is still
   reproducible.
2. **Label definitions.** Besides the published ``FMS > 2`` rule, three
   source-aware alternatives answer Reviewer 2, Comment 3. Each is a
   function of the window-mean FMS and the window's source only; none looks at
   model output.
3. **Repeated participant-level cross-validation.** Every participant in the
   evaluation scope is tested exactly once per repeat, so a repeat yields one
   pooled out-of-fold prediction for every window. This replaces the single
   fixed split as the primary estimate (Reviewer 2, Comment 1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from analysis.composition_robustness import _fast_auc
from analysis.retrain_camera_ready import FEATURE_COLUMNS, WINDOW_LENGTH, WINDOW_STRIDE

ELEVATED_THRESHOLD = 2.0
SEATED_SOURCES = ("simulations", "terrain")
ALL_SOURCES = ("maze", "simulations", "terrain")
SCOPES = {"seated": SEATED_SOURCES, "all": ALL_SOURCES,
          # the Islam et al. recordings alone, on which LiteVR was trained
          "simulations": ("simulations",)}

# Lowest point of each source's FMS response scale (the "no sickness" anchor).
# Simulations used a 0-10 variant; Maze a 1-10 verbal rating; observed Terrain
# ratings span 1-6 and never reach 0. See the Source Datasets subsection.
SCALE_FLOOR = {"simulations": 0.0, "terrain": 1.0, "maze": 1.0}


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------
def session_keys(df: pd.DataFrame) -> list[str]:
    """Columns that define one continuous recording."""
    if "recording_id" in df.columns:
        return ["recording_id"]
    return ["global_participant_id", "condition"]


def build_windows(df: pd.DataFrame):
    """30 s windows with 15 s stride, never crossing a recording boundary.

    Returns ``x`` (n, 30, 14), ``fms`` (n,) window-mean FMS, and metadata with
    one row per window. ``participant_id`` in the metadata is the global
    participant identifier, as in the frozen pipeline.
    """
    keys = session_keys(df)
    features, labels, rows = [], [], []
    for _, session in df.groupby(keys, sort=False):
        session = session.sort_values("elapsed_s").reset_index(drop=True)
        values = session[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        fms = session["fms"].to_numpy(dtype=np.float32)
        participant = str(session["global_participant_id"].iloc[0])
        condition = str(session["condition"].iloc[0])
        source = str(session["source_dataset"].iloc[0])
        recording = (
            str(session["recording_id"].iloc[0])
            if "recording_id" in session.columns
            else f"{participant}|{condition}"
        )
        baseline = float(fms[0]) if len(fms) else float("nan")
        for start in range(0, len(session) - WINDOW_LENGTH + 1, WINDOW_STRIDE):
            end = start + WINDOW_LENGTH
            features.append(values[start:end])
            labels.append(float(fms[start:end].mean()))
            rows.append((participant, condition, source, recording, start, end, baseline))
    metadata = pd.DataFrame(
        rows,
        columns=[
            "participant_id",
            "condition",
            "source_dataset",
            "recording_id",
            "start_index",
            "end_index",
            "session_baseline_fms",
        ],
    )
    return (
        np.stack(features).astype(np.float32),
        np.asarray(labels, dtype=np.float32),
        metadata,
    )


# ---------------------------------------------------------------------------
# Label definitions (Reviewer 2, Comment 3)
# ---------------------------------------------------------------------------
def _floor(metadata: pd.DataFrame) -> np.ndarray:
    unknown = set(metadata["source_dataset"]) - set(SCALE_FLOOR)
    if unknown:
        raise ValueError(f"No scale floor for sources {sorted(unknown)}")
    return metadata["source_dataset"].map(SCALE_FLOOR).to_numpy(dtype=float)


LABEL_DEFINITIONS = {
    # Published rule: window-mean FMS strictly above 2 on every source.
    "fms_gt2": lambda fms, meta: fms > ELEVATED_THRESHOLD,
    # Same distance (> 2 steps) above each source's own no-sickness anchor:
    # Simulations > 2, Terrain and Maze > 3.
    "floor_plus2": lambda fms, meta: (fms - _floor(meta)) > 2.0,
    # Lenient anchor-relative rule: > 1 step above the floor.
    "floor_plus1": lambda fms, meta: (fms - _floor(meta)) > 1.0,
    # Individual-baseline rule: > 1 step above the first rating of the same
    # recording, which removes both scale-floor and personal-offset differences.
    "baseline_plus1": lambda fms, meta: (
        fms - meta["session_baseline_fms"].to_numpy(dtype=float)
    ) > 1.0,
}


def make_labels(fms: np.ndarray, metadata: pd.DataFrame, definition: str) -> np.ndarray:
    if definition not in LABEL_DEFINITIONS:
        raise ValueError(f"Unknown label definition {definition!r}; "
                         f"choose from {sorted(LABEL_DEFINITIONS)}")
    return np.asarray(LABEL_DEFINITIONS[definition](fms, metadata), dtype=int)


# ---------------------------------------------------------------------------
# Repeated participant-level cross-validation
# ---------------------------------------------------------------------------
def participant_table(binary: np.ndarray, metadata: pd.DataFrame) -> pd.DataFrame:
    """One row per participant with a source x severity-tier stratum.

    Same stratification idea as the frozen split: participants are ranked by
    elevated-window rate within their source and cut at the median.
    """
    rate = (
        pd.DataFrame({"participant_id": metadata["participant_id"], "elevated": binary})
        .groupby("participant_id", sort=True)["elevated"]
        .mean()
        .rename("elevated_rate")
    )
    info = (
        metadata.groupby("participant_id", sort=True)["source_dataset"]
        .first()
        .to_frame()
        .join(rate)
        .reset_index()
    )
    info["rank"] = info.groupby("source_dataset")["elevated_rate"].rank(method="first")
    info["tier"] = info.groupby("source_dataset")["rank"].transform(
        lambda r: np.where(r <= np.ceil(len(r) / 2.0), "low", "high")
    )
    info["stratum"] = info["source_dataset"] + "_" + info["tier"]
    return info.drop(columns="rank")


@dataclass(frozen=True)
class Fold:
    repeat: int
    fold: int
    train: frozenset
    validation: frozenset
    test: frozenset


def repeated_participant_folds(
    info: pd.DataFrame,
    n_folds: int = 5,
    n_repeats: int = 5,
    seed: int = 2026,
    validation_fraction: float = 0.125,
) -> list[Fold]:
    """Stratified participant-level K-fold, repeated with fresh shuffles.

    Within each stratum participants are shuffled and dealt round-robin to the
    K folds, so every participant is tested exactly once per repeat and every
    fold keeps both sources and both severity tiers. The validation set of a
    fold is drawn from its training participants, stratum by stratum, and is
    used only for early stopping, threshold selection and tuning.
    """
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    folds: list[Fold] = []
    for repeat in range(n_repeats):
        rng = np.random.default_rng([seed, repeat])
        assignment: dict[str, int] = {}
        offset = int(rng.integers(n_folds))
        for _, group in info.groupby("stratum", sort=True):
            members = group["participant_id"].to_numpy()[rng.permutation(len(group))]
            for position, participant in enumerate(members):
                assignment[participant] = (position + offset) % n_folds
            offset = (offset + len(members)) % n_folds
        for fold in range(n_folds):
            test = {p for p, f in assignment.items() if f == fold}
            remaining = info[~info["participant_id"].isin(test)]
            validation: set[str] = set()
            for _, group in remaining.groupby("stratum", sort=True):
                members = group["participant_id"].to_numpy()[rng.permutation(len(group))]
                n_val = int(round(validation_fraction * len(members)))
                if len(members) >= 4:
                    n_val = max(1, n_val)
                validation.update(members[:n_val])
            train = set(remaining["participant_id"]) - validation
            folds.append(Fold(repeat, fold, frozenset(train), frozenset(validation), frozenset(test)))
    return folds


# ---------------------------------------------------------------------------
# Standardization and hand-crafted features
# ---------------------------------------------------------------------------
def standardize(x: np.ndarray, train_index: np.ndarray, other_indices: list):
    flat = x[train_index].reshape(-1, x.shape[-1]).astype(np.float64)
    mean = flat.mean(axis=0)
    scale = flat.std(axis=0)
    scale[scale == 0] = 1.0

    def transform(index):
        return ((x[index].astype(np.float64) - mean) / scale).astype(np.float32)

    return transform(train_index), [transform(i) for i in other_indices]


FEATURE_STATS = ("mean", "std", "min", "max", "slope", "delta")


def window_features(x: np.ndarray) -> np.ndarray:
    """Per-channel summary statistics of each window (14 x 6 = 84 features).

    These are the conventional hand-crafted descriptors used by feature-based
    cybersickness detectors: level, variability, range and linear trend of each
    signal within the window.
    """
    t = np.arange(x.shape[1], dtype=np.float64)
    t = (t - t.mean()) / ((t - t.mean()) ** 2).sum()
    parts = [
        x.mean(axis=1),
        x.std(axis=1),
        x.min(axis=1),
        x.max(axis=1),
        np.einsum("ntc,t->nc", x.astype(np.float64), t),
        x[:, -1, :] - x[:, 0, :],
    ]
    return np.concatenate(parts, axis=1).astype(np.float32)


def feature_names() -> list[str]:
    return [f"{channel}__{stat}" for stat in FEATURE_STATS for channel in FEATURE_COLUMNS]


# ---------------------------------------------------------------------------
# Metrics with participant-cluster uncertainty
# ---------------------------------------------------------------------------
def auc(y: np.ndarray, s: np.ndarray) -> float:
    return _fast_auc(np.asarray(y, dtype=int), np.asarray(s, dtype=float))


def within_group_auc(y: np.ndarray, s: np.ndarray, groups: np.ndarray) -> float:
    """AUC over pairs whose two windows share a group (e.g. the same condition).

    Equal to the per-group AUCs averaged with weights n_pos * n_neg. With
    ``groups`` = condition this is the environment-controlled AUC: a model that
    only recognizes the environment scores exactly 0.5 on it.
    """
    numerator, denominator = 0.0, 0.0
    for g in np.unique(groups):
        rows = groups == g
        n_pos = int(y[rows].sum())
        n_neg = int(rows.sum()) - n_pos
        if n_pos == 0 or n_neg == 0:
            continue
        numerator += auc(y[rows], s[rows]) * n_pos * n_neg
        denominator += n_pos * n_neg
    return numerator / denominator if denominator else float("nan")


def average_precision(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-np.asarray(s, dtype=float), kind="mergesort")
    hits = y[order]
    precision = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float((precision * hits).sum() / hits.sum())


def participant_bootstrap(
    statistic,
    participant_ids: np.ndarray,
    replicates: int = 2000,
    seed: int = 0,
):
    """Percentile interval of ``statistic(rows)`` over participant resamples.

    ``statistic`` receives an integer index array (with repeats) and returns a
    float, or an array of floats for paired / multi-repeat statistics.
    """
    rng = np.random.default_rng(seed)
    participants = np.unique(participant_ids)
    rows_for = {p: np.flatnonzero(participant_ids == p) for p in participants}
    draws = []
    for _ in range(replicates):
        sample = rng.choice(participants, size=len(participants), replace=True)
        rows = np.concatenate([rows_for[p] for p in sample])
        draws.append(statistic(rows))
    draws = np.asarray(draws, dtype=float)
    return draws


def percentile_ci(draws: np.ndarray, axis: int = 0):
    return (
        np.nanpercentile(draws, 2.5, axis=axis),
        np.nanpercentile(draws, 97.5, axis=axis),
    )


# ---------------------------------------------------------------------------
# Fast participant-cluster bootstrap
# ---------------------------------------------------------------------------
class WeightedAUC:
    """ROC AUC under arbitrary non-negative window weights, optionally within groups.

    Resampling participants with replacement is equivalent to giving every
    window the multiplicity of its participant in the draw. Scores are sorted
    once here; each call then costs two ``bincount`` passes and a cumulative
    sum, instead of a fresh sort of a concatenated resample. Tied scores get
    half credit, exactly as in :func:`auc`. With ``groups`` only pairs inside
    the same group are compared, which reproduces :func:`within_group_auc`.
    """

    def __init__(self, y, scores, groups=None):
        y = np.asarray(y, dtype=int)
        scores = np.asarray(scores, dtype=float)
        n = len(y)
        g = np.zeros(n, dtype=int) if groups is None else np.unique(
            np.asarray(groups).astype(str), return_inverse=True)[1]
        order = np.lexsort((scores, g))
        gs, ss = g[order], scores[order]
        new_unit = np.r_[True, (gs[1:] != gs[:-1]) | (ss[1:] != ss[:-1])]
        unit_sorted = np.cumsum(new_unit) - 1
        self.unit = np.empty(n, dtype=int)
        self.unit[order] = unit_sorted
        self.n_units = int(unit_sorted[-1] + 1) if n else 0
        self.unit_group = gs[new_unit]
        first = np.r_[True, self.unit_group[1:] != self.unit_group[:-1]]
        self.group_start = np.maximum.accumulate(np.where(first, np.arange(self.n_units), 0))
        self.positive = (y == 1).astype(float)
        self.negative = 1.0 - self.positive

    def __call__(self, weights=None) -> float:
        w = np.ones(len(self.unit)) if weights is None else weights
        wp = np.bincount(self.unit, weights=w * self.positive, minlength=self.n_units)
        wn = np.bincount(self.unit, weights=w * self.negative, minlength=self.n_units)
        cumulative = np.cumsum(wn)
        below = cumulative - wn - np.r_[0.0, cumulative][self.group_start]
        numerator = float((wp * (below + 0.5 * wn)).sum())
        denominator = float((np.bincount(self.unit_group, weights=wp)
                             * np.bincount(self.unit_group, weights=wn)).sum())
        return numerator / denominator if denominator > 0 else float("nan")


class ParticipantBootstrap:
    """Participant multiplicity weights for a fixed set of replicate draws.

    The same draws are reused for every statistic of one evaluation set, so
    paired differences (model A - model B) are computed on identical resamples.
    """

    def __init__(self, participant_ids, replicates=2000, seed=0):
        uniq, index = np.unique(np.asarray(participant_ids).astype(str), return_inverse=True)
        rng = np.random.default_rng(seed)
        draws = rng.integers(0, len(uniq), size=(replicates, len(uniq)))
        self.multiplicity = np.stack([np.bincount(d, minlength=len(uniq)) for d in draws])
        self.index = index

    def __iter__(self):
        for m in self.multiplicity:
            yield m[self.index].astype(float)

    def run(self, statistic) -> np.ndarray:
        return np.asarray([statistic(w) for w in self], dtype=float)


# ---------------------------------------------------------------------------
# Input variants: channel subsets and per-user calibration (environment shortcut)
# ---------------------------------------------------------------------------
CHANNEL_SETS = {
    # all 14 published channels
    "all": list(FEATURE_COLUMNS),
    # drop world-space gaze origin: it is the eye position in scene coordinates,
    # i.e. a location tag that identifies the environment for unseen people
    "no_origin": [c for c in FEATURE_COLUMNS if not c.startswith("gaze_origin")],
    "pupil": ["pupil_diam_L", "pupil_diam_R"],
    "pupil_head": ["pupil_diam_L", "pupil_diam_R",
                   "head_quat_X", "head_quat_Y", "head_quat_Z", "head_quat_W"],
}
CALIBRATION_SECONDS = 60
CALIBRATION_CLIP = 10.0


def parse_scope(scope: str) -> dict:
    """``seated`` or ``seated-<channels>-<calibration>-s<min_start>``.

    calibration: ``raw`` (published) or ``cal60`` (z-score every channel of a
    recording by its own first 60 s, clipped at +/-10 SD). ``min_start`` drops
    windows that start before that second, so calibrated and raw variants are
    scored on identical windows that never overlap the calibration period.
    """
    parts = scope.split("-")
    spec = {"sources": SCOPES[parts[0]], "channels": "all", "calibration": "raw", "min_start": 0}
    if len(parts) > 1:
        spec["channels"], spec["calibration"] = parts[1], parts[2]
        spec["min_start"] = int(parts[3].lstrip("s"))
    if spec["channels"] not in CHANNEL_SETS or spec["calibration"] not in ("raw", "cal60"):
        raise ValueError(f"bad scope {scope!r}")
    return spec


def calibrate_per_recording(df: pd.DataFrame, seconds: int = CALIBRATION_SECONDS,
                            clip: float = CALIBRATION_CLIP) -> pd.DataFrame:
    """Express every channel relative to the recording's own first ``seconds``.

    Deployable: a headset can record a one-minute baseline at session start.
    Uses no labels and no data after the baseline period for the reference.
    """
    df = df.copy()
    keys = session_keys(df)
    df = df.sort_values(keys + ["elapsed_s"]).reset_index(drop=True)
    for _, idx in df.groupby(keys, sort=False).indices.items():
        values = df.loc[idx, FEATURE_COLUMNS].to_numpy(dtype=float)
        base = values[:seconds]
        z = (values - base.mean(axis=0)) / (base.std(axis=0) + 1e-6)
        df.loc[idx, FEATURE_COLUMNS] = np.clip(z, -clip, clip)
    return df
