"""Tests for the revision modules (benchmark v1.1 and repeated participant CV)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis import benchmark_v11 as bv
from analysis.retrain_camera_ready import FEATURE_COLUMNS
from analysis.retrain_camera_ready import build_windows as frozen_build_windows
from analysis.revision_common import (
    auc,
    build_windows,
    make_labels,
    participant_table,
    repeated_participant_folds,
    window_features,
    within_group_auc,
)
from analysis.revision_cv import pick_thresholds, variance_components


def _table(recordings):
    """recordings: list of (source, participant, condition, recording_id, length, fms)."""
    rng = np.random.default_rng(0)
    frames = []
    for source, participant, condition, recording, length, fms in recordings:
        frame = pd.DataFrame(rng.normal(3, 0.3, (length, 14)), columns=FEATURE_COLUMNS)
        frame["elapsed_s"] = np.arange(length)
        frame["fms"] = fms
        frame["participant_id"] = float(participant)
        frame["condition"] = condition
        frame["source_dataset"] = source
        frame["global_participant_id"] = f"{source}_{float(participant)}"
        frame["recording_id"] = recording
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- windows
def test_windows_never_cross_recordings():
    table = _table([("simulations", 1, "sea", "a", 60, 1.0), ("simulations", 1, "sea", "b", 60, 5.0)])
    x, fms, meta = build_windows(table)
    # 60 s -> 3 windows per recording; a merged session would have yielded 7
    assert len(x) == 6
    assert set(np.unique(fms)) == {1.0, 5.0}          # no window mixes the two recordings
    assert meta.groupby("recording_id").size().to_dict() == {"a": 3, "b": 3}


def test_v10_fallback_matches_frozen_windowing():
    table = _table([("terrain", 1, "flat", "t1", 95, 2.0), ("maze", 2, "maze", "m2", 80, 3.0)])
    table = table.drop(columns="recording_id")
    x_new, y_new, _ = build_windows(table)
    x_old, y_old, _ = frozen_build_windows(table)
    np.testing.assert_array_equal(x_new, x_old)
    np.testing.assert_array_equal(y_new, y_old)


# --------------------------------------------------------------------------- labels
def test_label_definitions_respect_scale_floor():
    meta = pd.DataFrame({"source_dataset": ["simulations", "terrain", "terrain"],
                         "session_baseline_fms": [0.0, 1.0, 3.0]})
    fms = np.array([2.5, 2.5, 3.5])
    assert make_labels(fms, meta, "fms_gt2").tolist() == [1, 1, 1]
    assert make_labels(fms, meta, "floor_plus2").tolist() == [1, 0, 1]
    assert make_labels(fms, meta, "floor_plus1").tolist() == [1, 1, 1]
    assert make_labels(fms, meta, "baseline_plus1").tolist() == [1, 1, 0]


# --------------------------------------------------------------------------- folds
def _meta(n_sim=25, n_ter=22, seed=0):
    rng = np.random.default_rng(seed)
    rows, y = [], []
    for source, n in (("simulations", n_sim), ("terrain", n_ter)):
        for p in range(n):
            rate = rng.uniform(0, 0.6)
            for w in range(10):
                rows.append((f"{source}_{p}.0", source))
                y.append(int(rng.random() < rate))
    return pd.DataFrame(rows, columns=["participant_id", "source_dataset"]), np.array(y)


def test_every_participant_tested_once_per_repeat():
    meta, y = _meta()
    info = participant_table(y, meta)
    folds = repeated_participant_folds(info, n_folds=5, n_repeats=4, seed=7)
    everyone = set(info["participant_id"])
    for repeat in range(4):
        tested = [p for f in folds if f.repeat == repeat for p in f.test]
        assert sorted(tested) == sorted(everyone)
    for f in folds:
        assert not (f.train & f.test) and not (f.validation & f.test) and not (f.train & f.validation)
        assert f.train | f.validation | f.test == everyone
        assert len(f.validation) >= 2
        assert {p.split("_")[0] for p in f.test} == {"simulations", "terrain"}


def test_folds_are_deterministic_and_repeats_differ():
    meta, y = _meta()
    info = participant_table(y, meta)
    a = repeated_participant_folds(info, 5, 3, seed=1)
    b = repeated_participant_folds(info, 5, 3, seed=1)
    assert a == b
    assert {f.test for f in a if f.repeat == 0} != {f.test for f in a if f.repeat == 1}


# --------------------------------------------------------------------------- metrics
def test_within_group_auc_is_weighted_mean_and_blind_to_group_offsets():
    rng = np.random.default_rng(3)
    groups = np.repeat(["a", "b", "c"], 100)
    y = (rng.random(300) < np.repeat([0.1, 0.4, 0.7], 100)).astype(int)
    s = rng.random(300) + y * 0.5
    expected = sum(auc(y[groups == g], s[groups == g]) * y[groups == g].sum()
                   * (1 - y[groups == g]).sum() for g in "abc")
    expected /= sum(y[groups == g].sum() * (1 - y[groups == g]).sum() for g in "abc")
    assert within_group_auc(y, s, groups) == pytest.approx(expected)
    # a pure environment score has no within-environment discrimination
    env_score = np.repeat([0.1, 0.4, 0.7], 100)
    assert auc(y, env_score) > 0.7
    assert within_group_auc(y, env_score, groups) == pytest.approx(0.5)


def test_window_features_shape_and_slope():
    x = np.zeros((2, 30, 14), dtype=np.float32)
    x[1, :, 0] = np.arange(30)
    f = window_features(x)
    assert f.shape == (2, 84)
    slope_col = 4 * 14 + 0
    assert f[1, slope_col] == pytest.approx(1.0)
    assert f[0, slope_col] == pytest.approx(0.0)


def test_thresholds_use_validation_scores_only():
    y = np.array([0] * 90 + [1] * 10)
    s = np.linspace(0, 1, 100)
    chosen = pick_thresholds(y, s)
    assert (s[y == 1] >= chosen["sensitivity_ge_0.80"]).mean() >= 0.8
    assert (s[y == 0] < chosen["specificity_ge_0.90"]).mean() >= 0.9


def test_variance_components_detects_dominant_factor():
    rng = np.random.default_rng(0)
    rows = []
    comp_effect = rng.normal(0, 0.08, 20)
    for c in range(20):
        for m, model_effect in enumerate([0.0, 0.01, -0.01]):
            rows.append({"composition": c, "model": f"m{m}",
                         "auc": 0.65 + comp_effect[c] + model_effect + rng.normal(0, 0.005)})
    res = variance_components(pd.DataFrame(rows))
    assert res["share_composition"] > 0.8
    assert res["range_across_compositions_mean"] > res["range_across_models_mean"]


# --------------------------------------------------------------------------- benchmark v1.1
def test_simulations_masking_happens_before_averaging():
    eye = pd.DataFrame({
        "LeftPupilDiameter": [4.0, 4.0, -1.0, -1.0], "RightPupilDiameter": [4.0] * 4,
        "Left_Eye_Openness": [1.0, 1.0, -1.0, -1.0], "Right_Eye_Openness": [1.0] * 4,
        "GazeDirectionWrldSpc_X": [0.0] * 4, "GazeDirectionWrldSpc_Y": [0.0] * 4,
        "GazeDirectionWrldSpc_Z": [1.0, 1.0, 1.0, 0.0],
        "GazeOriginWrldSpc_X": [0.0] * 4, "GazeOriginWrldSpc_Y": [0.0] * 4, "GazeOriginWrldSpc_Z": [0.0] * 4,
    })
    masked, stats = bv.mask_invalid_simulations_samples(eye)
    assert stats["invalid_L"] == 2 and stats["invalid_R"] == 0 and stats["invalid_gaze"] == 1
    assert masked["LeftPupilDiameter"].mean() == pytest.approx(4.0)   # v1.0 would give 1.5


def test_resolve_recordings_rules():
    table = _table([
        ("simulations", 6, "sea", "s6a", 90, 2.0), ("simulations", 6, "sea", "s6b", 70, 3.0),  # restart: keep both
        ("simulations", 9, "sea", "s9a", 90, 2.0),
        ("terrain", 112, "speedbumps", "t112", 80, 2.0),
    ])
    # exact copy of s9a under the same key, and t112 copied under another participant
    copy = table[table.recording_id == "s9a"].assign(recording_id="s9copy")
    other = table[table.recording_id == "t112"].assign(
        recording_id="t113", participant_id=113.0, global_participant_id="terrain_113.0")
    table = pd.concat([table, copy, other], ignore_index=True)
    kept, decisions = bv.resolve_recordings(table)
    assert set(decisions["recording_id"]) == {"s9copy", "t112", "t113"}
    assert set(kept["recording_id"]) == {"s6a", "s6b", "s9a"}


def test_clean_applies_floor_fill_and_passes_verify():
    table = _table([("terrain", 1, "flat", "t1", 40, 2.0)])
    table.loc[3, "pupil_diam_L"] = 0.017
    table.loc[4, "pupil_diam_R"] = -0.9
    table.loc[5, "eye_open_L"] = -0.5
    table.loc[6, ["gaze_dir_world_X", "gaze_dir_world_Y", "gaze_dir_world_Z"]] = 0.0
    cleaned, report = bv.clean(table.copy())
    assert report["pupil_diam_L_below_floor"] == {"terrain": 1}
    assert report["gaze_near_zero"] == {"terrain": 1}
    assert bv.verify(cleaned) == []
    assert cleaned.loc[3, "pupil_diam_L"] == pytest.approx(table.loc[2, "pupil_diam_L"])


def test_blocked_split_is_per_recording_on_v11():
    from analysis.corrected_evaluation import build_window_metadata
    from analysis.split_protocols import assert_no_interval_overlap, chronological_purged_split

    table = _table([("simulations", 6, "sea", "s6a", 420, 2.0), ("simulations", 6, "sea", "s6b", 420, 3.0)])
    _, meta = build_window_metadata(table)
    assert "recording_id" in meta.columns
    assignment = chronological_purged_split(meta)
    assert_no_interval_overlap(meta, assignment)
    for _, group in meta.assign(split=assignment).groupby("recording_id"):
        assert {"train", "validation", "test"} <= set(group["split"])


def test_weighted_auc_matches_resampled_auc():
    from analysis.revision_common import ParticipantBootstrap, WeightedAUC

    rng = np.random.default_rng(11)
    n = 400
    pid = rng.integers(0, 25, n).astype(str)
    groups = rng.integers(0, 4, n)
    y = (rng.random(n) < 0.3).astype(int)
    s = np.round(rng.random(n) + 0.4 * y, 2)          # rounding forces ties
    assert WeightedAUC(y, s)() == pytest.approx(auc(y, s))
    assert WeightedAUC(y, s, groups)() == pytest.approx(within_group_auc(y, s, groups))
    boot = ParticipantBootstrap(pid, replicates=20, seed=3)
    uniq = np.unique(pid)
    for m in boot.multiplicity:
        rows = np.concatenate([np.flatnonzero(pid == p) for p, k in zip(uniq, m) for _ in range(k)])
        w = m[np.unique(pid, return_inverse=True)[1]].astype(float)
        assert WeightedAUC(y, s)(w) == pytest.approx(auc(y[rows], s[rows]))
        assert WeightedAUC(y, s, groups)(w) == pytest.approx(within_group_auc(y[rows], s[rows], groups[rows]))


def test_variant_scopes_share_windows_and_calibration_is_baseline_relative(tmp_path):
    from analysis.revision_common import calibrate_per_recording, parse_scope
    from analysis.revision_cv import prepare

    table = pd.concat([
        _table([("terrain", 1, "flat", "t1", 150, 3.0)]),
        _table([("simulations", 2, "sea", "s2", 170, 1.0)]),
    ], ignore_index=True)
    table = table.iloc[::-1].reset_index(drop=True)          # scrambled row order
    path = tmp_path / "t.csv"
    table.to_csv(path, index=False)
    x_raw, _, y_raw, m_raw = prepare(path, "seated-all-raw-s60", "fms_gt2", tmp_path / "c")
    x_cal, _, y_cal, m_cal = prepare(path, "seated-pupil-cal60-s60", "fms_gt2", tmp_path / "c")
    pd.testing.assert_frame_equal(m_raw, m_cal)
    assert (m_raw["start_index"] >= 60).all()
    assert x_raw.shape[-1] == 14 and x_cal.shape[-1] == 2
    cal = calibrate_per_recording(table)
    first = cal.sort_values("elapsed_s").groupby("recording_id").head(60)
    assert first.groupby("recording_id")["pupil_diam_L"].mean().abs().max() < 1e-6
    assert parse_scope("seated")["channels"] == "all"


@pytest.mark.parametrize("method", ["select", "balanced", "adversarial", "ranking"])
def test_debiasing_methods_train_and_score(method):
    tf = pytest.importorskip("tensorflow")
    from analysis.revision_debias import train_one

    rng = np.random.default_rng(0)
    n = 240
    env = np.repeat(np.array(["a", "b", "c"]), n // 3)
    y = (rng.random(n) < np.repeat([0.1, 0.5, 0.3], n // 3)).astype(int)
    x = rng.normal(0, 1, (n, 30, 5)).astype("float32")
    x[:, :, 0] += y[:, None] * 0.8
    predict, info = train_one(tf, method, x, y, env, x[:60], y[:60], env[:60], seed=1, epochs=3)
    scores = predict(x[:60])
    assert scores.shape == (60,) and np.all((scores >= 0) & (scores <= 1))
    assert info["epochs"] >= 1


# --------------------------------------------------------------------------- published detectors
@pytest.mark.parametrize("variant,params", [("LSTM", None), ("GRU", None), ("MLP", None)])
def test_litevr_architectures_follow_published_tables(variant, params):
    tf = pytest.importorskip("tensorflow")
    from analysis.litevr import build_litevr

    model = build_litevr(tf, variant)
    assert model.output_shape == (None, 1)
    kinds = [type(layer).__name__ for layer in model.layers]
    if variant == "LSTM":   # Table 8: one LSTM-64 then Dense-32
        assert kinds.count("LSTM") == 1 and model.layers[1].units == 64
    if variant == "GRU":    # Table 9: GRU-32 then GRU-64
        assert [layer.units for layer in model.layers if type(layer).__name__ == "GRU"] == [32, 64]
    if variant == "MLP":    # Table 10: 64, 32, 16
        dense = [layer.units for layer in model.layers if type(layer).__name__ == "Dense"]
        assert dense == [64, 32, 16, 1]


def test_window_folds_test_every_window_once_and_mix_participants():
    from analysis.litevr import window_folds

    y = np.r_[np.zeros(80, int), np.ones(20, int)]
    splits = window_folds(y, n_folds=10, n_repeats=2, seed=3)
    assert len(splits) == 20
    for r in (0, 1):
        tested = np.concatenate([te for rep, _, _, _, te in splits if rep == r])
        assert np.array_equal(np.sort(tested), np.arange(100))
    for _, _, tr, va, te in splits:
        assert not (set(tr) & set(te)) and not (set(va) & set(te)) and not (set(tr) & set(va))


def test_ebm_scores_window_features():
    pytest.importorskip("interpret")
    from analysis.revision_cv import model_features

    rng = np.random.default_rng(0)
    y = (rng.random(120) < 0.3).astype(int)
    x = rng.normal(0, 1, (120, 30, 14)).astype("float32")
    x[:, :, 0] += y[:, None]
    s_va, s_te, info = model_features("ebm", x[:80], y[:80], x[80:100], x[100:])
    assert s_te.shape == (20,) and info["n_features"] == 84
    assert auc(y[100:], s_te) > 0.7


def test_windows_protocol_runs_and_summarizes(tmp_path):
    """End-to-end: sample-level protocol through run and summarize (CPU models)."""
    import argparse

    from analysis.revision_cv import run, summarize

    recs = []
    for p in range(24):
        for cond, fms in (("sea", 1.0), ("coaster", 4.0)):
            recs.append(("simulations", p, cond, f"s{p}{cond}", 120, fms if p % 2 else 1.0))
    table = _table(recs)
    path = tmp_path / "t.csv"
    table.to_csv(path, index=False)
    out = tmp_path / "out"
    base = dict(dataset=path, output_dir=out, labels=["fms_gt2"], scope="simulations",
                members=[42], repeats=1, folds=4, seed=0, epochs=1, arch_seeds=[42], trials=1)
    for protocol in ("windows", "cv"):
        run(argparse.Namespace(**base, models=["feat:logreg", "nuisance:condition"], protocol=protocol))
    summarize(argparse.Namespace(dataset=path, output_dir=out, bootstrap=50,
                                 reference_model="nuisance:condition"))
    primary = pd.read_csv(out / "summary" / "primary_estimates.csv")
    assert set(primary["protocol"]) == {"windows", "cv"}
    env = pd.read_csv(out / "summary" / "environment_control.csv")
    assert {"protocol", "within_condition_auc"} <= set(env.columns)
    nuis = primary[primary.model == "nuisance:condition"]
    assert np.allclose(nuis["within_condition_auc"].dropna(), 0.5)


def test_litevr_regression_and_r2_summary(tmp_path):
    pytest.importorskip("tensorflow")
    import argparse

    from analysis.litevr import summarize_regression, weighted_r2
    from analysis.revision_cv import run

    assert weighted_r2(np.array([1.0, 2, 3]), np.array([1.0, 2, 3])) == 1.0
    assert abs(weighted_r2(np.array([1.0, 2, 3]), np.array([2.0, 2, 2]))) < 1e-12
    recs = [("simulations", p, c, f"s{p}{c}", 100, f) for p in range(24)
            for c, f in (("sea", 1.0), ("coaster", 4.0 if p % 2 else 1.0))]
    path = tmp_path / "t.csv"
    _table(recs).to_csv(path, index=False)
    out = tmp_path / "out"
    run(argparse.Namespace(dataset=path, output_dir=out, labels=["fms_gt2"], scope="simulations",
                           members=[42], repeats=1, folds=3, seed=0, epochs=1, arch_seeds=[42],
                           trials=1, models=["litevr_reg:MLP"], protocol="windows"))
    table = summarize_regression(path, out, bootstrap=20)
    assert list(table["model"]) == ["litevr_reg:MLP"] and np.isfinite(table["r2"]).all()
