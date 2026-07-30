"""Tests for the composition-robustness re-run and the harmonization module.

The key guarantees: the vectorized bootstrap AUC is the same estimator used for
the frozen headline result, and the harmonization module reproduces the
published preprocessing behaviour including its documented ordering defect.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.composition_robustness import (
    _fast_auc,
    balanced_class_weights,
    participant_strata,
    stratified_participant_split,
    summarize,
)
from analysis.corrected_evaluation import roc_auc


def test_fast_auc_matches_reference_on_random_data():
    rng = np.random.default_rng(0)
    for _ in range(50):
        n = int(rng.integers(20, 400))
        y = (rng.random(n) < rng.uniform(0.05, 0.6)).astype(int)
        if y.sum() in (0, n):
            continue
        scores = rng.random(n)
        assert _fast_auc(y, scores) == pytest.approx(roc_auc(y, scores), abs=1e-12)


def test_fast_auc_matches_reference_with_heavy_ties():
    """Tied scores are where a rank-based shortcut would diverge if wrong."""
    rng = np.random.default_rng(1)
    for _ in range(50):
        n = int(rng.integers(20, 300))
        y = (rng.random(n) < 0.3).astype(int)
        if y.sum() in (0, n):
            continue
        scores = rng.integers(0, 4, size=n).astype(float)
        assert _fast_auc(y, scores) == pytest.approx(roc_auc(y, scores), abs=1e-12)


def test_fast_auc_undefined_for_single_class():
    scores = np.array([0.1, 0.2, 0.3])
    assert np.isnan(_fast_auc(np.zeros(3, dtype=int), scores))
    assert np.isnan(_fast_auc(np.ones(3, dtype=int), scores))


def _toy_metadata(n_per_source=(("maze", 12), ("simulations", 10), ("terrain", 8))):
    rows, labels = [], []
    rng = np.random.default_rng(3)
    for source, count in n_per_source:
        for participant in range(count):
            for window in range(6):
                rows.append((f"{source}_{participant}", source, source, window, window + 30))
                labels.append(float(rng.uniform(1.0, 4.0)))
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
    return np.asarray(labels), metadata


def test_split_partitions_participants_without_overlap():
    labels, metadata = _toy_metadata()
    info = participant_strata(labels, metadata)
    everyone = set(metadata["participant_id"])
    for seed in range(10):
        train, validation, test = stratified_participant_split(info, seed)
        assert not (train & validation)
        assert not (train & test)
        assert not (validation & test)
        assert train | validation | test == everyone


def test_split_keeps_every_source_in_the_test_fold():
    labels, metadata = _toy_metadata()
    info = participant_strata(labels, metadata)
    for seed in range(10):
        _, _, test = stratified_participant_split(info, seed)
        sources = {pid.rsplit("_", 1)[0] for pid in test}
        assert sources == {"maze", "simulations", "terrain"}


def test_split_varies_with_seed():
    labels, metadata = _toy_metadata()
    info = participant_strata(labels, metadata)
    folds = {frozenset(stratified_participant_split(info, seed)[2]) for seed in range(10)}
    assert len(folds) > 1, "compositions must actually redraw the test fold"


def test_balanced_class_weights_equalize_total_mass():
    labels = np.array([0] * 90 + [1] * 10)
    weights = balanced_class_weights(labels)
    assert weights[0] * 90 == pytest.approx(weights[1] * 10)


def test_summarize_reports_distribution_and_no_significance_test():
    records = [
        {"status": "complete", "pooled_auc": 0.60, "cluster_ci_lower": 0.52,
         "test_participants": 9},
        {"status": "complete", "pooled_auc": 0.80, "cluster_ci_lower": 0.48,
         "test_participants": 9},
        {"status": "skipped_single_class_test"},
    ]
    summary = summarize(records)
    assert summary["n_compositions_complete"] == 2
    assert summary["n_with_ci_above_chance"] == 1
    assert summary["range_auc"] == pytest.approx(0.20)
    assert "p_value" not in summary
    assert "t_statistic" not in summary
    assert "independent" in summary["note"]


# --------------------------------------------------------------------------
# Harmonization
# --------------------------------------------------------------------------
from analysis.harmonize import harmonize, mask_and_fill_pupils, renormalize_gaze, verify


def _toy_pooled():
    return pd.DataFrame(
        {
            "global_participant_id": ["p1"] * 4 + ["p2"] * 3,
            "condition": ["c"] * 7,
            "elapsed_s": [0, 1, 2, 3, 0, 1, 2],
            # p1 second row is a raw sentinel; p2 first row is averaged residue
            "pupil_diam_L": [3.0, -1.0, 3.4, 3.6, -0.92, 2.9, 3.1],
            "pupil_diam_R": [3.1, 3.2, -1.0, 3.5, 2.8, 2.9, 3.0],
            "gaze_dir_world_X": [2.0, 0.0, 0.0, 1.0, 3.0, 0.0, 0.0],
            "gaze_dir_world_Y": [0.0, 2.0, 0.0, 1.0, 0.0, 4.0, 0.0],
            "gaze_dir_world_Z": [0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 5.0],
        }
    )


def test_gaze_renormalization_gives_unit_vectors():
    out = renormalize_gaze(_toy_pooled())
    mag = np.sqrt(
        out.gaze_dir_world_X**2 + out.gaze_dir_world_Y**2 + out.gaze_dir_world_Z**2
    )
    assert np.allclose(mag, 1.0, atol=1e-9)


def test_exact_sentinel_masked_and_filled_within_session():
    out = mask_and_fill_pupils(_toy_pooled())
    assert out.pupil_diam_L.isna().sum() == 0
    assert not (out.pupil_diam_L == -1).any()
    assert not (out.pupil_diam_R == -1).any()


def test_published_rule_leaves_averaged_residue():
    """The documented ordering defect: -0.92 is not caught by an == -1 test."""
    out = mask_and_fill_pupils(_toy_pooled(), strict=False)
    assert (out.pupil_diam_L < 0).sum() == 1


def test_strict_rule_removes_averaged_residue():
    out = mask_and_fill_pupils(_toy_pooled(), strict=True)
    assert (out.pupil_diam_L < 0).sum() == 0
    assert out.pupil_diam_L.isna().sum() == 0


def test_harmonize_is_idempotent():
    once = harmonize(_toy_pooled())
    twice = harmonize(once)
    cols = ["pupil_diam_L", "pupil_diam_R", "gaze_dir_world_X"]
    assert np.allclose(once[cols].to_numpy(), twice[cols].to_numpy(), atol=1e-12)


def test_verify_reports_clean_table_as_clean():
    report = verify(harmonize(_toy_pooled()))
    assert report["problems"] == []
    assert report["idempotent"] is True


# --------------------------------------------------------------------------
# Raw loader: duplicate-session detection
# --------------------------------------------------------------------------
from analysis.build_master import SCHEMA, discover_sim_sessions, merge_sources


def test_discover_sim_sessions_warns_on_duplicate_participants(tmp_path, capsys):
    for folder in ("09_aaa", "09_2_bbb", "10_ccc"):
        session = tmp_path / folder
        session.mkdir()
        (session / "eye_tracking_data-x.csv").write_text("x\n")
        (session / "head_tracking_x.csv").write_text("x\n")
    sessions = discover_sim_sessions(str(tmp_path))
    assert len(sessions) == 3
    assert sorted(p for p, _, _ in sessions) == [9, 9, 10]
    assert "WARNING" in capsys.readouterr().out


def test_discover_sim_sessions_is_sorted_for_stability(tmp_path):
    for folder in ("30_z", "10_a", "20_m"):
        session = tmp_path / folder
        session.mkdir()
        (session / "eye_tracking_data-x.csv").write_text("x\n")
        (session / "head_tracking_x.csv").write_text("x\n")
    order = [p for p, _, _ in discover_sim_sessions(str(tmp_path))]
    assert order == sorted(order), "enumeration must be deterministic"


def test_merge_preserves_float_participant_identifiers():
    """The published ids look like maze_1.0; that upcast must not be 'fixed'."""
    def block(source, pid):
        row = {c: 0.0 for c in SCHEMA}
        row.update(participant_id=pid, condition="c", source_dataset=source, fms=3.0)
        return pd.DataFrame([row])

    pooled = merge_sources(block("terrain", 101), block("maze", 1), block("simulations", 9.0))
    assert set(pooled.global_participant_id) == {"terrain_101.0", "maze_1.0", "simulations_9.0"}
    assert pooled.elevated.all()
