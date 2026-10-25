#!/usr/bin/env python3
"""How much does each input channel group reveal about the VR environment?

A linear probe predicts the environment (8 seated conditions) from per-window
channel statistics for participants it has never seen (participant-grouped
5-fold CV). High accuracy means a detector can recognise the scene -- and,
because sickness prevalence differs strongly between scenes, can score well
without measuring sickness. Run for raw and per-user calibrated inputs.

    python -m analysis.environment_probe --dataset v1_1.csv --output probe.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.retrain_camera_ready import FEATURE_COLUMNS
from analysis.revision_common import CHANNEL_SETS, SEATED_SOURCES, build_windows, calibrate_per_recording

GROUPS = {
    "pupil diameter": ["pupil_diam_L", "pupil_diam_R"],
    "eye openness": ["eye_open_L", "eye_open_R"],
    "gaze direction": ["gaze_dir_world_X", "gaze_dir_world_Y", "gaze_dir_world_Z"],
    "gaze origin (world position)": ["gaze_origin_world_X", "gaze_origin_world_Y", "gaze_origin_world_Z"],
    "head orientation": ["head_quat_X", "head_quat_Y", "head_quat_Z", "head_quat_W"],
    **{f"set:{k}": v for k, v in CHANNEL_SETS.items()},
}


def probe(frame: pd.DataFrame, min_start: int = 60, folds: int = 5) -> pd.DataFrame:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    x, _, meta = build_windows(frame)
    keep = (meta["source_dataset"].isin(SEATED_SOURCES) & (meta["start_index"] >= min_start)).to_numpy()
    x, meta = x[keep], meta[keep].reset_index(drop=True)
    env = (meta["source_dataset"] + ":" + meta["condition"]).to_numpy()
    pid = meta["participant_id"].to_numpy()
    stats = np.concatenate([x.mean(axis=1), x.std(axis=1)], axis=1)
    rows = []
    for name, channels in GROUPS.items():
        idx = [FEATURE_COLUMNS.index(c) for c in channels]
        cols = idx + [i + len(FEATURE_COLUMNS) for i in idx]
        correct = np.zeros(len(env), bool)
        for tr, te in GroupKFold(folds).split(stats, env, pid):
            model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
            model.fit(stats[tr][:, cols], env[tr])
            correct[te] = model.predict(stats[te][:, cols]) == env[te]
        rows.append({"channels": name, "n_channels": len(channels), "environment_accuracy": correct.mean()})
    out = pd.DataFrame(rows)
    out["chance_majority"] = pd.Series(env).value_counts(normalize=True).max()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.dataset)
    raw = probe(frame).assign(calibration="raw")
    cal = probe(calibrate_per_recording(frame)).assign(calibration="cal60")
    table = pd.concat([raw, cal], ignore_index=True)
    table.to_csv(args.output, index=False)
    print(table.pivot(index="channels", columns="calibration", values="environment_accuracy").round(3))


if __name__ == "__main__":
    main()
