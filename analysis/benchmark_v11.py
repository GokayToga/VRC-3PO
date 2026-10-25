#!/usr/bin/env python3
"""Build benchmark v1.1: the corrected harmonized table (Reviewer 2, Comment 7).

v1.0 (``vrc3po_master_dataset_fixed.csv``) is produced by ``build_master`` +
``harmonize`` and is kept unchanged so that the submitted numbers stay
reproducible. v1.1 corrects the three preprocessing defects documented there:

1. **Simulations sentinel residue.** Invalid eye samples (pupil diameter or
   openness < 0, i.e. the -1 marker, and near-zero gaze vectors) are masked at
   the native ~24 Hz rate *before* one-second averaging, exactly as Maze
   already did with its validity flags. v1.0 masked only after averaging, so
   partially invalid bins kept averaged marker values.
2. **Terrain sentinel residue.** Terrain is released pre-aggregated at 1 Hz, so
   partially invalid bins cannot be rebuilt. Pupil values below a 1.0 mm
   plausibility floor -- below the minimum observed in either natively masked
   source -- and negative openness values are masked and filled within the
   recording. Residue above the floor is undetectable and is reported as a
   limitation.
3. **Duplicate / multiple recordings.** Every recording folder keeps its own
   ``recording_id``; windows never cross a recording. Exact or prefix-identical
   copies are removed; identical content filed under two different
   participant/condition labels (Terrain 112/113 speedbumps) is removed
   entirely because the label is ambiguous; genuine restarts of one exposure
   are kept as separate recordings.

Participant identifiers keep the published ``<source>_<code>.0`` form, so the
frozen seed-42 split can be re-derived on v1.1.

USAGE
-----
    python -m analysis.benchmark_v11 stage --raw-dir RAW --stage-dir STAGE --source terrain
    python -m analysis.benchmark_v11 stage --raw-dir RAW --stage-dir STAGE --source simulations
    python -m analysis.benchmark_v11 stage --raw-dir RAW --stage-dir STAGE --source maze
    python -m analysis.benchmark_v11 finalize --stage-dir STAGE --output vrc3po_benchmark_v1_1.csv
    python -m analysis.benchmark_v11 verify vrc3po_benchmark_v1_1.csv
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.build_master import (
    SCHEMA,
    SIM_ENVIRONMENTS,
    SIM_EYE_COLUMNS,
    SIM_HEAD_COLUMNS,
    SIM_RENAME,
    extract_environment,
    load_maze,
    load_terrain,
    parse_clock_to_seconds,
    parse_fms_time,
)
from analysis.retrain_camera_ready import FEATURE_COLUMNS

VERSION = "1.1"
PUPIL_FLOOR_MM = 1.0
GAZE_DIR = ["gaze_dir_world_X", "gaze_dir_world_Y", "gaze_dir_world_Z"]
GAZE_ORIGIN = ["gaze_origin_world_X", "gaze_origin_world_Y", "gaze_origin_world_Z"]
PUPILS = ["pupil_diam_L", "pupil_diam_R"]
OPENNESS = ["eye_open_L", "eye_open_R"]


# ---------------------------------------------------------------------------
# Simulations with native-rate masking
# ---------------------------------------------------------------------------
def mask_invalid_simulations_samples(eye: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    eye = eye.copy()
    stats = {"samples": int(len(eye))}
    for pupil, openness, side in (
        ("LeftPupilDiameter", "Left_Eye_Openness", "L"),
        ("RightPupilDiameter", "Right_Eye_Openness", "R"),
    ):
        bad = (eye[pupil] < 0) | (eye[openness] < 0)
        stats[f"invalid_{side}"] = int(bad.sum())
        eye.loc[bad, [pupil, openness]] = np.nan
    direction = ["GazeDirectionWrldSpc_X", "GazeDirectionWrldSpc_Y", "GazeDirectionWrldSpc_Z"]
    origin = ["GazeOriginWrldSpc_X", "GazeOriginWrldSpc_Y", "GazeOriginWrldSpc_Z"]
    bad = np.sqrt((eye[direction] ** 2).sum(axis=1)) < 0.5
    stats["invalid_gaze"] = int(bad.sum())
    eye.loc[bad, direction + origin] = np.nan
    return eye, stats


def load_simulations_recording(eye_path, head_path, fms_all, participant_code, environment):
    eye = pd.read_csv(eye_path)
    head = pd.read_csv(head_path)
    if len(eye) == 0 or len(head) == 0:
        return pd.DataFrame(), None
    eye, stats = mask_invalid_simulations_samples(eye)
    fms = fms_all[fms_all["Participants Code"] == participant_code].dropna(subset=["FMS"]).copy()
    if len(fms) == 0:
        return pd.DataFrame(), stats
    fms["parsed_dt"] = parse_fms_time(fms["Time"])
    fms = fms.dropna(subset=["parsed_dt"])
    if len(fms) == 0:
        return pd.DataFrame(), stats
    fms["abs_time_s"] = fms["parsed_dt"].apply(
        lambda d: d.hour * 3600 + d.minute * 60 + d.second + d.microsecond / 1e6)

    def build(pm_offset):
        e, h = eye.copy(), head.copy()
        e["abs_time_s"] = e["Time"].apply(lambda t: parse_clock_to_seconds(t, pm_offset))
        h["abs_time_s"] = h["Time"].apply(lambda t: parse_clock_to_seconds(t, pm_offset))
        origin = e["abs_time_s"].iloc[0]
        e["elapsed_s"] = (e["abs_time_s"] - origin).astype(int)
        h["elapsed_s"] = (h["abs_time_s"] - origin).astype(int)
        merged = (e.groupby("elapsed_s")[SIM_EYE_COLUMNS].mean().reset_index()
                  .merge(h.groupby("elapsed_s")[SIM_HEAD_COLUMNS].mean().reset_index(),
                         on="elapsed_s", how="inner"))
        merged["abs_time_s"] = merged["elapsed_s"] + origin
        return pd.merge_asof(merged.sort_values("abs_time_s"),
                             fms[["abs_time_s", "FMS"]].sort_values("abs_time_s"),
                             on="abs_time_s", direction="backward").dropna(subset=["FMS"])

    am, pm = build(0), build(12)
    merged = am if len(am) >= len(pm) else pm
    if len(merged) == 0:
        return pd.DataFrame(), stats
    merged = merged.rename(columns=SIM_RENAME)
    merged["participant_id"] = participant_code
    merged["condition"] = environment
    merged["source_dataset"] = "simulations"
    return merged[SCHEMA], stats


def load_simulations_v11(sim_base: Path, extract_base: Path):
    frames, mask_log = [], []
    for environment in SIM_ENVIRONMENTS:
        folder = extract_environment(environment, sim_base, extract_base)
        fms_all = pd.read_csv(os.path.join(sim_base, environment, "Anonymous_FMS_All.csv"),
                              encoding="utf-8-sig")
        fms_all.columns = [c.strip() for c in fms_all.columns]
        fms_all["Participants Code"] = fms_all["Participants Code"].ffill()
        loaded = discovered = 0
        for entry in sorted(os.listdir(folder)):
            full = os.path.join(folder, entry)
            match = re.match(r"^(\d+)_", entry)
            if not (os.path.isdir(full) and match):
                continue
            eyes = sorted(glob.glob(os.path.join(full, "eye_tracking_data*.csv")))
            heads = sorted(glob.glob(os.path.join(full, "head_tracking*.csv")))
            if not (eyes and heads):
                continue
            discovered += 1
            recording = f"simulations_{environment}_{entry}"
            frame, stats = load_simulations_recording(
                eyes[0], heads[0], fms_all, float(match.group(1)), environment)
            if stats is not None:
                mask_log.append({"recording_id": recording, **stats})
            if len(frame):
                frame["recording_id"] = recording
                frames.append(frame)
                loaded += 1
        print(f"  {environment}: {loaded} recordings loaded ({discovered} folders)", flush=True)
    return pd.concat(frames, ignore_index=True), pd.DataFrame(mask_log)


# ---------------------------------------------------------------------------
# Deduplication and cleaning
# ---------------------------------------------------------------------------
def _hash(values: np.ndarray) -> str:
    return hashlib.md5(np.nan_to_num(np.round(values, 4)).tobytes()).hexdigest()


def resolve_recordings(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = []
    for recording, group in frame.groupby("recording_id", sort=True):
        values = group.sort_values("elapsed_s")[FEATURE_COLUMNS].to_numpy()
        records.append({"recording_id": recording,
                        "participant": group["global_participant_id"].iloc[0],
                        "condition": group["condition"].iloc[0], "seconds": len(group),
                        "hash_full": _hash(values), "hash_60": _hash(values[:60])})
    table = pd.DataFrame(records)
    log = []
    for _, group in table[table.duplicated("hash_full", keep=False)].groupby("hash_full"):
        if group[["participant", "condition"]].drop_duplicates().shape[0] > 1:
            log += [(r, "identical content under different participant/condition labels; "
                        "all copies removed (label ambiguous)") for r in group["recording_id"]]
    dropped = {r for r, _ in log}
    for _, group in table[~table["recording_id"].isin(dropped)].groupby(["participant", "condition"]):
        if len(group) < 2:
            continue
        group = group.sort_values("seconds", ascending=False)
        kept = [group.iloc[0]]
        for _, row in group.iloc[1:].iterrows():
            if any(row["hash_full"] == k["hash_full"] or row["hash_60"] == k["hash_60"] for k in kept):
                log.append((row["recording_id"], f"duplicate of {kept[0]['recording_id']}; removed"))
            else:
                kept.append(row)
    decisions = pd.DataFrame(log, columns=["recording_id", "action"]).drop_duplicates("recording_id")
    decisions = decisions.merge(table[["recording_id", "participant", "condition", "seconds"]],
                                on="recording_id")
    return frame[~frame["recording_id"].isin(decisions["recording_id"])].copy(), decisions


def clean(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    report = {}
    empty = frame[FEATURE_COLUMNS].isna().any(axis=1)
    report["bins_without_valid_eye_sample_pct"] = (
        empty.groupby(frame["source_dataset"]).mean().mul(100).round(3).to_dict())
    for column in PUPILS:
        bad = frame[column] < PUPIL_FLOOR_MM
        report[f"{column}_below_floor"] = frame.loc[bad, "source_dataset"].value_counts().to_dict()
        frame.loc[bad, column] = np.nan
    for column in OPENNESS:
        bad = frame[column] < 0
        report[f"{column}_negative"] = frame.loc[bad, "source_dataset"].value_counts().to_dict()
        frame.loc[bad, column] = np.nan
    present = frame[GAZE_DIR].notna().all(axis=1)
    bad = present & (np.sqrt((frame[GAZE_DIR] ** 2).sum(axis=1)) < 0.5)
    report["gaze_near_zero"] = frame.loc[bad, "source_dataset"].value_counts().to_dict()
    frame.loc[bad, GAZE_DIR + GAZE_ORIGIN] = np.nan

    frame = frame.sort_values(["recording_id", "elapsed_s"]).reset_index(drop=True)
    frame[FEATURE_COLUMNS] = frame.groupby("recording_id")[FEATURE_COLUMNS].transform(
        lambda values: values.ffill().bfill())
    magnitude = np.sqrt((frame[GAZE_DIR] ** 2).sum(axis=1)).clip(lower=1e-6)
    frame[GAZE_DIR] = frame[GAZE_DIR].div(magnitude, axis=0)
    frame["elevated"] = frame["fms"] > 2
    return frame, report


def verify(frame: pd.DataFrame) -> list[str]:
    problems = []
    if frame[FEATURE_COLUMNS].isna().any().any():
        problems.append("missing feature values")
    if (frame[PUPILS] < PUPIL_FLOOR_MM).any().any():
        problems.append(f"pupil values below {PUPIL_FLOOR_MM} mm")
    if (frame[OPENNESS] < 0).any().any():
        problems.append("negative openness")
    if frame.duplicated(["recording_id", "elapsed_s"]).any():
        problems.append("duplicate timestamps within a recording")
    if not np.allclose(np.sqrt((frame[GAZE_DIR] ** 2).sum(axis=1)), 1.0, atol=1e-5):
        problems.append("gaze vectors not unit length")
    if frame.groupby("recording_id")[["global_participant_id", "condition"]].nunique().max().max() > 1:
        problems.append("a recording spans several participants or conditions")
    return problems


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    st = sub.add_parser("stage")
    st.add_argument("--raw-dir", type=Path, required=True)
    st.add_argument("--stage-dir", type=Path, required=True)
    st.add_argument("--source", required=True, choices=["terrain", "maze", "simulations"])
    fi = sub.add_parser("finalize")
    fi.add_argument("--stage-dir", type=Path, required=True)
    fi.add_argument("--output", type=Path, required=True)
    ve = sub.add_parser("verify")
    ve.add_argument("table", type=Path)
    args = parser.parse_args()

    if args.command == "stage":
        stage = args.stage_dir
        stage.mkdir(parents=True, exist_ok=True)
        if args.source == "terrain":
            frame = load_terrain(args.raw_dir / "Terrain 2022")
            frame["recording_id"] = ("terrain_" + frame["participant_id"].astype(str)
                                     + "_" + frame["condition"])
        elif args.source == "maze":
            frame = load_maze(args.raw_dir / "maze 2023", parts_dir=stage / "maze_parts")
            frame["recording_id"] = "maze_" + frame["participant_id"].astype(str)
        else:
            frame, mask_log = load_simulations_v11(args.raw_dir / "simulations 2021",
                                                   stage / "sim_extracted")
            mask_log.to_csv(stage / "v11_simulations_mask_log.csv", index=False)
        frame.to_csv(stage / f"v11_{args.source}.csv", index=False)
        print(f"{args.source}: {len(frame):,} rows, {frame['participant_id'].nunique()} "
              f"participants, {frame['recording_id'].nunique()} recordings")
        return

    if args.command == "finalize":
        parts = [pd.read_csv(args.stage_dir / f"v11_{s}.csv") for s in ("terrain", "maze", "simulations")]
        pooled = pd.concat([p[SCHEMA + ["recording_id"]] for p in parts], ignore_index=True)
        # Published identifier form "<source>_<code>.0" (see build_master).
        pooled["participant_id"] = pooled["participant_id"].astype(float)
        pooled["global_participant_id"] = (pooled["source_dataset"] + "_"
                                           + pooled["participant_id"].astype(str))
        kept, decisions = resolve_recordings(pooled)
        table, report = clean(kept)
        problems = verify(table)
        if problems:
            raise SystemExit(f"v1.1 verification failed: {problems}")
        table.to_csv(args.output, index=False)
        mask_log_path = args.stage_dir / "v11_simulations_mask_log.csv"
        masks = {}
        if mask_log_path.exists():
            m = pd.read_csv(mask_log_path)
            m = m[m["recording_id"].isin(table["recording_id"].unique())]
            pct = 100 * np.maximum(m["invalid_L"], m["invalid_R"]) / m["samples"]
            masks = {"samples": int(m["samples"].sum()),
                     "invalid_L_pct": float(100 * m["invalid_L"].sum() / m["samples"].sum()),
                     "invalid_R_pct": float(100 * m["invalid_R"].sum() / m["samples"].sum()),
                     "invalid_gaze_pct": float(100 * m["invalid_gaze"].sum() / m["samples"].sum()),
                     "per_recording_invalid_pct_median": float(pct.median()),
                     "per_recording_invalid_pct_iqr": [float(pct.quantile(.25)), float(pct.quantile(.75))],
                     "per_recording_invalid_pct_max": float(pct.max()),
                     "recordings_over_20pct": m.loc[pct > 20, "recording_id"].tolist()}
        manifest = {
            "version": VERSION, "rows": int(len(table)),
            "participants": int(table["global_participant_id"].nunique()),
            "recordings": int(table["recording_id"].nunique()),
            "rows_by_source": table["source_dataset"].value_counts().to_dict(),
            "participants_by_source": table.groupby("source_dataset")["global_participant_id"].nunique().to_dict(),
            "recordings_removed": decisions.to_dict(orient="records"),
            "cleaning": report, "simulations_native_rate_masking": masks,
            "pupil_floor_mm": PUPIL_FLOOR_MM,
            "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        }
        args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
        print(json.dumps({k: v for k, v in manifest.items() if k != "cleaning"}, indent=2, default=str))
        return

    table = pd.read_csv(args.table)
    problems = verify(table)
    print("OK" if not problems else problems)
    raise SystemExit(1 if problems else 0)


if __name__ == "__main__":
    main()
