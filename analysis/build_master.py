#!/usr/bin/env python3
"""Build the pooled 1 Hz table from the raw SAVELab recordings.

This is a faithful transcription of the preprocessing pipeline used for the
article. It converts the three source datasets from their native formats and
sampling rates into one table with a shared 20-column schema, one row per
participant-second.

The output of this module is the *unfixed* pooled table. Two further
corrections -- gaze renormalization and pupil sentinel masking -- are applied by
``analysis.harmonize``, which produces the table the analyses actually consume.
The full chain is:

    raw recordings -> build_master -> harmonize -> vrc3po_master_dataset_fixed.csv

PER-SOURCE HANDLING
-------------------
**Terrain** is already at 1 Hz. Eye and head files are read per
participant-condition, truncated to their common length, and given a synthetic
``elapsed_s`` running index. FMS is carried in the eye file.

**Maze** is recorded at a nominal 60 Hz (54--77 Hz effective across participants) with Vive-style column names. Timestamps come
from ``S0100_RecTime`` in 100 ns ticks. Samples whose validity flag is zero are
set to missing *before* binning, which is why Maze carries none of the averaged
sentinel residue documented in ``analysis.harmonize``. Samples are averaged into
one-second bins, filled within the session, and joined to the per-minute FMS
report parsed from the accompanying text file.

**Simulations** stores wall-clock times as ``H-M-S-ms`` strings without an
AM/PM marker. Both interpretations are built and the one yielding more
FMS-matched rows is kept. Eye and head streams are averaged into one-second
bins independently, inner-joined, and matched to FMS by backward nearest
timestamp.

FAITHFULNESS
------------
Quirks of the original pipeline are preserved deliberately, including the
`float` participant code for Simulations, which upcasts the whole
``participant_id`` column and produces global identifiers of the form
``maze_1.0``. Do not "fix" these: they are part of the published identifiers.

KNOWN DISCREPANCY
-----------------
Running this module and then ``analysis.harmonize`` reproduces the published
table exactly in shape, row keys, participant set, per-source counts, ``fms``
and ``elevated``. 148 of 100,367 rows (0.15%) differ in feature values.

The cause is in the source release rather than in this code. Five
participant-environment pairs in Simulations have two recording folders each:
beach/25, roller/5, sea/6, sea/9 and sea/25. That is five pairs across four
distinct participants (5, 6, 9 and 25), because participant 25 is duplicated in
two different environments. Both are discovered, both are labelled with the
same participant and condition, and they are concatenated into one session key.
That produces 278 rows with colliding one-second indices, and 113 of the 6,442
windows (1.75%) are built from a session that mixes two recordings. Which
recording lands first depended on filesystem enumeration order in the original
run, which is why those rows are not bit-reproducible.

None of the affected participants are in the single-task held-out test set, so the
single-task detection results are unaffected. The article documents the issue in its
Methods and Limitations sections.

USAGE
-----
Per-source staging, so that a single source can be processed at a time:

    python -m analysis.build_master --raw-dir RAW --stage-dir STAGE --source terrain
    python -m analysis.build_master --raw-dir RAW --stage-dir STAGE --source maze
    python -m analysis.build_master --raw-dir RAW --stage-dir STAGE --source simulations
    python -m analysis.build_master --stage-dir STAGE --merge --output pooled.csv
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

SCHEMA = [
    "elapsed_s", "fms",
    "pupil_diam_L", "pupil_diam_R", "eye_open_L", "eye_open_R",
    "gaze_dir_world_X", "gaze_dir_world_Y", "gaze_dir_world_Z",
    "gaze_origin_world_X", "gaze_origin_world_Y", "gaze_origin_world_Z",
    "head_quat_X", "head_quat_Y", "head_quat_Z", "head_quat_W",
    "head_velocity", "participant_id", "condition", "source_dataset",
]
TERRAIN_CONDITIONS = ["flat", "noise", "speedbumps"]
SIM_ENVIRONMENTS = ["beach", "roller", "room", "sea", "walk"]


# ---------------------------------------------------------------------------
# Terrain: already 1 Hz
# ---------------------------------------------------------------------------
def load_terrain_session(eye_path, head_path, participant_id, condition):
    eye = pd.read_csv(eye_path)
    head = pd.read_csv(head_path)
    n = min(len(eye), len(head))
    eye, head = eye.iloc[:n], head.iloc[:n]
    return pd.DataFrame({
        "elapsed_s": np.arange(n),
        "fms": eye["fms"].values,
        "pupil_diam_L": eye["LeftPupilDiameter"].values,
        "pupil_diam_R": eye["RightPupilDiameter"].values,
        "eye_open_L": eye["Left_Eye_Openness"].values,
        "eye_open_R": eye["Right_Eye_Openness"].values,
        "gaze_dir_world_X": eye["GazeDirectionWrldSpc_X"].values,
        "gaze_dir_world_Y": eye["GazeDirectionWrldSpc_Y"].values,
        "gaze_dir_world_Z": eye["GazeDirectionWrldSpc_Z"].values,
        "gaze_origin_world_X": eye["GazeOriginWrldSpc_X"].values,
        "gaze_origin_world_Y": eye["GazeOriginWrldSpc_Y"].values,
        "gaze_origin_world_Z": eye["GazeOriginWrldSpc_Z"].values,
        "head_quat_X": head["HeadQRotationX"].values,
        "head_quat_Y": head["HeadQRotationY"].values,
        "head_quat_Z": head["HeadQRotationZ"].values,
        "head_quat_W": head["HeadQRotationW"].values,
        "head_velocity": head["Velocity"].values,
        "participant_id": participant_id,
        "condition": condition,
        "source_dataset": "terrain",
    })


def load_terrain(terrain_dir: Path) -> pd.DataFrame:
    frames = []
    folders = [
        f for f in glob.glob(os.path.join(terrain_dir, "*"))
        if os.path.isdir(f) and re.match(r"^\d+$", os.path.basename(f))
    ]
    for folder in sorted(folders):
        participant_id = int(os.path.basename(folder))
        for condition in TERRAIN_CONDITIONS:
            eye_path = os.path.join(folder, condition, "eye.csv")
            head_path = os.path.join(folder, condition, "head.csv")
            if os.path.exists(eye_path) and os.path.exists(head_path):
                try:
                    frames.append(
                        load_terrain_session(eye_path, head_path, participant_id, condition)
                    )
                except Exception as error:  # noqa: BLE001 - mirror original leniency
                    print(f"  Terrain P{participant_id}/{condition}: FAILED - {error}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---------------------------------------------------------------------------
# Maze: 60 Hz nominal (54--77 Hz effective), Vive column names, per-minute FMS
# ---------------------------------------------------------------------------
def parse_maze_fms(path) -> pd.DataFrame:
    with open(path, "r") as handle:
        lines = handle.readlines()
    rows = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        tokens = line.split()
        if len(tokens) >= 5 and all(re.match(r"^-?\d+$", t) for t in tokens):
            rows.append({"minute": len(rows), "fms": int(tokens[0])})
    return pd.DataFrame(rows)


MAZE_COLUMNS = {
    "pupil_diam_L": "S0100_Left_Diameter",
    "pupil_diam_R": "S0100_Right_Diameter",
    "eye_open_L": "S0100_Left_Openness",
    "eye_open_R": "S0100_Right_Openness",
    "gaze_dir_world_X": "V0300_Combine_GazeDir",
    "gaze_dir_world_Y": "V0301_Combine_GazeDir",
    "gaze_dir_world_Z": "V0302_Combine_GazeDir",
    "gaze_origin_world_X": "V0300_Combine_Origin",
    "gaze_origin_world_Y": "V0301_Combine_Origin",
    "gaze_origin_world_Z": "V0302_Combine_Origin",
    "head_quat_X": "V0400_HMDRot",
    "head_quat_Y": "V0401_HMDRot",
    "head_quat_Z": "V0402_HMDRot",
    "head_quat_W": "V0403_HMDRot",
}


def load_maze_session(eye_path, fms_path, participant_id) -> pd.DataFrame:
    eye = pd.read_csv(eye_path)
    fms_frame = parse_maze_fms(fms_path)

    origin = eye["S0100_RecTime"].iloc[0]
    eye["elapsed_s"] = ((eye["S0100_RecTime"] - origin) / 10000 / 1000).astype(int)

    # Validity flags are honoured before binning. This is the step Simulations
    # and Terrain lack, and it is why Maze has no averaged sentinel residue.
    eye.loc[eye["S0100_Left_Validity"] == 0,
            ["S0100_Left_Diameter", "S0100_Left_Openness"]] = np.nan
    eye.loc[eye["S0100_Right_Validity"] == 0,
            ["S0100_Right_Diameter", "S0100_Right_Openness"]] = np.nan
    eye.loc[eye["S0100_Combine_Validity"] == 0,
            ["V0300_Combine_GazeDir", "V0301_Combine_GazeDir", "V0302_Combine_GazeDir",
             "V0300_Combine_Origin", "V0301_Combine_Origin", "V0302_Combine_Origin"]] = np.nan

    targets = list(MAZE_COLUMNS.values())
    agg = eye.groupby("elapsed_s")[targets].mean().reset_index()
    agg.columns = ["elapsed_s"] + list(MAZE_COLUMNS.keys())
    agg[list(MAZE_COLUMNS.keys())] = agg[list(MAZE_COLUMNS.keys())].ffill().bfill()

    agg["minute"] = (agg["elapsed_s"] // 60).clip(upper=len(fms_frame) - 1)
    agg = agg.merge(fms_frame[["minute", "fms"]], on="minute", how="left")
    agg["head_velocity"] = np.nan
    agg["participant_id"] = participant_id
    agg["condition"] = "maze"
    agg["source_dataset"] = "maze"
    return agg.drop(columns=["minute"])


def discover_maze_pairs(maze_dir):
    eye_map, fms_map = {}, {}
    for path in glob.glob(os.path.join(maze_dir, "*")):
        name = os.path.basename(path)
        eye_match = re.match(r"^(\d+)[-_][Ee]ye\.csv$", name)
        fms_match = re.match(r"^(\d+)_fms\.txt$", name)
        if eye_match:
            eye_map[int(eye_match.group(1))] = path
        elif fms_match:
            fms_map[int(fms_match.group(1))] = path
    return [
        (pid, eye_map[pid], fms_map[pid])
        for pid in sorted(set(eye_map) & set(fms_map))
    ]


def load_maze(maze_dir: Path, parts_dir: Path | None = None) -> pd.DataFrame:
    """Load every Maze session.

    The raw Maze recordings are ~70 MB per participant. When ``parts_dir`` is
    given, each participant is cached there and skipped on a later run, so the
    source can be processed across several invocations.
    """
    pairs = discover_maze_pairs(maze_dir)
    frames = []
    for participant_id, eye_path, fms_path in pairs:
        cached = parts_dir / f"maze_{participant_id:03d}.csv" if parts_dir else None
        if cached is not None and cached.exists():
            frames.append(pd.read_csv(cached))
            continue
        try:
            frame = load_maze_session(eye_path, fms_path, participant_id)
        except Exception as error:  # noqa: BLE001
            print(f"  Maze P{participant_id}: FAILED - {error}")
            continue
        if cached is not None:
            cached.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(cached, index=False)
            print(f"  Maze P{participant_id}: {len(frame):,} rows cached", flush=True)
        frames.append(frame)

    if parts_dir is not None:
        done = len(list(parts_dir.glob("maze_*.csv")))
        if done < len(pairs):
            raise SystemExit(
                f"staged {done} of {len(pairs)} Maze participants; rerun the same "
                "command to continue"
            )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---------------------------------------------------------------------------
# Simulations: wall-clock strings without AM/PM, FMS matched by timestamp
# ---------------------------------------------------------------------------
def parse_clock_to_seconds(value, pm_offset=0):
    hours, minutes, seconds, milliseconds = value.split("-")
    return (
        (int(hours) + pm_offset) * 3600
        + int(minutes) * 60
        + int(seconds)
        + int(milliseconds) / 1000
    )


def parse_fms_time(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, format="%Y.%m.%d %H:%M:%S:%f", errors="coerce")
    failed = parsed.isna()
    if failed.any():
        cleaned = (
            series[failed].astype(str).str.strip().str.replace(r"[\[\]]", "", regex=True)
        )
        parsed.loc[failed] = pd.to_datetime(
            cleaned, format="%Y.%m.%d %H:%M:%S:%f", errors="coerce"
        )
    return parsed


SIM_EYE_COLUMNS = [
    "LeftPupilDiameter", "RightPupilDiameter", "Left_Eye_Openness", "Right_Eye_Openness",
    "GazeDirectionWrldSpc_X", "GazeDirectionWrldSpc_Y", "GazeDirectionWrldSpc_Z",
    "GazeOriginWrldSpc_X", "GazeOriginWrldSpc_Y", "GazeOriginWrldSpc_Z",
]
SIM_HEAD_COLUMNS = [
    "HeadQRotationX", "HeadQRotationY", "HeadQRotationZ", "HeadQRotationW", "Velocity",
]
SIM_RENAME = {
    "LeftPupilDiameter": "pupil_diam_L", "RightPupilDiameter": "pupil_diam_R",
    "Left_Eye_Openness": "eye_open_L", "Right_Eye_Openness": "eye_open_R",
    "GazeDirectionWrldSpc_X": "gaze_dir_world_X",
    "GazeDirectionWrldSpc_Y": "gaze_dir_world_Y",
    "GazeDirectionWrldSpc_Z": "gaze_dir_world_Z",
    "GazeOriginWrldSpc_X": "gaze_origin_world_X",
    "GazeOriginWrldSpc_Y": "gaze_origin_world_Y",
    "GazeOriginWrldSpc_Z": "gaze_origin_world_Z",
    "HeadQRotationX": "head_quat_X", "HeadQRotationY": "head_quat_Y",
    "HeadQRotationZ": "head_quat_Z", "HeadQRotationW": "head_quat_W",
    "Velocity": "head_velocity", "FMS": "fms",
}


def load_simulations_session(eye_path, head_path, fms_all, participant_code, environment):
    eye = pd.read_csv(eye_path)
    head = pd.read_csv(head_path)
    if len(eye) == 0 or len(head) == 0:
        return pd.DataFrame()

    participant_fms = (
        fms_all[fms_all["Participants Code"] == participant_code]
        .dropna(subset=["FMS"])
        .copy()
    )
    if len(participant_fms) == 0:
        return pd.DataFrame()
    participant_fms["parsed_dt"] = parse_fms_time(participant_fms["Time"])
    participant_fms = participant_fms.dropna(subset=["parsed_dt"])
    if len(participant_fms) == 0:
        return pd.DataFrame()
    participant_fms["abs_time_s"] = participant_fms["parsed_dt"].apply(
        lambda d: d.hour * 3600 + d.minute * 60 + d.second + d.microsecond / 1e6
    )

    def build_merged(pm_offset):
        eye_c, head_c = eye.copy(), head.copy()
        eye_c["abs_time_s"] = eye_c["Time"].apply(
            lambda t: parse_clock_to_seconds(t, pm_offset))
        head_c["abs_time_s"] = head_c["Time"].apply(
            lambda t: parse_clock_to_seconds(t, pm_offset))
        origin = eye_c["abs_time_s"].iloc[0]
        eye_c["elapsed_s"] = (eye_c["abs_time_s"] - origin).astype(int)
        head_c["elapsed_s"] = (head_c["abs_time_s"] - origin).astype(int)
        eye_agg = eye_c.groupby("elapsed_s")[SIM_EYE_COLUMNS].mean().reset_index()
        head_agg = head_c.groupby("elapsed_s")[SIM_HEAD_COLUMNS].mean().reset_index()
        merged = eye_agg.merge(head_agg, on="elapsed_s", how="inner")
        merged["abs_time_s"] = merged["elapsed_s"] + origin
        merged = merged.sort_values("abs_time_s")
        return pd.merge_asof(
            merged,
            participant_fms[["abs_time_s", "FMS"]].sort_values("abs_time_s"),
            on="abs_time_s",
            direction="backward",
        ).dropna(subset=["FMS"])

    # The recorded clock lacks an AM/PM marker; keep whichever reading matches
    # more FMS reports.
    candidate_am, candidate_pm = build_merged(0), build_merged(12)
    merged = candidate_am if len(candidate_am) >= len(candidate_pm) else candidate_pm
    if len(merged) == 0:
        return pd.DataFrame()

    merged = merged.rename(columns=SIM_RENAME)
    merged["participant_id"] = participant_code
    merged["condition"] = environment
    merged["source_dataset"] = "simulations"
    return merged[SCHEMA]


def extract_environment(environment, sim_base, extract_base):
    destination = os.path.join(extract_base, environment)
    if os.path.exists(destination) and len(os.listdir(destination)) > 0:
        return destination
    folder = os.path.join(sim_base, environment)
    archives = glob.glob(os.path.join(folder, "*.zip"))
    if not archives:
        return folder
    os.makedirs(destination, exist_ok=True)
    with zipfile.ZipFile(archives[0]) as archive:
        archive.extractall(destination)
    return destination


def discover_sim_sessions(extracted_dir):
    """Find (participant, eye file, head file) triples for one environment.

    Some participants have two recording folders for the same environment --
    ``25_...`` and ``25_2_...`` -- presumably a restart after an interruption.
    Both match the participant pattern, so both load and both are labelled with
    the same participant and condition. Downstream they are concatenated into a
    single session key, which means their one-second indices collide and a
    window can span two different recordings.

    This is a property of the source release, not of the code, and it is left
    in place because it is what produced the published table. It is reported
    loudly here so that nobody has to rediscover it, and the enumeration is
    sorted so the ordering is at least stable across machines.
    """
    sessions = []
    for entry in sorted(os.listdir(extracted_dir)):
        full = os.path.join(extracted_dir, entry)
        if not os.path.isdir(full):
            continue
        match = re.match(r"^(\d+)_", entry)
        if not match:
            continue
        eye_files = sorted(glob.glob(os.path.join(full, "eye_tracking_data*.csv")))
        head_files = sorted(glob.glob(os.path.join(full, "head_tracking*.csv")))
        if eye_files and head_files:
            sessions.append((int(match.group(1)), eye_files[0], head_files[0]))

    counts = {}
    for participant_id, _, _ in sessions:
        counts[participant_id] = counts.get(participant_id, 0) + 1
    duplicated = sorted(p for p, n in counts.items() if n > 1)
    if duplicated:
        print(
            f"  WARNING: participants {duplicated} have more than one recording "
            f"in {os.path.basename(extracted_dir)}. They will be merged into a "
            "single session key, so their one-second indices collide. See the "
            "module docstring.",
            flush=True,
        )
    return sessions


def load_simulations(sim_base: Path, extract_base: Path) -> pd.DataFrame:
    frames = []
    for environment in SIM_ENVIRONMENTS:
        extracted = extract_environment(environment, sim_base, extract_base)
        sessions = discover_sim_sessions(extracted)
        fms_all = pd.read_csv(
            os.path.join(sim_base, environment, "Anonymous_FMS_All.csv"),
            encoding="utf-8-sig",
        )
        fms_all.columns = [c.strip() for c in fms_all.columns]
        fms_all["Participants Code"] = fms_all["Participants Code"].ffill()
        loaded = 0
        for participant_id, eye_path, head_path in sessions:
            try:
                frame = load_simulations_session(
                    eye_path, head_path, fms_all, float(participant_id), environment
                )
                if len(frame) > 0:
                    frames.append(frame)
                    loaded += 1
            except Exception as error:  # noqa: BLE001
                print(f"  Sim {environment} P{participant_id}: FAILED - {error}")
        print(f"  {environment}: {loaded} sessions loaded ({len(sessions)} discovered)")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------
def merge_sources(terrain, maze, simulations) -> pd.DataFrame:
    """Concatenate in the original order: terrain, maze, simulations."""
    pooled = pd.concat(
        [terrain[SCHEMA], maze[SCHEMA], simulations[SCHEMA]], ignore_index=True
    )
    # Simulations carries a float participant code, which upcasts this column
    # and yields identifiers such as "maze_1.0". Preserved deliberately.
    pooled["global_participant_id"] = (
        pooled["source_dataset"] + "_" + pooled["participant_id"].astype(str)
    )
    pooled["elevated"] = pooled["fms"] > 2
    return pooled


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path,
                       help="utsa_eye_head_cybersickness_datasets")
    parser.add_argument("--stage-dir", type=Path, required=True,
                       help="where per-source intermediates are written")
    parser.add_argument("--source", choices=["terrain", "maze", "simulations"])
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    args.stage_dir.mkdir(parents=True, exist_ok=True)

    if args.source:
        if not args.raw_dir:
            raise SystemExit("--raw-dir is required when staging a source")
        if args.source == "terrain":
            frame = load_terrain(args.raw_dir / "Terrain 2022")
        elif args.source == "maze":
            frame = load_maze(args.raw_dir / "maze 2023",
                              parts_dir=args.stage_dir / "maze_parts")
        else:
            frame = load_simulations(
                args.raw_dir / "simulations 2021", args.stage_dir / "sim_extracted"
            )
        destination = args.stage_dir / f"{args.source}.csv"
        frame.to_csv(destination, index=False)
        print(f"{args.source}: {len(frame):,} rows, "
              f"{frame['participant_id'].nunique()} participants -> {destination}")
        return

    if args.merge:
        if not args.output:
            raise SystemExit("--output is required with --merge")
        frames = {
            name: pd.read_csv(args.stage_dir / f"{name}.csv")
            for name in ("terrain", "maze", "simulations")
        }
        pooled = merge_sources(frames["terrain"], frames["maze"], frames["simulations"])
        pooled.to_csv(args.output, index=False)
        print(f"pooled: {len(pooled):,} rows, "
              f"{pooled['global_participant_id'].nunique()} participants -> {args.output}")
        return

    raise SystemExit("provide --source to stage one dataset, or --merge")


if __name__ == "__main__":
    main()
