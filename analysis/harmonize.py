#!/usr/bin/env python3
"""Post-hoc harmonization fixes applied to the pooled 1 Hz table.

This module is a faithful transcription of the correction step that produced
`vrc3po_master_dataset_fixed.csv` from the pooled per-source 1 Hz table. It
applies two fixes:

1. **Gaze renormalization.** Gaze direction vectors are divided by their own
   magnitude so that every row carries a unit vector. This is the step that
   corrects the Terrain vectors, whose raw magnitudes average 1.906 (SD 0.190).
   Maze and Simulations vectors already have mean magnitude 1.000.

2. **Pupil sentinel masking and within-session fill.** Pupil-diameter values
   equal to -1 are set to missing, then forward- and backward-filled within
   each (participant, condition) session so that a fill never crosses a session
   boundary.

WHAT THIS DOES NOT FIX, AND WHY IT MATTERS
------------------------------------------
Fix 2 tests for exact equality with -1, and it runs *after* the raw samples
have been averaged into one-second bins. A bin that averaged valid samples
together with -1 sentinels therefore yields an intermediate negative value --
around -0.92 in the observed data -- which `== -1` does not match. Those rows
survive into the published table: 438 rows (0.44%), all between -1 and 0, all
in Simulations and Terrain, across 32 participants. Maze is unaffected because
its sentinels were masked before binning during per-source processing.

This ordering is a defect, not a design choice. It is preserved here because it
is what produced the published results, and correcting it would invalidate
every frozen artifact. Pass ``--strict-sentinel`` to apply the correct rule
(mask all non-positive pupil values before filling); the output will **not**
reproduce the published numbers and is offered only for readers who want a
clean table for new work.

The per-source raw-to-1 Hz conversion that precedes this step -- native-rate
binning, FMS alignment, and the Simulations 12-hour timestamp disambiguation --
is described in the Methods section of the article and is not implemented here.

USAGE
-----
    python -m analysis.harmonize --input pooled.csv --output fixed.csv
    python -m analysis.harmonize --verify fixed.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

GAZE_COLUMNS = ["gaze_dir_world_X", "gaze_dir_world_Y", "gaze_dir_world_Z"]
PUPIL_COLUMNS = ["pupil_diam_L", "pupil_diam_R"]
SESSION_KEYS = ["global_participant_id", "condition"]
MAGNITUDE_FLOOR = 1e-6


def renormalize_gaze(frame: pd.DataFrame) -> pd.DataFrame:
    """Scale gaze direction rows to unit length."""
    magnitude = np.sqrt(sum(frame[c] ** 2 for c in GAZE_COLUMNS))
    magnitude = magnitude.clip(lower=MAGNITUDE_FLOOR)
    for column in GAZE_COLUMNS:
        frame[column] = frame[column] / magnitude
    return frame


def mask_and_fill_pupils(frame: pd.DataFrame, strict: bool = False) -> pd.DataFrame:
    """Mask sentinel pupil readings, then fill within each session.

    With ``strict=False`` this reproduces the published behaviour: only values
    exactly equal to -1 are masked. With ``strict=True`` every non-positive
    value is masked, which also removes the averaged sentinel residue.
    """
    for column in PUPIL_COLUMNS:
        if strict:
            frame.loc[frame[column] <= 0, column] = np.nan
        else:
            frame.loc[frame[column] == -1, column] = np.nan

    frame = frame.sort_values(SESSION_KEYS + ["elapsed_s"])
    frame[PUPIL_COLUMNS] = frame.groupby(SESSION_KEYS)[PUPIL_COLUMNS].transform(
        lambda values: values.ffill().bfill()
    )
    return frame


def harmonize(frame: pd.DataFrame, strict_sentinel: bool = False) -> pd.DataFrame:
    frame = frame.copy()
    frame = renormalize_gaze(frame)
    frame = mask_and_fill_pupils(frame, strict=strict_sentinel)
    return frame


def verify(frame: pd.DataFrame) -> dict:
    """Check the invariants the published table is expected to satisfy."""
    magnitude = np.sqrt(sum(frame[c] ** 2 for c in GAZE_COLUMNS))
    negative = sum(int((frame[c] < 0).sum()) for c in PUPIL_COLUMNS)
    report = {
        "rows": int(len(frame)),
        "participants": int(frame["global_participant_id"].nunique()),
        "gaze_magnitude_min": float(magnitude.min()),
        "gaze_magnitude_max": float(magnitude.max()),
        "pupil_exactly_minus_one": sum(
            int((frame[c] == -1).sum()) for c in PUPIL_COLUMNS
        ),
        "pupil_missing": int(frame[PUPIL_COLUMNS].isna().sum().sum()),
        "pupil_negative_rows": negative,
    }

    problems = []
    if not np.allclose(magnitude, 1.0, atol=1e-6):
        problems.append("gaze vectors are not unit length")
    if report["pupil_exactly_minus_one"]:
        problems.append("pupil values exactly equal to -1 remain")
    if report["pupil_missing"]:
        problems.append("pupil values are still missing after the fill")
    report["problems"] = problems

    # Idempotency: re-applying the published fixes must change nothing.
    again = harmonize(frame)
    numeric = GAZE_COLUMNS + PUPIL_COLUMNS
    report["idempotent"] = bool(
        np.allclose(
            frame.sort_values(SESSION_KEYS + ["elapsed_s"])[numeric].to_numpy(),
            again.sort_values(SESSION_KEYS + ["elapsed_s"])[numeric].to_numpy(),
            atol=1e-9,
        )
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="pooled per-source 1 Hz table")
    parser.add_argument("--output", type=Path, help="destination for the fixed table")
    parser.add_argument("--verify", type=Path, help="check an existing fixed table")
    parser.add_argument(
        "--strict-sentinel",
        action="store_true",
        help="mask all non-positive pupil values; does NOT reproduce published results",
    )
    args = parser.parse_args()

    if args.verify:
        import json

        report = verify(pd.read_csv(args.verify))
        print(json.dumps(report, indent=2))
        raise SystemExit(1 if report["problems"] or not report["idempotent"] else 0)

    if not args.input or not args.output:
        raise SystemExit("provide --input and --output, or --verify")

    frame = pd.read_csv(args.input)
    result = harmonize(frame, strict_sentinel=args.strict_sentinel)
    result.to_csv(args.output, index=False)
    print(f"wrote {args.output} ({len(result):,} rows)")
    if args.strict_sentinel:
        print("NOTE: --strict-sentinel was used; this table will not reproduce "
              "the published results.")


if __name__ == "__main__":
    main()
