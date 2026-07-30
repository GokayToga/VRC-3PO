#!/usr/bin/env python3
"""Leakage-resistant split protocols for the VRC-3PO reanalysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.corrected_evaluation import build_window_metadata


def chronological_purged_split(
    metadata: pd.DataFrame,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
    purge_samples: int = 15,
) -> pd.Series:
    """Assign windows to chronological blocks within every session.

    Windows crossing a block boundary or falling in a purge interval are
    excluded. This prevents a raw observation from appearing in multiple
    splits even when the source windows overlap.
    """
    if train_fraction <= 0 or validation_fraction <= 0:
        raise ValueError("Train and validation fractions must be positive.")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Train and validation fractions must sum to less than one.")
    if purge_samples < 0:
        raise ValueError("purge_samples must be non-negative.")

    assignment = pd.Series("excluded", index=metadata.index, dtype="object")
    grouped = metadata.groupby(["participant_id", "condition"], sort=False)
    for _, session in grouped:
        session_end = int(session["end_index"].max())
        train_boundary = int(np.floor(session_end * train_fraction))
        validation_boundary = int(
            np.floor(session_end * (train_fraction + validation_fraction))
        )

        train = session["end_index"] <= train_boundary
        validation = (
            (session["start_index"] >= train_boundary + purge_samples)
            & (session["end_index"] <= validation_boundary)
        )
        test = session["start_index"] >= validation_boundary + purge_samples

        assignment.loc[session.index[train]] = "train"
        assignment.loc[session.index[validation]] = "validation"
        assignment.loc[session.index[test]] = "test"
    return assignment


def assert_no_interval_overlap(metadata: pd.DataFrame, assignment: pd.Series) -> None:
    """Raise if raw sample intervals overlap across retained split labels."""
    retained = metadata.assign(split=assignment)
    retained = retained[retained["split"].isin(["train", "validation", "test"])]
    for session_key, session in retained.groupby(
        ["participant_id", "condition"], sort=False
    ):
        intervals: dict[str, list[tuple[int, int]]] = {}
        for split, split_rows in session.groupby("split"):
            intervals[split] = list(
                zip(split_rows["start_index"], split_rows["end_index"])
            )
        split_names = list(intervals)
        for i, first in enumerate(split_names):
            for second in split_names[i + 1 :]:
                for first_start, first_end in intervals[first]:
                    for second_start, second_end in intervals[second]:
                        overlap = max(first_start, second_start) < min(
                            first_end, second_end
                        )
                        if overlap:
                            raise AssertionError(
                                f"Raw interval overlap in session {session_key}: "
                                f"{first} [{first_start}, {first_end}) and "
                                f"{second} [{second_start}, {second_end})"
                            )


def summarize(metadata: pd.DataFrame, assignment: pd.Series) -> dict[str, object]:
    table = metadata.assign(split=assignment)
    retained = table[table["split"] != "excluded"]
    by_split = {}
    for split, group in retained.groupby("split"):
        by_split[split] = {
            "windows": len(group),
            "participants": int(group["participant_id"].nunique()),
            "sessions": int(
                group[["participant_id", "condition"]].drop_duplicates().shape[0]
            ),
            "by_source_windows": {
                str(key): int(value)
                for key, value in group["source_dataset"].value_counts().items()
            },
        }
    return {
        "all_windows": len(metadata),
        "retained_windows": len(retained),
        "excluded_boundary_or_purge_windows": int((assignment == "excluded").sum()),
        "by_split": by_split,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/corrected"))
    parser.add_argument("--train-fraction", type=float, default=0.60)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--purge-samples", type=int, default=15)
    args = parser.parse_args()

    df = pd.read_csv(args.dataset)
    _, metadata = build_window_metadata(df)
    assignment = chronological_purged_split(
        metadata,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        purge_samples=args.purge_samples,
    )
    assert_no_interval_overlap(metadata, assignment)
    result = summarize(metadata, assignment)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata.assign(split=assignment).to_csv(
        args.output_dir / "blocked_window_split_manifest.csv", index=False
    )
    with (args.output_dir / "blocked_window_split_summary.json").open("w") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
