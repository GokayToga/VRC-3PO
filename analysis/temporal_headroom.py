#!/usr/bin/env python3
"""How much cybersickness signal lives at a scale the 30-second window cannot see?

Every model in the article operates inside one 30-second window and carries no
state from one window to the next, so nothing it computes can depend on *where*
in the exposure the window sits. This script measures what that costs, using
only frozen artifacts and no retraining.

Three questions:

1. How well does elapsed time alone rank elevated windows, for held-out
   participants? This is the crudest possible session-scale feature -- a clock,
   with no eye or head signal at all.
2. Is the clock complementary to the frozen ensemble, or redundant with it?
   A rank blend of the two gives an upper bound on what a session-aware model
   could recover, without training one.
3. Does the picture differ within a participant versus pooled across
   participants?

IMPORTANT -- these are descriptive, not out-of-sample. The blend weight is
chosen on the same nine test participants it is evaluated on, exactly like the
operating-point sweep in the article. Treat the blended AUC as an estimate of
headroom, not as a result that could be reported as a model.

The window rebuild here is independent of the training pipeline, so the script
asserts that it reproduces the frozen ordering (pooled AUC 0.756948) before
reporting anything that depends on alignment. If that assertion fails, the
window index does not match the saved predictions and the output is suppressed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

FROZEN_POOLED_AUC = 0.756947866036354
WINDOW = 30
STRIDE = 15
SINGLE_TASK = ("simulations", "terrain")


def rank_auc(y: np.ndarray, score: np.ndarray) -> float:
    """Rank-based ROC AUC; ties handled by average rank."""
    y = np.asarray(y)
    ranks = pd.Series(np.asarray(score)).rank().to_numpy()
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def build_windows(dataset: pd.DataFrame) -> pd.DataFrame:
    """Rebuild the window index, recording where each window sits in its session.

    Mirrors the article's windowing: sort by elapsed time within each
    participant-condition session, 30-second windows at a 15-second stride,
    target is the window-mean FMS.
    """
    records = []
    group_keys = ["source_dataset", "participant_id", "condition"]
    for (source, _pid, condition), session in dataset.groupby(group_keys):
        session = session.sort_values("elapsed_s")
        fms = session["fms"].to_numpy()
        elapsed = session["elapsed_s"].to_numpy()
        gid = session["global_participant_id"].iloc[0]
        duration = elapsed[-1] - elapsed[0] if len(elapsed) > 1 else 1
        for start in range(0, len(session) - WINDOW + 1, STRIDE):
            records.append(
                {
                    "source": source,
                    "gid": gid,
                    "condition": condition,
                    "t_start": elapsed[start],
                    "t_frac": (elapsed[start] - elapsed[0]) / max(duration, 1),
                    "y": fms[start : start + WINDOW].mean(),
                }
            )
    windows = pd.DataFrame.from_records(records)
    windows["elevated"] = (windows["y"] > 2).astype(int)
    return windows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    dataset = pd.read_csv(args.dataset)
    split = pd.read_csv(args.split_manifest)
    windows = build_windows(dataset)
    windows["split"] = windows["gid"].map(
        split.set_index("participant_id")["split"].to_dict()
    )

    test = windows[windows["split"] == "test"]
    single = (
        test[test["source"].isin(SINGLE_TASK)]
        .sort_values(["gid", "condition", "t_start"])
        .reset_index(drop=True)
    )

    predictions = np.load(args.predictions)
    if len(predictions) != len(single):
        raise SystemExit(
            f"prediction count {len(predictions)} does not match the rebuilt "
            f"single-task test set ({len(single)} windows)"
        )

    model_auc = rank_auc(single["elevated"], predictions)
    aligned = abs(model_auc - FROZEN_POOLED_AUC) < 1e-3
    single["model"] = predictions

    report = {
        "alignment": {
            "rebuilt_windows": int(len(windows)),
            "single_task_test_windows": int(len(single)),
            "model_auc_on_rebuilt_index": model_auc,
            "frozen_pooled_auc": FROZEN_POOLED_AUC,
            "aligned": bool(aligned),
        },
        "clock_alone": {
            "all_sources_test": rank_auc(test["elevated"], test["t_start"]),
            "single_task_test": rank_auc(single["elevated"], single["t_start"]),
        },
    }

    print("=" * 68)
    print("Alignment")
    print("=" * 68)
    print(f"  rebuilt windows            {len(windows)}")
    print(f"  single-task test windows   {len(single)}")
    print(f"  model AUC on rebuilt index {model_auc:.6f}")
    print(f"  frozen pooled AUC          {FROZEN_POOLED_AUC:.6f}")
    print(f"  aligned                    {aligned}")

    print()
    print("=" * 68)
    print("1. Elapsed time alone, held-out participants")
    print("=" * 68)
    print(f"  all sources   n={len(test):5d}  AUC={report['clock_alone']['all_sources_test']:.3f}")
    print(f"  single-task   n={len(single):5d}  AUC={report['clock_alone']['single_task_test']:.3f}")

    if not aligned:
        print("\nNOT ALIGNED -- skipping every result that pairs windows with predictions.")
        return

    print()
    print("=" * 68)
    print("2. Headroom: rank blend of the frozen ensemble with the clock")
    print("=" * 68)
    print("   (blend weight chosen on these same nine participants; descriptive)")
    pct = lambda v: pd.Series(v).rank(pct=True).to_numpy()
    blend = {}
    for weight in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0):
        score = (1 - weight) * pct(single["model"]) + weight * pct(single["t_start"])
        blend[f"{weight:.1f}"] = rank_auc(single["elevated"], score)
        label = "model alone" if weight == 0 else ("clock alone" if weight == 1 else f"{weight:.0%} clock")
        print(f"  {label:16s} AUC={blend[f'{weight:.1f}']:.3f}")
    report["blend_auc_by_clock_weight"] = blend
    report["headroom"] = max(blend.values()) - blend["0.0"]

    print()
    print("=" * 68)
    print("3. Per participant: where the model is weak, is the clock strong?")
    print("=" * 68)
    per_participant = {}
    for gid, group in single.groupby("gid"):
        if group["elevated"].nunique() < 2:
            continue
        entry = {
            "windows": int(len(group)),
            "prevalence": float(group["elevated"].mean()),
            "model_auc": rank_auc(group["elevated"], group["model"]),
            "clock_auc": rank_auc(group["elevated"], group["t_start"]),
        }
        per_participant[gid] = entry
        print(
            f"  {gid:16s} n={entry['windows']:4d}  "
            f"model={entry['model_auc']:.3f}  clock={entry['clock_auc']:.3f}"
        )
    report["per_participant"] = per_participant

    model_scores = [v["model_auc"] for v in per_participant.values()]
    clock_scores = [v["clock_auc"] for v in per_participant.values()]
    if len(model_scores) > 2:
        corr = float(np.corrcoef(model_scores, clock_scores)[0, 1])
        report["model_clock_correlation_across_participants"] = corr
        print(f"\n  correlation between the two columns: {corr:+.3f}")
        print("  (negative means the two signals are complementary across people)")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
