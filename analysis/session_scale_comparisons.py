#!/usr/bin/env python3
"""Every pairwise paired comparison among the session-scale conditions.

The notebook compares each condition against the window baseline. Several of the
scientifically interesting comparisons are between two non-baseline conditions --
ordered versus shuffled context, learned recurrence versus a hand-coded clock --
so this script recomputes the full matrix from the saved per-window predictions.
No retraining: it reads ``session_scale_predictions.npz`` only.

Why paired. All conditions score the same 768 windows from the same nine
participants, so their marginal intervals share one large source of variance: which
nine people happen to be in the test fold. Differencing two marginal intervals
throws that shared term away and makes every comparison look inconclusive.
Resampling participants once per replicate and scoring both conditions on that same
draw removes it. The window/window_clock pair is the clearest illustration -- the
marginal intervals overlap heavily while the paired difference excludes zero.

A paired interval containing zero says this experiment cannot separate two models on
nine participants. It is not evidence that they are the same model.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_REPLICATES = 5000
BASELINE = "window"


def rank_auc(y: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y)
    ranks = pd.Series(np.asarray(scores)).rank().to_numpy()
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def paired_difference(
    y: np.ndarray,
    participants: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    replicates: int,
    seed: int = 42,
) -> dict:
    """Interval on AUC(a) - AUC(b) over a shared participant resample."""
    rng = np.random.default_rng(seed)
    unique = np.array(sorted(set(participants.tolist())))
    index_of = {p: np.where(participants == p)[0] for p in unique}
    differences = []
    for _ in range(replicates):
        picked = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([index_of[p] for p in picked])
        auc_a = rank_auc(y[idx], scores_a[idx])
        auc_b = rank_auc(y[idx], scores_b[idx])
        if not (np.isnan(auc_a) or np.isnan(auc_b)):
            differences.append(auc_a - auc_b)
    differences = np.asarray(differences)
    lower = float(np.percentile(differences, 2.5))
    upper = float(np.percentile(differences, 97.5))
    return {
        "difference": rank_auc(y, scores_a) - rank_auc(y, scores_b),
        "lower": lower,
        "upper": upper,
        "excludes_zero": bool(not (lower <= 0 <= upper)),
        "fraction_favouring_a": float((differences > 0).mean()),
        "replicates": int(len(differences)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    archive = np.load(args.predictions, allow_pickle=True)
    y = archive["labels"]
    participants = archive["participants"]
    scores = {
        key.removeprefix("scores__"): archive[key]
        for key in archive.files
        if key.startswith("scores__")
    }

    print(f"{len(y)} windows, {int(y.sum())} elevated, "
          f"{len(set(participants.tolist()))} participants, {len(scores)} conditions\n")
    print("pooled AUC")
    for name, values in scores.items():
        print(f"  {name:20s} {rank_auc(y, values):.3f}")

    comparisons = {}
    for name_a, name_b in itertools.combinations(scores, 2):
        result = paired_difference(
            y, participants, scores[name_a], scores[name_b], args.replicates
        )
        comparisons[f"{name_a} - {name_b}"] = result

    print(f"\npaired differences ({args.replicates} participant-cluster replicates)")
    print("  * marks an interval that excludes zero\n")
    rows = []
    for label, result in comparisons.items():
        mark = "*" if result["excludes_zero"] else " "
        print(
            f" {mark} {label:42s} {result['difference']:+.3f}  "
            f"[{result['lower']:+.3f}, {result['upper']:+.3f}]  "
            f"P={result['fraction_favouring_a']:.2f}"
        )
        rows.append({"comparison": label, **result})

    decisive = [r for r in rows if r["excludes_zero"]]
    print(f"\n{len(decisive)} of {len(rows)} comparisons exclude zero.")
    print("Comparisons against the baseline that do NOT exclude zero mean the added")
    print("mechanism is not separable from ignoring session context entirely.")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "n_windows": int(len(y)),
                    "n_elevated": int(y.sum()),
                    "n_participants": len(set(participants.tolist())),
                    "replicates": args.replicates,
                    "pooled_auc": {k: rank_auc(y, v) for k, v in scores.items()},
                    "paired_comparisons": comparisons,
                },
                indent=2,
            )
        )
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
