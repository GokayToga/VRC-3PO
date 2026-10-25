#!/usr/bin/env python3
"""Familiar vs unseen participants: is there a person-specific sickness signal
beyond environment, person recognition and elapsed time? (benchmark v1.1)

Scores the architecture-study predictions (participant-held-out and purged
blocked protocols) against sensor-free baselines: elapsed time, an
environment prior and a person x environment prior taken from each
recording's training block. Reported in Section IV-D of the revised article.

    python -m analysis.familiar_vs_unseen --predictions architecture_test_predictions.csv \
        --dataset vrc3po_benchmark_v1_1.csv --output familiar_vs_unseen.csv [--sources maze]
"""
import argparse
import numpy as np, pandas as pd
from pathlib import Path
from analysis.revision_common import WeightedAUC, ParticipantBootstrap, percentile_ci
from analysis.corrected_evaluation import build_window_metadata
from analysis.split_protocols import chronological_purged_split

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--predictions", type=Path, required=True)
ap.add_argument("--dataset", type=Path, required=True, help="benchmark v1.1 table")
ap.add_argument("--output", type=Path, required=True)
ap.add_argument("--sources", default="simulations,terrain")
args = ap.parse_args()
pred = pd.read_csv(args.predictions)
SOURCES = args.sources.split(",")
pred = pred[pred.source_dataset.isin(SOURCES)].copy()
pred["y"] = (pred.y_true > 2).astype(int)
key = ["protocol", "variant", "recording_id", "start_index"]
s = pred.groupby(key).agg(score=("prediction", "mean"), y=("y", "first"), pid=("participant_id", "first"),
                          cond=("condition", "first"), src=("source_dataset", "first")).reset_index()

# training-block labels for the blocked protocol -> person x environment prior
table = pd.read_csv(args.dataset)
labels, meta = build_window_metadata(table)
meta["y"] = (labels > 2).astype(int)
meta["split"] = chronological_purged_split(meta).to_numpy()
train = meta[meta.split == "train"]
rec_prior = train.groupby("recording_id").y.mean()
env_prior_blocked = train.assign(env=train.source_dataset + ":" + train.condition).groupby("env").y.mean()
missing = set(s[s.protocol == "blocked_within_participant"].recording_id) - set(rec_prior.index)
print("recordings without a training block:", len(missing))

rows = []
B = 1000
for (protocol, variant), d in s.groupby(["protocol", "variant"]):
    d = d.reset_index(drop=True)
    y, sc = d.y.to_numpy(), d.score.to_numpy()
    env = (d.src + ":" + d.cond).to_numpy()
    rec = d.recording_id.to_numpy()
    minute = np.minimum(d.start_index.to_numpy() // 60, 30).astype(str)
    boot = ParticipantBootstrap(d.pid.to_numpy(), B, seed=1)
    cand = {"model": sc, "clock": d.start_index.to_numpy(float)}
    if protocol == "blocked_within_participant":
        cand["env prior"] = pd.Series(env).map(env_prior_blocked).fillna(0).to_numpy()
        cand["person x env prior"] = pd.Series(rec).map(rec_prior).fillna(pd.Series(env).map(env_prior_blocked)).to_numpy()
    for name, score in cand.items():
        r = {"protocol": protocol, "variant": variant, "score": name, "windows": len(y),
             "participants": d.pid.nunique(), "positives": int(y.sum())}
        for metric, g in (("pooled", None), ("within_env", env), ("within_env_minute", np.char.add(env.astype(str), minute)),
                          ("within_recording", rec)):
            f = WeightedAUC(y, score, g)
            lo, hi = percentile_ci(boot.run(f))
            r[metric] = f(); r[metric + "_ci"] = f"{lo:.3f}-{hi:.3f}"
        # pairs available within recordings (how much evidence the within-person metric rests on)
        if name == "model":
            r["recordings_with_both_classes"] = int(pd.DataFrame({"r": rec, "y": y}).groupby("r").y.agg(lambda v: 0 < v.sum() < len(v)).sum())
        rows.append(r)
    # model beyond person recognition: within strata of equal person x env prior is impossible (one prior per
    # recording), so test the paired difference model - person prior on within-env AUC
    if protocol == "blocked_within_participant":
        fm, fp = WeightedAUC(y, sc, env), WeightedAUC(y, cand["person x env prior"], env)
        lo, hi = percentile_ci(boot.run(lambda w: fm(w) - fp(w)))
        rows.append({"protocol": protocol, "variant": variant, "score": "model minus person prior (within env)",
                     "within_env": fm() - fp(), "within_env_ci": f"{lo:.3f}-{hi:.3f}"})
res = pd.DataFrame(rows)
res.to_csv(args.output, index=False)
pd.set_option("display.width", 260); pd.set_option("display.max_rows", 200)
cols = ["protocol", "variant", "score", "pooled", "pooled_ci", "within_env", "within_env_ci", "within_env_minute", "within_recording", "within_recording_ci", "recordings_with_both_classes"]
print(res[cols].round(3).to_string(index=False))
