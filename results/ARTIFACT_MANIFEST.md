# Frozen result artifact manifest

Created 24 July 2026. Session-scale and temporal-headroom entries added
28 July 2026 for the VRC-3PO release.

## Principal artifacts

| Artifact | SHA-256 | Purpose |
|---|---|---|
| `corrected/cnn_ensemble_preds_OG.npy` | `3b14ddc3e459c2d42c2fb710238b83d357bc1bda8db9b95f74ea1c1d3e20cda5` | Five-seed single-task CNN ensemble scores behind the binary headline |
| `colab/vrc3po_camera_ready_results_full.zip` | `d647c90703fd06a97dd5e34ec8ba6f0125db70583be5783ac53b55fb399be1de` | Complete Colab export for 42 runs (7 architectures, 2 protocols, 3 seeds) |

`colab/full/` is the tested extraction of the Colab ZIP. The archive includes
the protocol manifest, seed-level metrics, summaries, and test predictions.
Do not replace these files without regenerating their dependent tables,
figures, manuscript values, and checksums.

## Derived artifacts

Added 24 July 2026. These are recomputed by `analysis/operating_points.py`
from the two principal artifacts above and contain no new training.

| Artifact | SHA-256 | Purpose |
|---|---|---|
| `corrected/operating_points.csv` | `cdb87c44f1801679a29998d9400d0cb2a94edfba38ad39389af722ea3a95134e` | Four reference operating points with participant-cluster intervals; source of the operating-point table |
| `corrected/operating_point_sweep.csv` | `790d51069dde4872e6c1a65ba0828fc8a30e2a424227f50b6a328119ed49a8f9` | Confusion counts and rates at every distinct threshold; source of the threshold-sweep panel |
| `corrected/operating_point_summary.json` | `b39a71caf66dca31149b9c1834b9e1ec705e13c3e4fe711bf5997429da3260e6` | Machine-readable calibration and operating-point summary |
| `corrected/calibration_bins.csv` | `2ab30591fcf11f37e08b1c1a9479c0274edb01f09329995868ad221f9953efac` | Eight equal-count reliability bins; source of the reliability panel |
| `corrected/participant_diagnostics.csv` | `700d9fda2cb22387910fab2957a20799b7db653d812539b8938f9cd0b997c335` | Per-participant counts, prevalence and within-participant AUC; source of the single-task held-out participant table |

### Alignment guarantee

`cnn_ensemble_preds_OG.npy` stores 768 scores without metadata. The labels and
participant identifiers are recovered from
`colab/full/architecture_test_predictions.csv`, filtered to the
participant-held-out protocol and the nine single-task test participants, in file
order. `analysis/operating_points.py` refuses to write any output unless that
pairing reproduces the pooled AUC, window count, elevated count and
participant count recorded in `corrected/corrected_headline_metrics.json`, so
a silent misalignment cannot reach the manuscript.

## Composition robustness (protocol v2)

Added 25 July 2026. Produced by `analysis.composition_robustness` from cells
9--12 of `VRC3PO_Camera_Ready_Retrain.ipynb`: 20 participant compositions,
five CNN ensemble members each, 100 fits.

| Artifact | SHA-256 | Purpose |
|---|---|---|
| `composition/composition_results.csv` | `222b7eb5f8453ee596871a164cb9d7c4bb1aeab3a1181dfe3c07922d36e16c20` | Per-composition AUC, AP, participant-cluster interval, partition sizes; source of Fig. 7 and the composition subsection |
| `composition/composition_summary.json` | `1f347eb9db197a950435b28e99a863103ed81afed958148ef451e0647d1dda60` | Distributional summary quoted in Results, Discussion and the abstract |

### Protocol version

Every composition record carries `protocol_version`. Version 2 restricts
training, validation and test windows to the two single-task sources and includes
the leading `Masking` layer, matching the frozen reference endpoint exactly, so
the compositions differ from it in participant assignment alone. The resume
logic refuses to reuse records written under a different version.

A version 1 run trained on all three settings and reported a mean of 0.631
against 0.628 for version 2. The difference is negligible, but v1 results are
not comparable to the reference endpoint by construction and were discarded
rather than reported.

### Relationship to the reference split

The reference split (seed 42, AUC 0.757) is higher than all 20 redrawn
compositions and sits 1.53 SD above their mean. Seed 42 was fixed before any
model was trained and no composition was inspected before this analysis, so
the value was not selected; the manuscript states its position in the
distribution explicitly rather than leaving it to be derived from Fig. 7.

## Rebuild path from raw recordings

Added 25 July 2026. `analysis/build_master.py` converts the raw SAVELab
recordings to the pooled 1 Hz table, and `analysis/harmonize.py` applies the
two corrections that produce `vrc3po_master_dataset_fixed.csv`. The full chain
is therefore released:

    raw recordings -> build_master -> harmonize -> vrc3po_master_dataset_fixed.csv

Verified against the published table: identical shape (100,367 x 22), row keys,
participant set, per-source counts (terrain 18,900 / simulations 46,703 /
maze 34,764), `fms` and `elevated`.

**Known discrepancy.** 148 rows (0.15%) differ in feature values. Five
participant-condition pairs in Simulations contain two recording folders each
(participants 5, 6, 9, 25); both carry the same participant and condition label
and are concatenated into one session key. The original run ordered them by
filesystem enumeration, which is not stable across machines. The released loader
sorts the enumeration and warns when duplicates are detected. Consequences:
278 rows share a one-second index within a nominal session, and 113 of 6,442
windows (1.75%) mix two recordings. No affected participant is in the single-task
held-out test set, so the single-task endpoint artifacts above are unaffected.

Do not regenerate `vrc3po_master_dataset_fixed.csv` and reuse the frozen result
artifacts together: the rebuilt table differs in those 148 rows.

## Sentinel sensitivity check

Added 25 July 2026. Produced by `analysis/sentinel_sensitivity.py`. Answers
whether the residual negative-pupil artefact changes the single-task headline.

Design: the pooled table was harmonized twice from one source, once with the
published masking rule and once with `--strict-sentinel`, and the five-model
single-task ensemble was retrained on the frozen seed-42 split in both cases with
identical library and hardware. Only the masking rule differs between the runs.

| Run | Negative-pupil rows | Pooled single-task AUC | AP |
|---|---:|---:|---:|
| default (published rule) | 438 | 0.7726 | 0.2743 |
| strict | 0 | 0.7704 | 0.2704 |

Difference in pooled AUC: 0.0021. Largest per-member difference: 0.0032. The
artefact is not material to the single-task conclusion.

Both runs used TensorFlow 2.21 on CPU and land near 0.771, against 0.757 for the
published TensorFlow 2.20 GPU run. That 0.016 offset is library and hardware
variation and is eight times larger than the artefact under test, which is why
the comparison is between two local runs rather than against the published
value. These runs are a sensitivity check and do not supersede the frozen
single-task artifacts.

## Session-scale and temporal-headroom artifacts

Added 28 July 2026. The session-scale conditions were trained on a T4; the
temporal-headroom summary involves no training and is recomputed from the
principal artifacts above.

| Artifact | SHA-256 | Purpose |
|---|---|---|
| `session_scale/session_scale_predictions.npz` | `c2f489a9e52ff42f6aa39b8bd95c1782d6030cff3b2537ea240cf81f70b48948` | Per-window scores for all seven session-scale conditions on the same 768 held-out windows |
| `session_scale/session_scale_results.json` | `3399d60a5890b82f3b5583a46563f4a6430e81f2d619034028280eadc7a8bba1` | Pooled AUC and marginal participant-cluster intervals per condition |
| `session_scale/session_scale_paired_comparisons.json` | `97557bfbfa407d2861b0eb766a08005cd26d40673adb931e7adc44115c2a0dfa` | Paired differences against the window baseline, as written by the notebook |
| `session_scale/pairwise_comparisons.json` | `6adb0cc7c8c93a7f45d22b8962a9cce773d7961c51cc0dad754bac29c8cc43c8` | All 21 pairwise paired comparisons, recomputed from the predictions |
| `temporal_headroom/summary.json` | `627d05bad2d513cd1d1c51cbfb288d40c77d2b360bf6c6ee8f47bfb21355b9b5` | Clock-alone AUC, blend headroom and the per-participant model/clock split |

### Alignment guarantee for the session-scale family

Both the notebook and `analysis/session_scale_comparisons.py` rebuild the window
index independently of the training pipeline and check it against the frozen
ordering before reporting anything: the rebuilt single-task test set must
reproduce pooled AUC 0.756948 under the frozen predictions. The notebook
additionally asserts that every condition lands on identical label and
participant vectors, which is what makes the paired comparison valid.
