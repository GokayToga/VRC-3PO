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


## First revision (benchmark v1.1)

Added 25 October 2026. Provenance and table mapping in `revision/README.md`.

| Artifact | SHA-256 |
|---|---|
| `revision/architecture_v1_1/architecture_seed_metrics.csv` | `7fe5174f00ef6d42f2526d86d8c5818af964447db2e16706b3f6c63a1003af5a` |
| `revision/architecture_v1_1/architecture_summary.csv` | `53df5dad27163adb05fc5be348591ae559a3755d576ceacc351bad1ac52e3963` |
| `revision/architecture_v1_1/architecture_test_predictions.csv` | `469de0bcd9919ce9e0f48b6bd729adfa033393cdfffa778f503b1af67a6921f3` |
| `revision/architecture_v1_1/protocol_manifest.json` | `0f887f8e7b17d72d292b2bca3703f2d759f24eb1666f45fb233b6b9bce9f6a57` |
| `revision/architecture_v1_1/table4_protocol_comparison.csv` | `ee732e597bf450f45b6db24d9f72e82269dddc033f3d37d4b07df99fd67a56b6` |
| `revision/architecture_v1_1/table5_architecture_ablation.csv` | `4825812c08b790d358bdc93b71eed7b33674d9415b19a92f930d8aa6b1929e6d` |
| `revision/benchmark_v1_1/vrc3po_benchmark_v1_1.manifest.json` | `457ddec4fd3b20e5a1c6044147f76933cff1f11526ed44fa0c58ce29635ff995` |
| `revision/composition_v1_1/composition_000.json` | `5ee5115bae12150e16920919f0fcae4fc7a27c89b78c0b58daf920bf536a6977` |
| `revision/composition_v1_1/composition_001.json` | `a4411ab88051618b93377bda9611bf7d7f29f6eecae2e2d6c7ead14213bcccd0` |
| `revision/composition_v1_1/composition_002.json` | `9f4552a0f722dbf0fb89adfe7e732115888d465792de2ebf585ecf0d13af8584` |
| `revision/composition_v1_1/composition_003.json` | `aadb346b04961216b110414647532ae2679be2c7e8d3bfccf4dab8a00b44107d` |
| `revision/composition_v1_1/composition_004.json` | `53f02e22c773b2a696ae786143441436bd302926621761dd4ddde54ec043de19` |
| `revision/composition_v1_1/composition_005.json` | `28ea4c86d623b49a9a75e229709ebdd7ed8242e038ef583e65a71c76525523df` |
| `revision/composition_v1_1/composition_006.json` | `729d2c818a6b0281095413bcb503bac954899cfbccbc1beca23820735c9ada4c` |
| `revision/composition_v1_1/composition_007.json` | `4e43d7857eb765d13e940e8fca7f09f64b33b76e03ecf3d3b5ee2f3d14092b41` |
| `revision/composition_v1_1/composition_008.json` | `eacc7d30e098fb06027285a6f71357983f1b764af6f66083f419ef6ee834e6b4` |
| `revision/composition_v1_1/composition_009.json` | `513f07273ecd3fa73f268734296a47acded78b2244b73e1eff893fb8fc243261` |
| `revision/composition_v1_1/composition_010.json` | `c435ddfe26b9893f120e5c399e688050a861c6f08ba118b642badc5e1c3e6315` |
| `revision/composition_v1_1/composition_011.json` | `5a5befcdadf082fb30da2ce7b16cb41d558ad02dbc593b5b8f8bb6dbe88418e1` |
| `revision/composition_v1_1/composition_012.json` | `f36e28532cdb181a536fa4507ff8b70326cfe054bb4ea3f5ea292204bc38a16e` |
| `revision/composition_v1_1/composition_013.json` | `656516af164a70808db35761f347b2c3097f1222af1717d0b9a7aedd9f76bb1b` |
| `revision/composition_v1_1/composition_014.json` | `e0e082fd8303027514b20bfef15779b78e717268f505eab948ec5837efd32b8f` |
| `revision/composition_v1_1/composition_015.json` | `baec510163773019e7e18838d78345ee6d8871adbde987168811826ad03a9138` |
| `revision/composition_v1_1/composition_016.json` | `52d93164359575642f1d27e142bc135e85ceef93c22a438dec798514ee7cfa23` |
| `revision/composition_v1_1/composition_017.json` | `ee3e34b7a30bee363b1dd9756df79e35577819f89715cad114c71b862d2d6546` |
| `revision/composition_v1_1/composition_018.json` | `12c3129f460e997cf36ece107214e8f7c634156542111151bdf96179c61acd87` |
| `revision/composition_v1_1/composition_019.json` | `2707d81022e14dd134e2019252fb450fd3a3aea87fe3695531086c0aff2c18f3` |
| `revision/composition_v1_1/composition_results.csv` | `62aafc0b28d7113dd2496d28cd13ef7e93f56c416925cff2c6cbbb49e742059e` |
| `revision/composition_v1_1/composition_summary.json` | `3e05e2ee6108b2ff2cd213edf5d976c8030810a6931c8a4dd4f4a4cdd89472f3` |
| `revision/paradigm_gap_v1_1/paradigm_gap_cluster_summary_v11.json` | `9ad37062ef7027ce631bfba4336f75e44394cffcdcac78c94a0f18a9153f3b9c` |
| `revision/r1_primary_cv/environment_control.csv` | `1a739874e1cef50b97cd12fa4a8505433afe65d4bdbf346a976147b7d7b4313d` |
| `revision/r1_primary_cv/fold_auc.csv` | `e29c58ec5484a093428240011678cb0aaedbd2775e2bfadc062f3d35711084c3` |
| `revision/r1_primary_cv/operating_points_validation_selected.csv` | `dcdff48518cab543aeec3cd20cfb1c6fad9c251c728556d335313bd4633a4283` |
| `revision/r1_primary_cv/paired_comparisons.csv` | `8c2fdcd5dee05a64344f47321d74852a783e1ad1b3c7ce16c7b15d201f7d313f` |
| `revision/r1_primary_cv/per_condition_auc.csv` | `00d16674c778eda251101e7f935d8a15ecce46770f88f7f0c28cba487c8b3b17` |
| `revision/r1_primary_cv/primary_estimates.csv` | `1a13e5862a30aca0024b874541e7bed76a8c0691f810ec51f439348315211945` |
| `revision/r1_primary_cv/run_status.csv` | `90ad79b77cd9e8bd03da4ac17c9731d1e5b3ae5e17b8510e956bcc559a7eacb4` |
| `revision/r1_primary_cv/variance_components.json` | `0a4194ebd7cf40dde6aea27358729df6dfbe43316cb2b97b2289318fa02eae9c` |
| `revision/r1b_environment/environment_control.csv` | `ce469f1562f75744d7d9e1fd298b14a14ee126d8f302a33ee682e85754136e36` |
| `revision/r1b_environment/environment_probe.csv` | `f56b937592562954a2595cd42fc30ff1598ef569d3901e6bbec5d40b800eebc3` |
| `revision/r1b_environment/familiar_vs_unseen/familiar_vs_unseen.csv` | `9c0548a9d94b5f4811122a2289f132a3b84a4ab720f8f7a881a658b8f6aa6696` |
| `revision/r1b_environment/familiar_vs_unseen/familiar_vs_unseen_maze.csv` | `ee130e453b915496302ab717d9be2eeefa5ccb09e20f59dd5884341edd28d396` |
| `revision/r1b_environment/fold_auc.csv` | `90981007d7d5dc1dcb5a6ff70e06b1f04ad60613ea5be72cbb28c0683a7f7e52` |
| `revision/r1b_environment/input_variants.csv` | `f3b1077c26bdc0d7d19322b27081b2f4501f015512747b5991069d929c9df96a` |
| `revision/r1b_environment/operating_points_validation_selected.csv` | `a50eaad0519af4f2fb297e89a1748cd4096b5af42d7c2bac1d67ebf9ddc2259e` |
| `revision/r1b_environment/paired_comparisons.csv` | `455426b80a11b6880a1332d80317932e01fe7fbb7422ce8f694e8760aafa6c82` |
| `revision/r1b_environment/per_condition_auc.csv` | `0daccf6214e34b8b801092c4c750939cbd228598efc5b463cee544e336623b20` |
| `revision/r1b_environment/primary_estimates.csv` | `a29d4d4ef338f25c3acdcafd88c774eea4dab7e8be5aed76528bea743d04acf1` |
| `revision/r1b_environment/variance_components.json` | `cc40a6ff2ebb8a3c890acbaa5c96e7fd1f83bc3e2ba842f8af044b8073710c65` |
| `revision/r1c_shortcut_aware/environment_control.csv` | `1a8615ab4307df7eb7e660fe5b336bdc64993f77d5baea0784ee17d945477e32` |
| `revision/r1c_shortcut_aware/fold_auc.csv` | `e83b911eac5faab03bb12a4341cd2ec25476ea5b9e6860fa0a0e4903dc1a0ecb` |
| `revision/r1c_shortcut_aware/input_variants.csv` | `f3b1077c26bdc0d7d19322b27081b2f4501f015512747b5991069d929c9df96a` |
| `revision/r1c_shortcut_aware/operating_points_validation_selected.csv` | `b206f516b95d67d425e2166db3139b9a8954786571eb48769bc37855fb709574` |
| `revision/r1c_shortcut_aware/paired_comparisons.csv` | `7639c755be80a1ff0c1a89b50a36b3448589812c7ec380cd78c5cc3bda282028` |
| `revision/r1c_shortcut_aware/per_condition_auc.csv` | `577a23569f7781f4441b743a878cca4f3852d4380f4001752d5fdd8c2a27d221` |
| `revision/r1c_shortcut_aware/primary_estimates.csv` | `956600903bedb3df4fdede10538623997ab6bbf72a49aaf322598d0f897c1915` |
| `revision/r1c_shortcut_aware/variance_components.json` | `cc40a6ff2ebb8a3c890acbaa5c96e7fd1f83bc3e2ba842f8af044b8073710c65` |
| `revision/r1d_published_detectors/environment_control.csv` | `b44b43079fff2fd02ffcb922d826781b41306122b233a30646a98f17f05e168b` |
| `revision/r1d_published_detectors/fold_auc.csv` | `215b824f2f115ae8bf1c93e3567e7a7f09d0089303d5f21dd9e163325582e263` |
| `revision/r1d_published_detectors/input_variants.csv` | `01ba4719c80b6fe911b091a7c05124b64eeece964e09c058ef8f9805daca546b` |
| `revision/r1d_published_detectors/operating_points_validation_selected.csv` | `9a9ecac60636dc6b232a5a91e73eb0d0878fcbc528539f713f9407ca12c3418b` |
| `revision/r1d_published_detectors/paired_comparisons.csv` | `036b2480004309859cdc30dc2a34c055b38400f2a1a600173f09a16e9a738498` |
| `revision/r1d_published_detectors/per_condition_auc.csv` | `ae3925046c54eeeccc482105b079842f9b24c2855c5a722ba13ba8f46a1e6756` |
| `revision/r1d_published_detectors/primary_estimates.csv` | `c3fd48387b723e5be644c027b0c67dd578b3d7fac370648e13d424dda67f9c63` |
| `revision/r1d_published_detectors/regression.csv` | `fb1d5d34aeb59b61439239b7c9cf52155f9e07a7b024b69cadabe9f9d1d2583d` |
| `revision/r1d_published_detectors/variance_components.json` | `ca3d163bab055381827226140568f3bef7eaac187cebd76878e0b63e9e442356` |
