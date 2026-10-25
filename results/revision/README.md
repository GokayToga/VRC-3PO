# Results of the first revision (benchmark v1.1)

Every number added or changed in the revised article traces to a file here.
All intervals resample participants (2,000 replicates unless stated).

| Article item | File |
|---|---|
| Table 2, harmonization counts (Section III-B) | `benchmark_v1_1/vrc3po_benchmark_v1_1.manifest.json` |
| Table 4 (repeated CV and fixed split), Section IV-A | `r1_primary_cv/primary_estimates.csv` (`fms_gt2`, `seated`, `cv` / `reference`) |
| Feature-based and single-classifier comparisons (Section IV-A) | `r1_primary_cv/primary_estimates.csv` (`feat:*`, `clf:*`) |
| Table 5, validation-selected operating points | `r1_primary_cv/operating_points_validation_selected.csv` |
| Table 6, environment control; increments | `r1c_shortcut_aware/primary_estimates.csv`, `r1c_shortcut_aware/environment_control.csv` |
| Table 7, environment probe | `r1b_environment/environment_probe.csv` |
| Input variants (Section IV-B) | `r1b_environment/primary_estimates.csv`, `r1b_environment/environment_control.csv` (scopes `seated-<channels>-<raw|cal60>-s60`) |
| Table 8, shortcut-aware training (98.75% intervals in `*_ci98_*` columns) | `r1c_shortcut_aware/primary_estimates.csv`, `r1c_shortcut_aware/environment_control.csv` (`debias:*`) |
| Table 9, architecture comparison (v1.1); Fig. 3 | `architecture_v1_1/table5_architecture_ablation.csv`, `architecture_v1_1/table4_protocol_comparison.csv` |
| Section IV-D, person prior and elapsed time | `r1b_environment/familiar_vs_unseen/` (`analysis/familiar_vs_unseen.py`) |
| Section IV-E, 20 compositions; Fig. 4 | `composition_v1_1/` |
| Section IV-E, architectures crossed with test folds | `r1_primary_cv/variance_components.json` |
| Section IV-F, single-task vs Maze; Fig. 5 | `architecture_v1_1/architecture_test_predictions.csv`, `paradigm_gap_v1_1/paradigm_gap_cluster_summary_v11.json` |
| Table 10, source-aware thresholds | `r1c_shortcut_aware/primary_estimates.csv`, `r1c_shortcut_aware/environment_control.csv` (labels `floor_plus2`, `floor_plus1`, `baseline_plus1`) |
| Equal-budget tuning (Section IV-G) | `r1c_shortcut_aware/primary_estimates.csv`, `r1c_shortcut_aware/paired_comparisons.csv` (`tuned:*`) |
| Table 11, published detectors; FMS regression (Section IV-H) | `r1d_published_detectors/` (`regression.csv` for R^2) |

Notes

- The stores are cumulative summaries of one run store each: `r1c_shortcut_aware`
  contains everything in `r1_primary_cv` and `r1b_environment` plus the
  shortcut-aware runs; `r1d_published_detectors` contains all runs of notebooks
  R1d and R1e plus the sample-level runs on the 47 single-task participants.
- In the summaries, the scope name `seated` denotes the two single-task sources
  (Simulations and Terrain); not every Simulations environment is seated.
- `composition_v1_1/composition_summary.json` reports `frozen_split_auc` 0.757,
  which is the v1.0 reference split. On v1.1 the reference split scores 0.796
  (`r1_primary_cv/primary_estimates.csv`, protocol `reference`); the article
  uses 0.796.
- `architecture_test_predictions.csv` holds per-window predictions with
  window-mean FMS; it contains no raw recordings.
