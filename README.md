# VRC-3PO

Reproducibility archive for *VRC-3PO: A Cross-Participant Temporal Benchmark for
Cybersickness Detection From Eye and Head Tracking Across Three VR Settings*
(IEEE Access, first revision).

VRC-3PO names the full temporal stack the article evaluates -- convolution for
local gaze events, dilated convolution for medium-range build-up, recurrence for
accumulated state, attention over timesteps. It is the hypothesis under test, not
a proposal: it does not win under participant holdout.

This archive contains the analysis code, the frozen result artifacts behind every
number in the article, and the scripts that regenerate every table and figure. **It does not contain the source recordings** -- see Data access
below.

## Revision 1 (benchmark v1.1)

The first revision of the article changes the benchmark and its primary analyses.
Everything added for it is listed here; the first-submission material below is
kept unchanged so that v1.0 numbers remain reproducible.

- **Benchmark v1.1** (`analysis/benchmark_v11.py`) rebuilds the 1 Hz table from
  the raw recordings with three corrections: invalid eye samples are masked at
  the native sampling rate before binning (Maze validity flag; Simulations pupil
  or openness < 0), Terrain pupil values below a 1.0 mm plausibility floor are
  masked, and windows are formed per recording (`recording_id`), with
  duplicated recordings removed. 99,767 rows, 221 recordings, 6,401 windows.
  The manifest of removed recordings and masked values is
  `results/revision/benchmark_v1_1/vrc3po_benchmark_v1_1.manifest.json`.
- **Repeated participant-level cross-validation** (`analysis/revision_cv.py`,
  `analysis/revision_common.py`): 5 x 5 stratified participant folds over the
  47 single-task participants, participant-cluster bootstrap, validation-selected
  operating points, sensor-free baselines (`nuisance:condition`,
  `nuisance:condition_clock`), within-environment AUC, the increment over the
  baselines, source-aware threshold definitions, equal-budget tuning, and input
  variants (channel subsets, first-minute calibration).
- **Environment probe** (`analysis/environment_probe.py`): how well each channel
  group identifies the VR environment for unseen participants.
- **Shortcut-aware training** (`analysis/revision_debias.py`), pre-registered in
  `analysis/PREREGISTRATION_R1c.md` before it was run.
- **Familiar participants and time on task** (`analysis/familiar_vs_unseen.py`):
  person x environment prior and elapsed-time baselines.
- **Published detectors** (`analysis/litevr.py`): re-implementations of the
  LiteVR LSTM/GRU/MLP and the TruVR boosting model, under participant holdout and
  under a sample-level split over windows (`--protocol windows`), plus FMS
  regression (`litevr_reg:*`, summarized by `python -m analysis.litevr regression`).

Colab notebooks, run in order on a T4 (each is resumable):

```
VRC3PO_Revision_R1.ipynb                          benchmark v1.1, primary CV, baselines,
                                                  thresholds, tuning, architectures x folds
VRC3PO_Revision_R1b_Environment_Shortcut.ipynb    environment probe and input variants
VRC3PO_Revision_R1c_Shortcut_Aware_Training.ipynb pre-registered shortcut-aware training
VRC3PO_Revision_R1d_Published_Detectors.ipynb     LiteVR and TruVR re-implementations
VRC3PO_Revision_R1e_Matched_Comparison.ipynb      CNN ensemble and FMS regression under both protocols
```

The notebooks install the code from a zip of this repository uploaded to Google
Drive (the "bundle"); a zip of the repository root works.

Every revised number traces to `results/revision/`; see
`results/revision/README.md` for the table-by-table map.

## What is here

```
analysis/                     harmonization, analysis and figure-generation
                              modules
tests/                        unit tests for the split and metric code
VRC3PO_Camera_Ready_Retrain.ipynb
                              Colab notebook: full retraining (42 fits)
VRC3PO_Composition_Robustness.ipynb
                              Colab notebook: 20-composition robustness study
VRC3PO_Session_Scale_Temporality.ipynb
                              Colab notebook: context beyond the 30-second window
results/corrected/            single-task binary endpoint, participant-cluster
                              statistics, operating points, calibration
results/composition/          20-composition robustness outputs
results/session_scale/        session-scale conditions, per-window predictions
                              and paired participant-cluster comparisons
results/temporal_headroom/    what the 30-second window cannot see, computed
                              from frozen predictions without retraining
results/sentinel_check/       A/B isolation of the pupil masking rule
results/colab/full/           42-run architecture bundle (7 architectures x
                              2 protocols x 3 seeds), seed-level metrics and
                              per-fit predictions
results/revision/             benchmark v1.1 and every result of the first revision
results/ARTIFACT_MANIFEST.md  SHA-256 checksums and provenance for every
                              frozen artifact
DATA_DICTIONARY.md            schema, per-source counts and reconstruction
                              checks for the harmonized input table
manuscript/figures/           final figures as vector PDF
manuscript/data/              numerical sources behind the article tables
manuscript/reference_audit.md publisher-level bibliography and data-provenance
                              audit
```

## Data access

The three source datasets have different access routes and none of them are
redistributed here.

- **Maze** is the openly released *Mazed and Confused* / VRWalking dataset and
  can be downloaded directly:
  <https://github.com/Jyotinag/VRWalking_Dataset>
- **Simulations** and **Terrain** are held by the SAVELab at the University of
  Texas at San Antonio, are not part of that release, and were obtained by
  completing the group's data request form. Other researchers can obtain them
  the same way. Direct requests to the SAVELab, not to the authors of this
  archive.

The single-task detection result reported in the article uses Simulations and
Terrain only, so reproducing it requires a data request. The scripts expect a single harmonized CSV,
`vrc3po_master_dataset_fixed.csv`, with one row per participant-second and the
14 shared channels. `DATA_DICTIONARY.md` gives the full schema, per-source row
and window counts, and a set of checks to confirm a rebuilt table matches the
one used here before comparing any numbers.

## Reproducing the reported numbers

Two of the three analyses need no training and run from the frozen artifacts.

**Operating points and calibration** (operating-point and per-participant
tables, reliability figure):

```sh
python -m analysis.operating_points
python -m analysis.generate_operating_point_figure
```

`analysis/operating_points.py` refuses to write output unless the score/label
pairing reproduces the pooled AUC, window count, elevated count and
participant count recorded in
`results/corrected/corrected_headline_metrics.json`. A silent misalignment
therefore cannot reach the manuscript.

**Schematic and architecture figures** need no dataset:

```sh
python -c "from pathlib import Path; from analysis.generate_ieee_figures import figure_study_design, set_style; set_style(); figure_study_design(Path('manuscript/figures'))"
python -m analysis.generate_architecture_figure
```

The study-design schematic measures each label against the box it sits in and
shrinks the font until it fits, so a wording change cannot silently overflow.
The architecture figure reads its numbers from
`manuscript/data/table_architecture_ablation.csv`, so it cannot drift from the
architecture table.

Figure numbers in the article are assigned by LaTeX in order of appearance and
do not match the `figN_` filenames. Reference figures by label.

**Data-dependent figures** (detection curves, protocol gap, participant
variation, setting gap) need the harmonized dataset:

```sh
python -m pip install -r analysis/requirements-figures.txt
MPLCONFIGDIR=/tmp/mpl python -m analysis.generate_ieee_figures \
    --dataset /path/to/vrc3po_master_dataset_fixed.csv
```

**Retraining** (42 fits) and the **composition-robustness study** (20
compositions x 5 ensemble members) are GPU jobs. Open
`VRC3PO_Camera_Ready_Retrain.ipynb` in Colab, choose a T4 runtime, and run the
cells in order; both jobs are resumable and skip completed work.

Composition records carry a `protocol_version` field. Version 2 restricts
training, validation and test windows to the two single-task sources and includes
the leading `Masking` layer, matching the frozen reference endpoint so that
compositions differ from it in participant assignment alone. The resume logic
refuses to mix versions.

**Harmonization.** `analysis/harmonize.py` implements the two corrections
applied to the pooled 1 Hz table: gaze renormalization to unit vectors, and
pupil sentinel masking with within-session forward/backward fill. Verify a table
against the published invariants with:

```sh
python -m analysis.harmonize --verify /path/to/vrc3po_master_dataset_fixed.csv
```

The module reproduces the published behaviour by default, including a known
ordering defect: the invalid-marker test matches values exactly equal to -1 but
runs after one-second averaging, so bins that mixed valid samples with markers
leave 438 rows (0.44%) with negative pupil values in Simulations and Terrain.
Pass `--strict-sentinel` to apply the correct rule; that output will **not**
reproduce the published numbers and exists for new work only.

**Rebuilding the pooled table from raw recordings.** `analysis/build_master.py`
converts the three sources from their native formats into the pooled 1 Hz table.
Sources are staged separately because Maze is ~2.6 GB; Maze additionally caches
per participant and can be resumed:

```sh
RAW=/path/to/utsa_eye_head_cybersickness_datasets
python -m analysis.build_master --raw-dir $RAW --stage-dir stage --source terrain
python -m analysis.build_master --raw-dir $RAW --stage-dir stage --source simulations
python -m analysis.build_master --raw-dir $RAW --stage-dir stage --source maze
python -m analysis.build_master --stage-dir stage --merge --output pooled.csv
python -m analysis.harmonize --input pooled.csv --output vrc3po_master_dataset_fixed.csv
```

This reproduces the released table in shape (100,367 rows), row keys,
participant set, per-source counts (18,900 / 46,703 / 34,764), `fms` and
`elevated`. 148 rows (0.15%) differ in feature values: five participant-condition
pairs in Simulations contain two recording folders each, both are labelled with
the same participant and condition, and the original run ordered them by
filesystem enumeration. This module sorts the enumeration so results are stable
across machines, and warns when duplicates are found. 278 rows have colliding
one-second indices and 113 windows (1.75%) mix two recordings; none of the
affected participants are in the single-task held-out test set. See the article's
Methods and Limitations sections.

**Session-scale context** (does anything beyond the 30-second window help?) is a
GPU job: open `VRC3PO_Session_Scale_Temporality.ipynb` in Colab on a T4 and run
the cells in order, roughly 35 minutes. The notebook refuses to proceed unless its
independently rebuilt window index reproduces the frozen ordering (pooled AUC
0.756948), so a misalignment cannot reach a reported number. It writes per-window
scores for every condition, from which

```sh
python analysis/session_scale_comparisons.py \
    --predictions results/session_scale/session_scale_predictions.npz \
    --output results/session_scale/pairwise_comparisons.json
```

recomputes all 21 pairwise paired comparisons without retraining. Conditions are
compared by resampling participants once per replicate and scoring both models on
that same draw; comparing two marginal intervals instead would make every
comparison look inconclusive, because most of each interval is the shared fact
that only nine people are scored.

**Temporal headroom** needs no GPU and no retraining:

```sh
python analysis/temporal_headroom.py \
    --dataset /path/to/vrc3po_master_dataset_fixed.csv \
    --split-manifest results/corrected/split_manifest.csv \
    --predictions results/corrected/cnn_ensemble_preds_OG.npy
```

**Tests:**

```sh
python -m pytest tests/ -q
```

## Environment

The reported results were produced in Google Colab on an NVIDIA T4 GPU with
Python 3.12, TensorFlow 2.20.0, Keras 3.13.2 and keras-tcn 3.5.6. The
revision notebooks additionally install `interpret-core` for the boosting model. Analysis and
figure scripts additionally use the versions pinned in
`analysis/requirements-figures.txt`. Deep-learning results are not
bit-reproducible across GPU models and library versions; seeds are fixed
(42, 123, 456 for the architecture comparison, and 42, 123, 456, 789, 2024 for
the five-model ensemble) but small numerical differences should be expected.

## Scope of this archive

This is the pipeline that produces the published results. Exploratory
development notebooks are not included: their intermediate numbers were
produced under earlier, superseded evaluation protocols and do not correspond
to anything reported in the article. Every value in the article traces to a
file listed in `results/ARTIFACT_MANIFEST.md`.

## Citation

See `CITATION.cff`. Please cite the article and, where relevant, the source
datasets — in particular *Mazed and Confused* for the Maze source.

## License

Code in this archive is released under the MIT License (`LICENSE`). It does not
cover the source recordings, which remain under the terms set by the SAVELab.
