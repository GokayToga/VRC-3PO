# Corrected VRC-3PO evaluation

`corrected_evaluation.py` evaluates the saved passive CNN ensemble predictions
without treating overlapping windows as independent observations.

Run:

```bash
python corrected_evaluation.py \
  --dataset /path/to/vrc3po_master_dataset_fixed.csv \
  --predictions /path/to/cnn_ensemble_preds_OG.npy \
  --output-dir ../results/corrected
```

Generated files:

- `corrected_headline_metrics.json`: machine-readable source of headline values.
- `per_participant_metrics.csv`: participant counts, prevalence, AUC, and AP.
- `per_source_metrics.csv`: source-specific results with participant counts.
- `manuscript_replacements.md`: replacement-ready text and caption corrections.

The participant-cluster bootstrap resamples whole participants. It preserves
the dependence created by repeated and overlapping windows within each person.
The participant-macro AUC is defined only for participants who have at least one
elevated and one non-elevated window.

## Leakage-resistant window comparison

Generate the chronological split manifest with:

```bash
python -m analysis.split_protocols \
  --dataset /path/to/vrc3po_master_dataset_fixed.csv \
  --output-dir ../results/corrected
```

The default protocol uses 60/20/20 chronological blocks within each session,
retains only windows wholly contained in their block, and discards a 15-second
buffer at each boundary.

## Seven-architecture retraining

`retrain_camera_ready.py` is intended for the TensorFlow/Colab environment used
for the original experiments:

```bash
python -m analysis.retrain_camera_ready \
  --dataset /content/drive/MyDrive/vrc3po_master_dataset_fixed.csv \
  --output-dir /content/drive/MyDrive/vrc3po_camera_ready
```

It trains all seven architectures under the participant-held-out and blocked
within-participant protocols. Tables 4 and 5 are derived from the same
`architecture_seed_metrics.csv`, preventing the VRC-3PO row from diverging
between tables.
