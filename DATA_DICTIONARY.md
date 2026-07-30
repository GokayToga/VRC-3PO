# Data dictionary: `vrc3po_master_dataset_fixed.csv`

The harmonized table that every analysis in this archive consumes. The file
itself is **not** redistributed here; see README "Data access". This
dictionary lets a reader who has obtained the source recordings verify that
they have rebuilt the same table before comparing any numbers.

Shape: **100,367 rows x 22 columns**, one row per
participant-second at 1 Hz.

## Per-source composition

| Source | Rows | Participants | Conditions | Windows (30 s, 15 s stride) | Elevated windows |
|---|---:|---:|---:|---:|---:|
| maze | 34,764 | 37 | 1 | 2,262 | 598 (26.4%) |
| simulations | 46,703 | 25 | 5 | 2,983 | 411 (13.8%) |
| terrain | 18,900 | 22 | 3 | 1,197 | 247 (20.6%) |
| **total** | **100,367** | **84** | --- | **6,442** | **1,256** (19.5%) |

## Columns

| Column | Type | Role | Notes |
|---|---|---|---|
| `elapsed_s` | int/float | index | Seconds from session start; sorted within each participant-condition session. |
| `fms` | float | label source | Fast Motion Sickness score, 0-10, carried forward between ratings. Sampled every 30-60 s depending on source, so values are stepwise. |
| `pupil_diam_L` | float | **model input** | range [-0.924, 7.531], 0 missing |
| `pupil_diam_R` | float | **model input** | range [-0.916, 6.650], 0 missing |
| `eye_open_L` | float | **model input** | range [0.000, 1.000], 0 missing |
| `eye_open_R` | float | **model input** | range [0.000, 1.000], 0 missing |
| `gaze_dir_world_X` | float | **model input** | range [-1.000, 1.000], 0 missing |
| `gaze_dir_world_Y` | float | **model input** | range [-1.000, 0.999], 0 missing |
| `gaze_dir_world_Z` | float | **model input** | range [-1.000, 1.000], 0 missing |
| `gaze_origin_world_X` | float | **model input** | range [-198.412, 779.936], 0 missing |
| `gaze_origin_world_Y` | float | **model input** | range [-11.537, 49.917], 0 missing |
| `gaze_origin_world_Z` | float | **model input** | range [-93.309, 829.970], 0 missing |
| `head_quat_X` | float | **model input** | range [-0.885, 0.989], 0 missing |
| `head_quat_Y` | float | **model input** | range [-1.000, 1.000], 0 missing |
| `head_quat_Z` | float | **model input** | range [-0.748, 0.795], 0 missing |
| `head_quat_W` | float | **model input** | range [-1.000, 1.000], 0 missing |
| `head_velocity` | float | not used | Present in the file but not among the 14 shared channels; excluded from all models. |
| `participant_id` | object | identifier | Anonymized code assigned by the original data collectors. |
| `condition` | object | grouping | Experimental condition. Sessions are defined by (`global_participant_id`, `condition`); windows never cross a session boundary. |
| `source_dataset` | object | grouping | One of `maze`, `simulations`, `terrain`. |
| `global_participant_id` | object | identifier | `source_dataset` + `participant_id`, e.g. `maze_1`, `terrain_101`. Required because the same numeric code denotes different people in different sources. |
| `elevated` | bool | convenience | Row-level `fms > 2`. Models use the window-mean FMS thresholded at 2, not this column. |

## The 14 model input channels

In this exact order:

1. `pupil_diam_L`
2. `pupil_diam_R`
3. `eye_open_L`
4. `eye_open_R`
5. `gaze_dir_world_X`
6. `gaze_dir_world_Y`
7. `gaze_dir_world_Z`
8. `gaze_origin_world_X`
9. `gaze_origin_world_Y`
10. `gaze_origin_world_Z`
11. `head_quat_X`
12. `head_quat_Y`
13. `head_quat_Z`
14. `head_quat_W`

## Reconstruction checks

A correctly rebuilt table satisfies all of the following:

- 100,367 rows and 84 unique `global_participant_id` values
- windowing at 30 s / 15 s stride yields exactly 6,442 windows
- 1,256 of those windows are elevated (19.50%)
- gaze direction vectors are unit length; the raw Terrain vectors have mean
  length 1.906 (SD 0.190) before normalization, and normalizing changes
  76.8% of Terrain rows
- FMS spans [0.0, 10.0]
- 438 rows (0.44%) carry a negative pupil-diameter value, all between -1 and 0,
  all in Simulations and Terrain, affecting 32 participants. These are sentinel
  -1 markers that survived one-second averaging in those two sources. They are
  present in the table used for the published results and were deliberately not
  corrected; see the Methods section of the article. A rebuild that cleans them
  will not match these artifacts bit-for-bit.

## Note on elevated-window percentages

The per-source percentages in the article's harmonized data summary are
**window-level**: the fraction of 30 s windows whose mean FMS exceeds 2. Row-level
(per-second) fractions differ -- 26.0 / 12.4 / 18.7 for maze / simulations /
terrain against 26.4 / 13.8 / 20.6 at window level -- so check which unit you are
comparing before concluding a mismatch.

Windowing, normalization and the split protocols are implemented in
`analysis/composition_robustness.py` and `analysis/retrain_camera_ready.py`;
read those rather than reimplementing from this description.
