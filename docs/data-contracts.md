# Data contracts and validation boundaries

The project keeps historical model configurations and results. These checks
improve input/output reliability; they do not establish a new benchmark score.

## Submission output

All eight historical runners that previously had local `make_submission`
implementations now call the shared helper in `drw_crypto.pipeline`, as later
runners already did.

- Predictions must be a one-dimensional array of finite real numbers, with one
  value per test row. Template row counts and prediction column names are checked.
- When test and template both have IDs, they must be unique, non-missing and
  identical in the same order. Misordered or different IDs raise `ValueError`
  before CSV output; predictions are not silently reordered.
- Known identifier names are detected even when an ID only exists in the test
  data. Custom shared identifiers can be supplied through `id_col`.
- The supplied DRW test Parquet has **no ID column**. This benchmark path keeps
  positional alignment with the original test row order. Template IDs are checked
  for uniqueness and missing values, but their correspondence to ID-free test
  rows cannot be verified. A shuffled template cannot be detected from counts
  alone; retain the original test and template order.
- Prediction-only templates and ID-free outputs remain supported. Template
  metadata and input DataFrames are not modified in place.

For example, test IDs `[10, 20]`, template IDs `[20, 10]` and predictions
`[0.1, 0.2]` previously produced a valid-looking but mislabelled CSV. They now fail
explicitly. An aligned input retains its original prediction values.

## Preprocessing

Imputation, mean and scale are fitted only on the supplied training partition;
`transform` reuses them. Infinities become missing values, wholly missing columns
use zero, and constant columns use unit scale. Empty training partitions,
duplicate feature definitions and non-finite float32 results fail explicitly.
If a new fit fails, the previous fitted statistics remain intact.

## Temporal evaluation

The historical `purged_group_time_series_splits` function is a **two-sided group
diagnostic**. Groups on both sides of validation can enter training, after
neighboring groups are removed. Its old behavior and results are preserved; do
not describe it as forward-only forecast evaluation.

The rolling runner uses `rolling_time_window_indices` to require
`train_start < train_end <= valid_start < valid_end`. Fractions must be finite
and within `[0, 1]`. Embargo removes rows from the training-window tail; if that
empties training, execution fails rather than silently ignoring the embargo.
The checked-in rolling configuration retains its original row indices.

Both approaches operate on **input row order**; the guard does not prove that a
provided dataset was sorted chronologically. Existing rolling validation still
selects model parameters and blend weights on each validation window, so its
reported metrics are tuning results, not a new independent test score.

## Lightweight verification

```sh
python -m pip install numpy pandas PyYAML matplotlib
python -m unittest discover -s tests -v
```

Tests cover submission rejection, preprocessing boundaries/statistic reuse,
rolling-window ordering and historical split compatibility. Synthetic CSV tests
run the actual Ridge CLI with and without IDs, reject a shuffled template, and
reject invalid rolling definitions before model training. No Kaggle download,
GPU, tree-model dependency or full experiment rerun is required.
