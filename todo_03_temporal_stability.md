# TODO 03: Extension Task 2 - Temporal Stability and Market State Analysis

## Goal

Analyze whether model performance is stable across time and whether performance changes indicate market state shifts or distribution drift.

## Checkpoints

- [x] Define time-aware validation protocol
  - [ ] Confirm whether row order represents chronological order from official metadata.
  - [x] Check whether an official time column is available in the provided parquet files.
  - [x] Avoid random K-fold as the main validation method.
  - [x] Keep the 80/20 time split as the primary validation baseline.
  - [x] Add expanding rolling validation as a stability check.

- [x] Rolling validation
  - [x] Fold 1: train early period, validate next period.
  - [x] Fold 2: expand training window, validate next period.
  - [x] Fold 3: expand training window, validate next period.
  - [x] Fold 4: expand training window, validate next period.
  - [x] Save fold-level Pearson, RMSE, target std, and prediction std.

- [ ] Optional embargo experiment
  - [ ] Add a gap between training and validation windows.
  - [ ] Compare results with and without embargo.
  - [ ] Use this to discuss look-ahead bias risk.

- [x] Time-segment performance analysis
  - [x] Split validation predictions into chronological rolling blocks.
  - [x] Compute Pearson per block.
  - [x] Plot rolling or block-wise Pearson.
  - [x] Identify high-performance and low-performance periods.

- [ ] Distribution drift analysis
  - [x] Compare target mean, std, and quantiles by time segment.
  - [x] Compare prediction mean, std, and quantiles by time segment.
  - [ ] Track missing ratio or feature distribution changes over time.
  - [x] Save drift summary tables.

- [ ] Feature stability analysis
  - [x] Train LightGBM on different rolling folds.
  - [x] Compare top feature importance across folds.
  - [x] Compute overlap ratio of top-k features.
  - [ ] Plot feature importance changes for selected features.

## Required Outputs

- [x] `metrics_rolling_validation.csv`
- [x] Rolling Pearson line plot
- [x] Target distribution by time segment
- [x] Prediction distribution by time segment
- [x] Feature importance stability table
- [x] Market-state discussion notes

## Acceptance Criteria

- [x] At least four chronological validation folds are evaluated.
- [x] The report can clearly state whether performance is stable or regime-dependent.
- [x] The analysis explains score variation using target, prediction, or feature distribution evidence.
- [x] The validation design avoids random leakage-prone splits as the main evidence.

## Current Artifacts

- Config: `configs/03_temporal_rolling_validation.yaml`
- Script: `scripts/run_rolling_validation.py`
- Metrics: `runs/03_temporal/rolling_validation/metrics_rolling_validation.csv`
- Stability summary: `runs/03_temporal/rolling_validation/scheme_stability_summary.csv`
- Feature stability:
  - `runs/03_temporal/rolling_validation/feature_overlap.csv`
  - `runs/03_temporal/rolling_validation/feature_selection_frequency.csv`
  - `runs/03_temporal/rolling_validation/lightgbm_importance_stability.csv`
- Figures:
  - `reports/figures/03_temporal/rolling_pearson_by_scheme.png`
  - `reports/figures/03_temporal/target_distribution_by_fold.png`
