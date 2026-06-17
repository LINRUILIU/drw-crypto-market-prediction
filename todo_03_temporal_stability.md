# TODO 03: Extension Task 2 - Temporal Stability and Market State Analysis

## Goal

Analyze whether model performance is stable across time and whether performance changes indicate market state shifts or distribution drift.

## Checkpoints

- [ ] Define time-aware validation protocol
  - [ ] Confirm whether row order represents chronological order.
  - [ ] Use the official time column if available.
  - [ ] Avoid random K-fold as the main validation method.
  - [ ] Keep the 80/20 time split as the primary validation baseline.

- [ ] Rolling validation
  - [ ] Fold 1: train early period, validate next period.
  - [ ] Fold 2: expand training window, validate next period.
  - [ ] Fold 3: expand training window, validate next period.
  - [ ] Fold 4: expand training window, validate next period.
  - [ ] Save fold-level Pearson, RMSE, target std, and prediction std.

- [ ] Optional embargo experiment
  - [ ] Add a gap between training and validation windows.
  - [ ] Compare results with and without embargo.
  - [ ] Use this to discuss look-ahead bias risk.

- [ ] Time-segment performance analysis
  - [ ] Split validation predictions into equal time blocks.
  - [ ] Compute Pearson per block.
  - [ ] Plot rolling or block-wise Pearson.
  - [ ] Identify high-performance and low-performance periods.

- [ ] Distribution drift analysis
  - [ ] Compare target mean, std, skewness, and quantiles by time segment.
  - [ ] Compare prediction mean, std, and quantiles by time segment.
  - [ ] Track missing ratio or feature distribution changes over time.
  - [ ] Save drift summary tables.

- [ ] Feature stability analysis
  - [ ] Train LightGBM on different rolling folds.
  - [ ] Compare top feature importance across folds.
  - [ ] Compute overlap ratio of top-k features.
  - [ ] Plot feature importance changes for selected features.

## Required Outputs

- [ ] `metrics_rolling_validation.csv`
- [ ] Rolling Pearson line plot
- [ ] Target distribution by time segment
- [ ] Prediction distribution by time segment
- [ ] Feature importance stability table
- [ ] Market-state discussion notes

## Acceptance Criteria

- [ ] At least four chronological validation folds are evaluated.
- [ ] The report can clearly state whether performance is stable or regime-dependent.
- [ ] The analysis explains score variation using target, prediction, or feature distribution evidence.
- [ ] The validation design avoids random leakage-prone splits as the main evidence.

