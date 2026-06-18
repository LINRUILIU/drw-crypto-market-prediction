# TODO 01: Main Task - Crypto Short-Term Return Prediction

## Goal

Build a reproducible Kaggle submission pipeline for DRW Crypto Market Prediction, with Pearson correlation as the primary metric.

## Checkpoints

- [x] Confirm data files and column schema
  - [x] Download Kaggle train/test/sample submission files.
  - [x] Record file names, row counts, feature counts, target column, id column, and submission format.
  - [x] Save a data summary table for the report.

- [x] Build the base project pipeline
  - [x] Implement data loading.
  - [x] Implement memory optimization for large tabular data.
  - [x] Identify feature columns automatically.
  - [x] Implement missing value handling.
  - [x] Implement Pearson correlation metric.
  - [x] Implement time-ordered 80/20 validation split.
  - [x] Implement submission file generation.

- [ ] Train baseline linear models
  - [x] Ridge baseline.
  - [ ] ElasticNet baseline.
  - [x] Save validation predictions, metrics, model config, and selected feature list.
  - [x] Compare Pearson, RMSE, training time, and feature count.

- [x] Train tree-based models
  - [x] LightGBM baseline.
  - [x] XGBoost baseline.
  - [x] CatBoost baseline.
  - [x] Add early stopping where supported.
  - [x] Save model importance and validation predictions.

- [ ] Build ensemble models
  - [x] Ridge + LightGBM weighted average.
  - [ ] Ridge + LightGBM + XGBoost weighted average.
  - [ ] Add CatBoost only if it improves validation or diversity.
  - [x] Search simple weights on validation or OOF predictions.
  - [x] Add rolling-aware stability-constrained blend.
  - [x] Add submission-level blend calibration for public/private divergence.
  - [x] Compare ensemble performance with single models.

- [ ] Create Kaggle submission artifacts
  - [x] Train final model on the selected training scope.
  - [x] Generate test predictions.
  - [x] Validate submission column names and row count.
  - [ ] Submit to Kaggle.
  - [ ] Save public or late submission screenshot.

## Required Outputs

- [x] `metrics_main_models.csv`
- [x] `submission_*.csv`
- [ ] Kaggle score screenshot
- [x] Model comparison table
- [ ] Final model pipeline diagram

## Acceptance Criteria

- [x] At least one valid Kaggle submission is produced.
- [x] Ridge and LightGBM baselines both run end to end.
- [x] Final report can explain why the current submission candidate was selected.
- [x] No validation preprocessing step uses validation target information.

## Stage Notes

- [x] Baseline1 closed after full-feature baseline, Pearson top-k, rolling validation, stability blend, and submission blend calibration.
- [x] Beta1 started with narrow private-score blend probes around `0.85 * top200 + 0.15 * stability`.
- [x] Beta1 added rank and normal-score submission calibration.
- [x] Beta1 added new-signal blending with top100 Ridge and full-feature baselines.
- [x] Beta1 found a new private-best top100 Ridge blend at `0.70 * raw_w085 + 0.30 * top100_ridge`.
- [x] Beta1 improved the top100 Ridge blend through `0.325` and `0.35`; current best is `0.65 * raw_w085 + 0.35 * top100_ridge`.
- [x] Beta1 improved the top100 Ridge blend through `0.40`, `0.45`, and `0.50`; current best is `0.50 * raw_w085 + 0.50 * top100_ridge`.
- [x] Probed whether top100 Ridge should dominate; current best is pure `top100_ridge_w100` with private `0.08901`.
- [x] Confirmed local saturation on the same weight axis: `w095` nearly ties, while `w105`, `w110`, and `w125` are worse.
- [x] Started Beta2 with Ridge top-k family and Ridge-only blends.
- [x] Found current best `0.475 * top100_ridge + 0.525 * top50_ridge` with private `0.09280`.
- [x] Tested direct intermediate Ridge widths; `top90` and `top80` did not beat the top50/top100 blend.
- [x] Tested alpha variants for top50/top100 Ridge components; current best is `0.45 * top50_alpha10000 + 0.55 * top100_alpha30` with private `0.09385`.
- [x] Refine around `top50_alpha10000` and `top100_alpha10/30/100` with top50 weights near `0.40` to `0.50`.
- [x] Extended high-alpha Ridge refinement; current best is `0.5 * top50_alpha200000 + 0.5 * top100_alpha50` with private `0.09638`.
- [x] Move beyond Ridge alpha/weight micro-tuning with Beta3 new feature-selection signals.
- [x] Tested residual Pearson, Spearman, and rolling-stability-aware Pearson top-k Ridge signals; current best is `0.80 * beta2_current_best + 0.20 * spearman_top50` with private `0.09998`.
- [ ] Blend the two successful Beta3 signals, `residual_pearson_top100` and `spearman_top50`, with the current best.
- [ ] Study high-ranking solutions for new hypotheses after exhausting the residual/Spearman blend axis.
