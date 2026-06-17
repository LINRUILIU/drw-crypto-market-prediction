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
- [ ] Final report can explain why the final submitted model was selected.
- [x] No validation preprocessing step uses validation target information.
