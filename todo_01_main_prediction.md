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
  - [x] Submit to Kaggle.
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
- [x] Blend the two successful Beta3 signals, `residual_pearson_top100` and `spearman_top50`, with the current best.
- [x] Confirmed direct residual + Spearman blends do not beat Spearman-only on private; fine-tuned Spearman weight to a `0.225-0.235` plateau with best private `0.10003`.
- [x] Study high-ranking solutions for new hypotheses after exhausting the residual/Spearman blend axis.
- [x] Implemented Beta4-A correlation-cluster medoid features; direct medoid Ridge blend did not transfer, private `0.09565`.
- [x] Implemented Beta4-B purged XGB TreeSHAP stable features; current best is `0.75 * beta3_current_best + 0.25 * shap_stable_xgboost` with private `0.10043`.
- [x] Implemented Beta4-C CPU MLP baseline with SGD and mixed MSE/Pearson loss.
- [x] Submitted Beta4-C `0.85 * current_best + 0.15 * hybrid_mlp`; public `0.05959`, private `0.09729`, not selected.
- [x] Implemented Beta4.1 SHAP-stable XGB refinement: feature-rule sweep, XGB training sweep, and blend-weight sweep.
- [x] Submitted Beta4.1 `top30_min3_fill20` probes at signal weights `0.325` and `0.25`; private scores were `0.09956` and `0.09974`, below the current best.
- [x] Kept Beta4-B as the current best: `0.75 * beta3_current_best + 0.25 * shap_stable_xgboost`, private `0.10043`.
- [x] Implemented Beta5-A interaction-first feature expansion with 5,460 pairwise symbolic candidates and 120 selected interaction features.
- [x] Submitted Beta5-A XGB interaction probe; public `0.05657`, private `0.10044`, effectively tied with the previous best.
- [x] Submitted Beta5-A Ridge interaction-only blend; current best is `0.85 * beta4_current_best + 0.15 * ridge_interactions_only`, public `0.06362`, private `0.10302`.
- [x] Implemented Beta5-B interaction Ridge refinement over interaction counts `60-240`, high-alpha Ridge variants, and blend weights `0.10-0.20`.
- [x] Submitted Beta5-B `int240_alpha300000_w0.20`; public `0.06729`, private `0.10214`, below the Beta5-A best.
- [x] Submitted Beta5-B `int240_alpha100000_w0.20`; public `0.06797`, private `0.10256`, below the Beta5-A best.
- [x] Kept Beta5-A as the current best: `0.85 * beta4_current_best + 0.15 * ridge_interactions_only`, private `0.10302`.
- [x] Closed Beta5 with Beta5-A as the selected main-task best; do not continue XGB interactions, int240 widening, or heavier interaction weights without a new hypothesis.
- [x] Implemented Beta6-AE using 40 core features plus 120 selected Beta5-A interaction features and an 8-dimensional bottleneck.
- [x] Submitted Beta6-AE `0.85 * current_best + 0.15 * ae_ridge`; public `0.06488`, private `0.10169`, below the Beta5-A best.
- [x] Submitted Beta6-AE `0.90 * current_best + 0.10 * ae_ridge`; public `0.06454`, private `0.10225`, below the Beta5-A best.
- [x] Kept Beta5-A as the current best after Beta6-AE: private `0.10302`.
- [x] Tested Beta6.1 AE refinement with bottleneck 8/16/32, light denoising, and a seed probe; conservative AE blend scored public `0.06412`, private `0.10269`, below the Beta5-A best.
- [x] Implemented Beta6.2 supervised AE with an auxiliary target head; best conservative blend scored public `0.06454`, private `0.10303`, narrowly above the Beta5-A best.
- [x] Implemented Beta7 supervised MLP signal generator over fixed 160 Beta5 structured inputs with six MLPs, seed-mean ensembles, and low-weight blends.
- [x] Submitted Beta7 conservative AdamW seed-mean blend; score was public `0.06588`, private `0.10411`.
- [x] Submitted Beta7 `0.075` holdout-best probe; current best is `0.925 * beta6_2_current_best + 0.075 * wide_adamw_lr001_seed2026`, public `0.06553`, private `0.10550`.
- [x] Submitted Beta7 `0.10` low-correlation probe; public `0.06930`, private `0.10304`, not selected.
- [x] Ran Beta7.1 MLP weight calibration at `0.05`, `0.0625`, and `0.0875`; `0.0875` tied private `0.10550` but had lower public, so `0.075` remains selected.
- [x] Revisited the first-place solution gap and documented the sprint decision in `docs/main_task_retrospective.md`.
- [x] Sprint-A fold-stable interaction selection completed; best submitted candidate scored public `0.06637`, private `0.10514`, below current best.
- [x] Sprint-B AdamW MLP seed-stability check completed; best submitted candidate scored public `0.06573`, private `0.10376`, below current best.
- [x] Freeze the main task for report consolidation because Sprint-A and Sprint-B did not beat private `0.10550`.
- [x] Drafted the main-task report outline in `docs/main_report_outline.md`.
- [x] Generated main-task report figures under `reports/figures/01_main/`.
- [x] Polished `reports/report-draft.md`, fixed formulas, inserted figures, and aligned wording with the development log.


