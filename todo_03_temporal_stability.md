# TODO 03: Extension Task 2 - Temporal Stability and Distribution Drift

## Goal

Use chronological validation and distribution analysis to support model credibility. This extension should explain why the task is not an i.i.d. regression problem and why the final model selection avoided random K-fold and public-score chasing.

## Completed Scope

- [x] Define a time-aware validation protocol.
- [x] Keep the chronological 80/20 split as the primary baseline.
- [x] Add expanding rolling validation.
- [x] Evaluate four chronological validation folds.
- [x] Save fold-level Pearson, RMSE, target statistics, and prediction statistics.
- [x] Plot rolling Pearson by feature/model scheme.
- [x] Plot target distribution by time segment.
- [x] Compare feature-selection stability across folds.
- [x] Save LightGBM feature-importance stability outputs.
- [x] Draft market-state and validation-stability notes.

## Remaining Cleanup

- [x] Confirm report wording says "time segment" or "regime-like shift" rather than overclaiming exact market states.
- [x] Add a concise table of rolling mean, standard deviation, minimum, and maximum Pearson by scheme.
- [x] Add one paragraph explaining why top rolling mean alone was not enough for final model selection.
- [x] Add a short leakage-control note: random splits are not used as main evidence.
- [x] Add a lightweight feature-drift summary for top stable raw features.
- [x] Add an embargo comparison as a robustness appendix, not as a required main result.
- [x] Add public/private divergence explanation tied to submitted candidates.
- [x] Add final leaderboard CSV context and distinguish official leaderboard rows from post-competition Beta7 final resubmission.

## Required Outputs

- [x] `runs/03_temporal/rolling_validation/metrics_rolling_validation.csv`
- [x] `runs/03_temporal/rolling_validation/scheme_stability_summary.csv`
- [x] `runs/03_temporal/rolling_validation/target_distribution_by_fold.csv`
- [x] `runs/03_temporal/rolling_validation/feature_overlap.csv`
- [x] `runs/03_temporal/rolling_validation/feature_selection_frequency.csv`
- [x] `runs/03_temporal/rolling_validation/lightgbm_importance_stability.csv`
- [x] `reports/figures/03_temporal/rolling_pearson_by_scheme.png`
- [x] `reports/figures/03_temporal/target_distribution_by_fold.png`
- [x] `runs/03_temporal/report_artifacts/rolling_stability_report_table.csv`
- [x] `runs/03_temporal/report_artifacts/target_drift_report_table.csv`
- [x] `runs/03_temporal/report_artifacts/feature_overlap_report_table.csv`
- [x] `runs/03_temporal/report_artifacts/stable_feature_frequency_table.csv`
- [x] `runs/03_temporal/report_artifacts/feature_drift_summary.csv`
- [x] `runs/03_temporal/report_artifacts/embargo_comparison.csv`
- [x] `runs/03_temporal/report_artifacts/public_private_submission_table.csv`
- [x] `runs/03_temporal/report_artifacts/public_private_gap_summary.csv`
- [x] `runs/03_temporal/report_artifacts/leaderboard_public_private_summary.csv`
- [x] `runs/03_temporal/report_artifacts/leaderboard_private_desc_clean.csv`
- [x] `runs/03_temporal/report_artifacts/leaderboard_public_desc_clean.csv`
- [x] `reports/figures/03_temporal/feature_overlap_summary.png`
- [x] `reports/figures/03_temporal/stable_feature_drift_heatmap.png`
- [x] `reports/figures/03_temporal/embargo_comparison.png`
- [x] `reports/figures/03_temporal/public_private_gap_bar.png`
- [x] `reports/figures/03_temporal/leaderboard_score_scatter.png`
- [x] `reports/figures/03_temporal/leaderboard_rank_scatter.png`
- [x] Report-ready rolling-stability summary table.
- [x] Report paragraph connecting temporal instability to conservative low-weight ensembling.

## Acceptance Criteria

- [x] At least four chronological validation folds are evaluated.
- [x] The report can state whether performance is stable or regime-dependent.
- [x] The analysis explains score variation using target, prediction, or feature-distribution evidence.
- [x] The validation design avoids random leakage-prone splits as the main evidence.
- [x] Final wording avoids claiming access to real timestamped market regimes if the data only supports row-order time segments.
- [x] Post-competition submission scores are used as best observed scores, not as official leaderboard ranks.
