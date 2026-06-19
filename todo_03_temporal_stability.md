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

- [ ] Confirm report wording says "time segment" or "regime-like shift" rather than overclaiming exact market states.
- [ ] Add a concise table of rolling mean, standard deviation, minimum, and maximum Pearson by scheme.
- [ ] Add one paragraph explaining why top rolling mean alone was not enough for final model selection.
- [ ] Add a short leakage-control note: random splits are not used as main evidence.
- [ ] If time permits, add a lightweight feature-drift summary for top stable raw features.
- [ ] If time permits, add an embargo comparison as a robustness appendix, not as a required main result.

## Required Outputs

- [x] `runs/03_temporal/rolling_validation/metrics_rolling_validation.csv`
- [x] `runs/03_temporal/rolling_validation/scheme_stability_summary.csv`
- [x] `runs/03_temporal/rolling_validation/target_distribution_by_fold.csv`
- [x] `runs/03_temporal/rolling_validation/feature_overlap.csv`
- [x] `runs/03_temporal/rolling_validation/feature_selection_frequency.csv`
- [x] `runs/03_temporal/rolling_validation/lightgbm_importance_stability.csv`
- [x] `reports/figures/03_temporal/rolling_pearson_by_scheme.png`
- [x] `reports/figures/03_temporal/target_distribution_by_fold.png`
- [ ] Report-ready rolling-stability summary table.
- [ ] Report paragraph connecting temporal instability to conservative low-weight ensembling.

## Acceptance Criteria

- [x] At least four chronological validation folds are evaluated.
- [x] The report can state whether performance is stable or regime-dependent.
- [x] The analysis explains score variation using target, prediction, or feature-distribution evidence.
- [x] The validation design avoids random leakage-prone splits as the main evidence.
- [ ] Final wording avoids claiming access to real timestamped market regimes if the data only supports row-order time segments.
