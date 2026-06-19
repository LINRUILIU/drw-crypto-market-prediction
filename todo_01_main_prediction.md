# TODO 01: Main Task - Frozen Crypto Short-Term Return Prediction

## Goal

Keep the DRW Crypto Market Prediction main task frozen around the selected Kaggle submission, and use future work only when an extension produces a clearly new, leakage-controlled signal.

## Current Selected Result

- Selected submission formula:
  - `0.925 * beta6_2_current_best + 0.075 * wide_adamw_lr001_seed2026`
- Selected leaderboard score:
  - public Pearson: `0.06553`
  - private Pearson: `0.10550`
- Main-task status:
  - Frozen after Sprint-A and Sprint-B failed to beat private `0.10550`.

## Completed Main-Task Scope

- [x] Confirm data files and schema.
- [x] Build reproducible loading, preprocessing, validation, metric, and submission pipeline.
- [x] Train Ridge, LightGBM, XGBoost, CatBoost, and simple ensemble baselines.
- [x] Build Pearson/Spearman top-k Ridge backbone.
- [x] Add rolling-aware validation and stability checks.
- [x] Add correlation medoid and SHAP-stable XGBoost signal.
- [x] Add symbolic interaction Ridge signal.
- [x] Add supervised AE and supervised MLP low-weight signals.
- [x] Run bounded Sprint-A interaction-stability check.
- [x] Run bounded Sprint-B MLP seed-stability check.
- [x] Freeze current best after no sprint improved private `0.10550`.
- [x] Draft main-task report narrative and figures.

## Remaining Main-Task Cleanup

- [ ] Lock the selected submission path and formula in the final report.
- [ ] Add or reference the Kaggle score screenshot if required by the course submission.
- [ ] Confirm final report and PPT use the frozen score ladder, final pipeline, and signal-family figures.
- [ ] Keep public/private divergence discussion tied to concrete submitted candidates.
- [ ] Do not run more leaderboard probes from the old pipeline unless Extension 1 passes the reopen gate below.

## Reopen Gate

Only reopen the main Kaggle task if the clean-room Extension 1 pipeline produces all of the following:

- [ ] A new validation signal generated without using the old Beta5 40-feature pool as its starting point.
- [ ] Outer-holdout or rolling improvement under a validation protocol where the final holdout is not used for feature selection, early stopping, or blend-weight selection.
- [ ] Submission/test prediction correlation with current best low enough to be genuinely complementary.
- [ ] A conservative blend candidate that improves local evidence without requiring a large signal weight.
- [ ] At most one or two Kaggle probes, with clear stop criteria before submission.

## Required Outputs

- [x] `metrics_main_models.csv`
- [x] selected submission CSV
- [x] score ladder figure
- [x] final pipeline figure
- [x] feature funnel figure
- [x] public/private divergence figure
- [ ] Kaggle score screenshot or documented screenshot reference
- [ ] final report section aligned with the frozen main-task state

## Acceptance Criteria

- [x] At least one valid Kaggle submission is produced.
- [x] The selected submission and formula are reproducible from logged artifacts.
- [x] The final report can explain why the selected model was frozen.
- [x] The report does not imply that public leaderboard or optimistic holdout was the final selection criterion.
- [ ] Any future main-task change is justified by Extension 1 clean-room evidence, not by old-pipeline weight tuning.
