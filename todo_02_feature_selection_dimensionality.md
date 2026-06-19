# TODO 02: Extension Task 1 - Clean-Room Stable Feature Factory

## Goal

Revisit first-place-inspired high-dimensional feature engineering from a cleaner starting point, so Extension 1 becomes a rigorous study of anonymous feature selection, representation, and feature-factory design. The goal is not an unbounded first-place reproduction; it is to test whether a clean-room pipeline can create a genuinely new signal that may also feed back into the frozen main task.

## Motivation

The current main pipeline already borrowed the first-place method family: correlation-aware reduction, SHAP-stable features, symbolic interactions, AE-style representations, and MLP signals. However, those attempts were attached to the existing Beta pipeline, especially the 40-feature Beta5 core pool and 160-dimensional structured input. A clean-room pass is useful because the old feature pool may be too path-dependent.

## Phase 0: Audit and Boundary Setting

- [ ] Freeze the current best main-task artifacts as the comparison baseline.
- [ ] Record which previous artifacts are allowed as references and which are not allowed as starting inputs.
- [ ] Document metric/version inconsistencies found in old late-stage artifacts, especially standalone signal metrics versus blend metrics.
- [ ] Define an outer holdout that is never used for feature selection, early stopping, architecture selection, or blend-weight selection.
- [ ] Define inner folds for feature ranking, residual scoring, and early stopping.

## Phase 1: Raw Feature Bank

- [ ] Start from all valid raw numeric features, not from the Beta5 40-feature pool.
- [ ] Compute train-only Pearson, Spearman, missing-rate, variance, and correlation-cluster statistics.
- [ ] Build correlation clusters using `1 - abs(corr)` or an equivalent redundancy distance.
- [ ] Select a stable raw feature bank by combining target correlation, fold appearance, and redundancy control.
- [ ] Save a manifest explaining why each feature entered the bank.

## Phase 2: Symbolic Interaction Factory

- [ ] Generate pairwise symbolic candidates from the clean raw feature bank.
- [ ] Include the existing simple operators as a baseline: add, subtract, multiply, min, max, and safe ratio.
- [ ] Add only a small number of extra operators if they are justified and numerically stable.
- [ ] Score interactions inside inner folds using label correlation, rank correlation, and current residual correlation.
- [ ] Prune highly redundant interactions before model fitting.
- [ ] Compare clean-room interaction sets against the old Beta5 selected interactions.

## Phase 3: Feature Recycling and Residual Signal

- [ ] Train a clean Ridge backbone on the selected raw/interaction bank using inner-fold choices only.
- [ ] Generate out-of-fold residuals for feature recycling.
- [ ] Re-score raw and interaction candidates against residuals without touching the outer holdout.
- [ ] Test whether residual-selected features provide a new signal or only overfit local validation.

## Phase 4: Clean Representation Signals

- [ ] Train AE or supervised AE only on the clean-room feature bank.
- [ ] Ensure AE/MLP early stopping uses an inner validation split, not the final outer holdout.
- [ ] Compare Ridge-on-latent, Ridge-on-input-plus-latent, and MLP signal generator variants.
- [ ] Track seed stability before considering any leaderboard probe.

## Phase 5: Main-Task Feedback Gate

- [ ] Compare the clean-room signal with the frozen current best on outer holdout and rolling folds.
- [ ] Measure prediction correlation with the frozen current best.
- [ ] Test conservative blend weights only, such as `0.025`, `0.05`, `0.075`, and `0.10`.
- [ ] Submit at most one or two candidates only if local evidence is materially different from old-pipeline tuning.
- [ ] Close the branch if improvements are only public/holdout traps.

## Required Outputs

- [ ] `runs/02_feature_cleanroom/feature_bank_manifest.csv`
- [ ] `runs/02_feature_cleanroom/clean_feature_metrics.csv`
- [ ] `runs/02_feature_cleanroom/interaction_candidate_scores.csv`
- [ ] `runs/02_feature_cleanroom/selected_interaction_definitions.csv`
- [ ] `runs/02_feature_cleanroom/outer_holdout_metrics.csv`
- [ ] `runs/02_feature_cleanroom/signal_correlation_summary.csv`
- [ ] Feature-bank funnel plot.
- [ ] Clean-room versus old-pipeline comparison table.
- [ ] Short report section explaining whether the first-place-inspired ideas failed because of the idea itself or because of old-pipeline path dependence.

## Acceptance Criteria

- [ ] All feature selection and representation choices are made without using the final outer holdout.
- [ ] At least three feature construction schemes are compared: raw stable bank, symbolic interactions, and representation features.
- [ ] The clean-room branch provides either a new useful signal or a defensible negative result.
- [ ] The report can explain the tradeoff between dimension reduction, interaction expansion, leakage control, and transferability.
- [ ] The branch has a clear stop point and does not become open-ended leaderboard chasing.
