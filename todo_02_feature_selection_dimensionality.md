# TODO 02: Extension Task 1 - Feature Selection and Dimensionality Reduction

## Goal

Study how to build lower-dimensional and more stable predictive features under high-dimensional anonymous feature settings.

## Checkpoints

- [ ] Establish the full-feature baseline
  - [ ] Reuse the main-task validation split.
  - [ ] Train Ridge and LightGBM using all valid features.
  - [ ] Save full-feature Pearson, RMSE, training time, and feature count.

- [ ] Pearson correlation feature selection
  - [ ] Compute feature-target correlation using training split only.
  - [ ] Test top 50, top 100, top 200, and top 300 features.
  - [ ] Train Ridge and LightGBM for each feature set.
  - [ ] Save selected feature lists and metrics.

- [ ] Low-variance and missing-rate filtering
  - [ ] Measure missing ratio per feature.
  - [ ] Measure variance per feature.
  - [ ] Try removing near-constant or high-missing features.
  - [ ] Compare with full-feature baseline.

- [ ] PCA dimensionality reduction
  - [ ] Fit scaler on training split only.
  - [ ] Fit PCA on training split only.
  - [ ] Test 50, 100, 200, and 300 PCA components.
  - [ ] Train Ridge and LightGBM or Ridge-only if tree models become unsuitable.
  - [ ] Record explained variance ratio and validation Pearson.

- [ ] Correlation clustering
  - [ ] Compute feature correlation matrix on training split only.
  - [ ] Define distance as `1 - abs(corr)`.
  - [ ] Run hierarchical clustering.
  - [ ] For each cluster, select the feature with highest train-only target correlation.
  - [ ] For each cluster, also test cluster mean features.
  - [ ] Compare representative-feature and cluster-mean variants.

- [ ] Model-based feature importance
  - [ ] Extract LightGBM importance.
  - [ ] Test importance top 100, top 200, and top 300.
  - [ ] Run permutation importance on a validation sample if runtime allows.
  - [ ] Run SHAP on a sample if runtime allows.

## Required Outputs

- [ ] `metrics_feature_selection.csv`
- [ ] Feature count vs Pearson plot
- [ ] Feature count vs training time plot
- [ ] PCA explained variance plot
- [ ] Correlation clustering summary table
- [ ] Top feature importance plot

## Acceptance Criteria

- [ ] At least three feature schemes are compared against full features.
- [ ] The report can discuss the tradeoff between dimension reduction, speed, and Pearson score.
- [ ] All feature selection methods are fitted on training data only.
- [ ] At least one method provides a clear modeling insight, even if it does not beat the full-feature model.

