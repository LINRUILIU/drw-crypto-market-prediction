# TODO 02: Extension Task 1 - High-Dimensional Feature Selection and Dimensionality Reduction

## Goal

Answer RQ2 as an independent extension task:

> In a high-dimensional anonymous feature space with strong correlation and redundancy, how can feature selection or dimensionality reduction reduce feature count while preserving predictive ability?

This extension should analyze the structure of the anonymous feature space and compare compression/selection schemes. It should support the report narrative, but it should not reopen the frozen Kaggle main-task model.

## Boundary

- The selected main-task submission remains frozen.
- This extension is not a leaderboard sprint and is not used to select a new final submission.
- No result from this extension should be described as replacing the final model.
- If a method improves local validation, write it as evidence about feature-space structure and tradeoffs, not as a new main-task branch.
- The report should explicitly state that this section explains why feature screening, dimensionality control, and conservative fusion are necessary in the main task.

Suggested report wording:

> This extension task does not participate in final Kaggle model selection. It analyzes the high-dimensional anonymous feature space and compares feature compression strategies in terms of dimensionality, redundancy, predictive preservation, and computational cost.

## Research Question

### RQ2: High-Dimensional Feature Selection and Dimensionality Reduction

When many anonymous, highly correlated, and potentially redundant features are present, how can we select or compress features to balance lower dimensionality and retained prediction performance?

Key comparison dimensions:

- Feature count after selection or compression.
- Holdout Pearson and auxiliary RMSE.
- Training/inference cost.
- Redundancy reduction.
- Interpretability of selected or compressed features.
- Stability of selected feature sets across time splits, if available.

## Methods to Compare

| Method | Role | Notes |
| --- | --- | --- |
| Full features | Raw baseline | Uses all valid numeric features. |
| Pearson top-k | Supervised correlation screening | Keep features with highest train-only absolute Pearson correlation with the target. |
| Spearman top-k | Rank-based screening | Optional supplement to Pearson; useful for monotonic but non-linear signal. |
| PCA | Unsupervised linear compression | Compress variance structure; may reduce dimension without preserving target signal. |
| Correlation clustering + representative feature | Redundancy reduction | Cluster by `1 - abs(corr)` and keep a representative/medoid feature. |
| Correlation clustering + cluster mean | Noise smoothing | Replace each cluster with the average of its members. |
| ElasticNet | Regularized supervised selection | Uses mixed L1/L2 regularization, suitable for correlated high-dimensional features. |
| LightGBM importance top-k | Model-driven screening | Optional; compare tree-driven feature selection with correlation-based methods. |
| SHAP top-k | Explainability-driven screening | Optional and sample-based only if runtime is acceptable. |

Recommended top-k values:

- `k = 50`
- `k = 100`
- `k = 200`
- `k = 300`
- `k = 500`

## Phase 0: Scope and Inputs

- [x] Confirm the frozen main-task formula and submission path are referenced, not modified.
- [x] Confirm the train/test schema and valid numeric feature list.
- [x] Define the evaluation split, preferably the same chronological 80/20 split used in the report.
- [x] Make sure all feature selection, PCA fitting, clustering, scaling, and imputation are fit on the train split only.
- [x] Decide whether this extension uses only Ridge for clean comparison or Ridge + LightGBM for model-family comparison.

## Phase 1: Feature-Space Structure Analysis

- [x] Compute missing-rate and variance summaries for all raw numeric features.
- [x] Compute feature-target Pearson correlation on the train split.
- [x] Optionally compute feature-target Spearman correlation on a sample or full train split.
- [x] Compute feature-feature correlation structure or an efficient approximation.
- [x] Summarize redundancy through correlation distribution, highly correlated pair counts, and cluster sizes.
- [x] Generate a feature-space diagnostic table for the report.

Suggested outputs:

- `runs/02_feature_dimensionality/raw_feature_summary.csv`
- `runs/02_feature_dimensionality/feature_target_correlation.csv`
- `runs/02_feature_dimensionality/feature_correlation_summary.csv`
- `reports/figures/02_feature_dimensionality/correlation_distribution.png`
- `reports/figures/02_feature_dimensionality/cluster_size_distribution.png`

## Phase 2: Selection and Compression Experiments

- [x] Train the full-feature baseline.
- [x] Train Pearson top-k Ridge models for the selected `k` values.
- [x] Optionally train Spearman top-k Ridge models for selected `k` values.
- [x] Fit PCA on train features only and evaluate PCA dimensions such as 50, 100, 200, and 300.
- [x] Build correlation clusters and evaluate representative-feature models.
- [x] Build correlation-cluster mean features and evaluate cluster-mean models.
- [x] Train ElasticNet as a regularized high-dimensional selection baseline.
- [x] Optionally evaluate LightGBM importance top-k or SHAP top-k if runtime is acceptable.

Suggested output table columns:

| Method | Feature count | Holdout Pearson | RMSE | Fit time | Notes |
| --- | ---: | ---: | ---: | ---: | --- |

Suggested outputs:

- `runs/02_feature_dimensionality/feature_selection_metrics.csv`
- `runs/02_feature_dimensionality/pca_metrics.csv`
- `runs/02_feature_dimensionality/cluster_reduction_metrics.csv`
- `runs/02_feature_dimensionality/elasticnet_metrics.csv`
- `reports/figures/02_feature_dimensionality/dimension_vs_pearson.png`
- `reports/figures/02_feature_dimensionality/method_comparison_bar.png`

## Phase 3: Tradeoff and Stability Analysis

- [x] Compare dimension reduction ratio versus Holdout Pearson.
- [x] Identify whether the best predictive method is also the most compact method.
- [x] Compare supervised methods against unsupervised compression.
- [x] Check whether PCA preserves variance but loses target-relevant signal.
- [x] Check whether cluster representatives preserve raw-feature interpretability better than PCA.
- [x] If rolling-fold artifacts are available, compare feature overlap across folds for Pearson top-k or cluster methods.
- [x] Summarize the tradeoff: highest score, best compression, best stability, and best interpretability.

Suggested outputs:

- `runs/02_feature_dimensionality/tradeoff_summary.csv`
- `runs/02_feature_dimensionality/feature_overlap_summary.csv`
- `reports/figures/02_feature_dimensionality/compression_tradeoff_frontier.png`
- `reports/figures/02_feature_dimensionality/feature_overlap_summary.png`

## Phase 4: Report Section

- [x] Add or update the report section:
  - `扩展任务 1：高维匿名特征的筛选与降维分析`
- [x] Explain why full features are not necessarily optimal in high-dimensional anonymous data.
- [x] Explain the difference between supervised screening and unsupervised compression.
- [x] Use PCA as a compression baseline, not as a guaranteed performance booster.
- [x] Use ElasticNet to motivate mixed L1/L2 regularization for correlated high-dimensional features.
- [x] Explain how this extension supports the main-task design without changing the final model.

Recommended conclusion shape:

> Feature selection and dimensionality reduction do not simply maximize the leaderboard score. Their value is to reduce redundancy, control noise, improve training efficiency, and reveal why the main task benefits from a compact and stable feature backbone.

## Required Outputs

- [x] `runs/02_feature_dimensionality/raw_feature_summary.csv`
- [x] `runs/02_feature_dimensionality/feature_selection_metrics.csv`
- [x] `runs/02_feature_dimensionality/pca_metrics.csv`
- [x] `runs/02_feature_dimensionality/cluster_reduction_metrics.csv`
- [x] `runs/02_feature_dimensionality/tradeoff_summary.csv`
- [x] `reports/figures/02_feature_dimensionality/dimension_vs_pearson.png`
- [x] `reports/figures/02_feature_dimensionality/compression_tradeoff_frontier.png`
- [x] Report section for Extension Task 1.

## Acceptance Criteria

- [x] At least four schemes are compared: full features, Pearson top-k, PCA, and correlation-cluster reduction.
- [x] All fitting and feature selection are train-only with no validation leakage.
- [x] The extension reports both feature count reduction and predictive performance.
- [x] The report discusses why PCA or clustering may reduce dimension without necessarily improving Pearson.
- [x] The final writeup does not imply that Extension Task 1 changes the frozen main-task submission.

