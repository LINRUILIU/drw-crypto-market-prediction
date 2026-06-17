# Development Log

## 2026-06-17: Project Setup and Main Baseline

### Planning

- Confirmed project direction: DRW Crypto Market Prediction as the main task, with three extension tasks:
  - high-dimensional feature selection and dimensionality reduction;
  - temporal stability and market state analysis;
  - predictive signal interpretation.
- Created four checkpoint files:
  - `todo_01_main_prediction.md`
  - `todo_02_feature_selection_dimensionality.md`
  - `todo_03_temporal_stability.md`
  - `todo_04_signal_interpretation.md`

### Directory Structure

Adopted a shared-code layout instead of creating four independent task directories.

Current convention:

```text
src/drw_crypto/              Shared reusable code
scripts/                     Command-line entry points
configs/01_*                 Main-task experiment configs
runs/01_main/                Main-task experiment outputs
runs/02_feature/             Feature-selection outputs, to be added
runs/03_temporal/            Temporal-stability outputs, to be added
runs/04_signal/              Signal-interpretation outputs, to be added
submissions/01_main/         Kaggle submission files
docs/                        Environment notes, audit notes, result logs
reports/figures/             Stable report and PPT figures
```

Reasoning: the four tasks share the same raw data, preprocessing, metric, model wrappers, validation utilities, and artifact conventions. Keeping one shared `src/` avoids duplicated code and inconsistent behavior across tasks.

### Environment

- Standardized on the shared environment:

```text
D:\PythonEnvs\envs\py313-science
```

- Added missing packages to that environment:
  - `pyarrow`
  - `PyYAML`
  - `xgboost`
  - `catboost`

- Updated `README.md` and `docs/environment.md` to use this environment.
- Verified the project scripts run under `py313-science`.
- Avoided using system Python for project dependencies.
- Did not create an in-project `.venv`.

### Data Schema

Confirmed Kaggle files in `data/raw/`:

| File | Shape | Notes |
| --- | ---: | --- |
| `train.parquet` | 525,886 x 786 | target column is `label` |
| `test.parquet` | 538,150 x 786 | `label` is all-zero placeholder |
| `sample_submission.csv` | 538,150 x 2 | columns are `ID`, `prediction` |

Feature count used by the baseline: 785.

### Implemented Code

Reusable modules:

- `src/drw_crypto/io.py`
  - file discovery
  - table loading
  - JSON/text artifact writing
- `src/drw_crypto/metrics.py`
  - Pearson correlation
  - RMSE
- `src/drw_crypto/preprocessing.py`
  - target/id/feature inference
  - median imputation
  - standardization fitted on training split only
- `src/drw_crypto/models.py`
  - NumPy Ridge baseline
- `scripts/run_main_baseline.py`
  - train/validation split
  - Ridge alpha search
  - optional ElasticNet, LightGBM, XGBoost, CatBoost
  - weighted Ridge + LightGBM ensemble
  - submission generation

Main configs:

- `configs/01_main_baseline_smoke.yaml`
- `configs/01_main_baseline_full.yaml`

### Validation Protocol

- Primary metric: Pearson correlation.
- Split: chronological 80/20 by row order.
- Validation rows: 105,178.
- Preprocessing is fitted on the training split only.

### Baseline Results

Full-feature validation results:

| Model | Pearson | RMSE |
| --- | ---: | ---: |
| Ridge + LightGBM weighted ensemble | 0.100263 | 1.126039 |
| Mean ensemble | 0.099955 | 1.107722 |
| Ridge | 0.097166 | 1.235401 |
| LightGBM | 0.063721 | 1.079047 |
| XGBoost | 0.047563 | 1.227862 |
| CatBoost | 0.040489 | 1.106226 |

Current first submission candidate:

```text
submissions/01_main/baseline_full/submission_best.csv
```

This is the Ridge + LightGBM weighted ensemble with Ridge weight `0.60`.

### Observations

- Ridge is a surprisingly strong baseline, which is plausible for anonymized high-quality tabular production features.
- LightGBM, XGBoost, and CatBoost needed careful handling because RMSE-based early stopping is misaligned with the official Pearson objective.
- LightGBM was changed to use Pearson evaluation for early stopping.
- XGBoost and CatBoost were run with fixed iteration counts for this baseline.
- ElasticNet was not included in the full baseline because it was slow and did not converge in the 20k smoke run. It should be revisited after feature selection or dimensionality reduction.

### Current Decision Point

Recommendation: create an initial git checkpoint before further optimization.

Reason:

- The baseline pipeline is working end to end.
- The environment policy is now documented.
- The data schema and first validation results are recorded.
- The next stage will involve more experimental changes, so a clean baseline checkpoint is useful.

Current blocker:

- `D:\final-modeling` is not yet a git repository. A commit requires initializing a repository first.

Suggested initial commit scope:

- Source code under `src/`
- Scripts under `scripts/`
- Configs under `configs/`
- Docs under `docs/`
- Todo files and `README.md`
- `.gitignore` and `requirements.txt`

Do not commit:

- `data/raw/`
- `runs/`
- `submissions/`

## 2026-06-17: Git Repository Publication

### Local Versioning

- Initialized a local git repository in `D:\final-modeling`.
- Created initial baseline checkpoint:

```text
c7ec08d Initialize DRW crypto baseline pipeline
```

### Remote Repository

- Target repository: `LINRUILIU/drw-crypto-market-prediction`
- Remote URL:

```text
https://github.com/LINRUILIU/drw-crypto-market-prediction.git
```

- Intended project branch: `master`.

### Push Notes

- SSH push first failed because no usable GitHub SSH public key was configured on this machine.
- Switched `origin` to HTTPS.
- The repository metadata reported default branch `main`, so the first push was sent to `main`.
- The branch was then corrected back to `master` for this project.
- HTTPS push succeeded through the available Git credential flow.

## 2026-06-17: Pearson Top-k Main-Task Optimization

### Goal

Add a reproducible optimization line for the main Kaggle prediction task using train-only Pearson top-k feature selection with Ridge, LightGBM, and their weighted ensemble.

### Implementation

- Added config: `configs/01_main_pearson_topk.yaml`.
- Added script: `scripts/run_pearson_topk.py`.
- Added shared feature-ranking utility: `src/drw_crypto/feature_selection.py`.
- Default top-k values: `50`, `100`, `200`, `300`, `500`, plus `full` as a control.
- Feature ranking uses absolute Pearson correlation with `label`, fitted on the training split only.
- Each feature set independently fits missing-value imputation, standardization, Ridge, and LightGBM.
- LightGBM keeps Pearson-based early stopping.
- Ensemble searches Ridge weight from `0.00` to `1.00` with step `0.05`.

### Validation Results

Compared against the previous full-feature ensemble baseline Pearson `0.100263`.

| Scheme | Model | Features | Pearson | RMSE | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| top200 | Ridge + LightGBM | 200 | 0.125069 | 1.079047 | Ridge weight `0.95` |
| top200 | Ridge | 200 | 0.124899 | 1.081521 | alpha `1000.0` |
| top100 | Ridge + LightGBM | 100 | 0.114977 | 1.071005 | Ridge weight `0.95` |
| full | Ridge + LightGBM | 785 | 0.100263 | 1.126039 | Ridge weight `0.60` |

### Artifacts

- Metrics: `runs/01_main/pearson_topk/metrics_pearson_topk.csv`
- Feature ranking: `runs/01_main/pearson_topk/pearson_feature_ranking.csv`
- Best submission: `submissions/01_main/pearson_topk/submission_best.csv`
- Final summary: `runs/01_main/pearson_topk/final_summary.json`

### Outcome

- Best validation model: `top200` Ridge + LightGBM ensemble.
- Pearson improved from `0.100263` to `0.125069`.
- A new Kaggle submission file was generated because the top-k model beat the full-feature baseline.

### Kaggle Result

Submitted `submissions/01_main/pearson_topk/submission_best.csv`.

| Split | Score |
| --- | ---: |
| Public | 0.02826 |
| Private | 0.06610 |

Interpretation: the single chronological 80/20 validation split overestimated leaderboard generalization. The top-k model is useful as a feature-selection result, but the next optimization step should prioritize rolling validation and temporal stability before further leaderboard-oriented feature tuning.

## 2026-06-17: Rolling Validation and Temporal Stability

### Goal

Check whether the main-task validation signal is stable over time after the Pearson top-k submission scored materially lower on Kaggle than on the single 80/20 validation split.

### Implementation

- Added config: `configs/03_temporal_rolling_validation.yaml`.
- Added script: `scripts/run_rolling_validation.py`.
- Evaluated four expanding chronological folds:
  - `fold_50_60`: train 0-50%, validate 50-60%.
  - `fold_60_70`: train 0-60%, validate 60-70%.
  - `fold_70_80`: train 0-70%, validate 70-80%.
  - `fold_80_90`: train 0-80%, validate 80-90%.
- Recomputed train-only Pearson feature ranking for each fold.
- Tested `top50`, `top100`, `top200`, `top300`, `top500`, and `full`.
- For each fold and scheme, fitted imputation, standardization, Ridge, LightGBM, and Ridge + LightGBM ensemble independently.
- Added feature-set overlap, selection frequency, and LightGBM feature-importance stability outputs.

### Full Rolling Results

Rows below summarize Ridge + LightGBM ensemble Pearson over the four folds.

| Scheme | Mean Pearson | Std | Min | Max | Mean RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| full | 0.115468 | 0.019837 | 0.098538 | 0.147126 | 1.056822 |
| top50 | 0.121328 | 0.035865 | 0.078946 | 0.170520 | 1.032792 |
| top100 | 0.135306 | 0.058504 | 0.079048 | 0.233110 | 1.025695 |
| top500 | 0.139575 | 0.036829 | 0.090258 | 0.175179 | 1.044568 |
| top200 | 0.153862 | 0.073044 | 0.081128 | 0.272860 | 1.025452 |
| top300 | 0.162315 | 0.051951 | 0.103448 | 0.239494 | 1.023193 |

### Target Drift

| Fold | Target Mean | Target Std | q05 | Median | q95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| fold_50_60 | 0.023199 | 0.945710 | -1.305820 | 0.013257 | 1.327380 |
| fold_60_70 | 0.117703 | 1.044563 | -1.229271 | 0.059195 | 1.611012 |
| fold_70_80 | 0.054910 | 1.003248 | -1.476842 | 0.042353 | 1.649716 |
| fold_80_90 | -0.002802 | 1.029727 | -1.430521 | -0.002125 | 1.490496 |

### Artifacts

- Metrics: `runs/03_temporal/rolling_validation/metrics_rolling_validation.csv`
- Scheme summary: `runs/03_temporal/rolling_validation/scheme_stability_summary.csv`
- Target distribution: `runs/03_temporal/rolling_validation/target_distribution_by_fold.csv`
- Feature overlap: `runs/03_temporal/rolling_validation/feature_overlap.csv`
- Feature selection frequency: `runs/03_temporal/rolling_validation/feature_selection_frequency.csv`
- LightGBM importance stability: `runs/03_temporal/rolling_validation/lightgbm_importance_stability.csv`
- Figures:
  - `reports/figures/03_temporal/rolling_pearson_by_scheme.png`
  - `reports/figures/03_temporal/target_distribution_by_fold.png`

### Interpretation

- `top200` and `top300` have the strongest mean rolling Pearson, but their fold-to-fold variance is high.
- `full` has the lowest Pearson standard deviation and the smallest gap to the reported private score `0.06610`, so it is the most conservative stability reference.
- The target distribution shifts by time segment, especially in `fold_60_70`, where target mean and upper quantiles are higher. This supports treating the task as regime-dependent instead of a static i.i.d. regression problem.
- The single 80/20 top200 validation result `0.125069` should not be used alone for model selection. Future leaderboard-oriented submissions should prefer rolling-aware selection or blend candidates chosen for stability, not only peak validation Pearson.

### Next Direction

The next main-task optimization should build a robust candidate from this evidence:

- use full or top50 as stability baselines;
- compare top200/top300 only if the selection rule penalizes rolling variance;
- consider a weighted blend of stable full-feature Ridge/LightGBM with top-k Ridge signals;
- generate a new submission only after the rolling criterion is defined.

## 2026-06-17: Rolling-Aware Stability Blend

### Goal

Build a new main-task submission candidate that trades some peak validation Pearson for lower fold-to-fold variance. This is a direct response to the previous top200 submission, where single-split validation looked strong but Kaggle generalization was weaker.

### Implementation

- Added config: `configs/01_main_stability_blend.yaml`.
- Added script: `scripts/run_stability_blend.py`.
- Candidate components:
  - `full_ridge`
  - `full_lightgbm`
  - `top50_ridge`
  - `top50_lightgbm`
  - `top200_ridge`
  - `top300_ridge`
- Weight search constraints:
  - stable components (`full` and `top50`) must carry at least `0.70` total weight;
  - top-k Ridge signal components (`top200`, `top300`) can carry at most `0.30` total weight;
  - weight step is `0.05`.
- Selection objective:

```text
mean rolling Pearson - 1.0 * rolling std + 0.25 * minimum fold Pearson
```

- Added `--final-only` mode so final submission generation can reuse the saved rolling blend artifacts without rerunning all folds.

### Selected Blend

| Component | Weight |
| --- | ---: |
| full Ridge | 0.20 |
| full LightGBM | 0.25 |
| top50 Ridge | 0.05 |
| top50 LightGBM | 0.20 |
| top200 Ridge | 0.30 |
| top300 Ridge | 0.00 |

Selected final parameters:

| Scheme | Ridge alpha | LightGBM trees |
| --- | ---: | ---: |
| full | 1000.0 | 154 |
| top50 | 1000.0 | 156 |
| top200 | 1000.0 | n/a |
| top300 | 1000.0 | n/a |

### Rolling Results

| Fold | Pearson | RMSE |
| --- | ---: | ---: |
| fold_50_60 | 0.108175 | 0.964976 |
| fold_60_70 | 0.144969 | 1.052979 |
| fold_70_80 | 0.154487 | 1.010371 |
| fold_80_90 | 0.122564 | 1.077876 |

Summary:

| Metric | Value |
| --- | ---: |
| rolling mean Pearson | 0.132549 |
| rolling std | 0.018230 |
| rolling min | 0.108175 |
| rolling max | 0.154487 |
| 80/20 holdout Pearson | 0.103034 |
| 80/20 holdout RMSE | 1.081795 |

Compared with the earlier rolling check, this blend has lower mean Pearson than top200/top300, but it also has much lower variance and a stronger minimum fold.

### Artifacts

- Metrics: `runs/01_main/stability_blend/best_blend_fold_metrics.csv`
- Candidate grid: `runs/01_main/stability_blend/blend_candidates_top.csv`
- Selected weights: `runs/01_main/stability_blend/best_blend_weights.csv`
- Holdout metrics: `runs/01_main/stability_blend/holdout_stability_blend_metrics.csv`
- Final summary: `runs/01_main/stability_blend/final_summary.json`
- Submission: `submissions/01_main/stability_blend/submission_best.csv`

Submission shape was verified as `538150 x 2` with columns `ID,prediction`.

### Interpretation

This is a more defensible next submission candidate than the pure top200 80/20 winner. It does not maximize the optimistic single-split score, but it directly addresses the stability problem observed after the Kaggle result.

### Kaggle Result

Submitted `submissions/01_main/stability_blend/submission_best.csv`.

| Split | Score |
| --- | ---: |
| Public | 0.06237 |
| Private | 0.05589 |

Interpretation: the stability blend substantially improved public score versus the previous top200 submission (`0.06237` vs `0.02826`), but private score decreased (`0.05589` vs `0.06610`). The rolling-stability objective improved one kind of robustness, but it did not match the private leaderboard distribution. The current best private result remains the Pearson top200 submission.

Next modeling implication: do not further optimize only for low rolling variance. The next round should compare candidate behavior against public/private-like splits, likely by constructing validation slices that better resemble the private segment, or by using rank/scale post-processing and calibration checks before adding model complexity.

## 2026-06-17: Submission Blend Calibration

### Goal

Use the two submitted models as endpoints and search a submission-level blend that can recover some public score without giving up the private-like validation behavior of the Pearson top200 model.

Endpoint leaderboard results:

| Submission | Public | Private |
| --- | ---: | ---: |
| Pearson top200 | 0.02826 | 0.06610 |
| stability blend | 0.06237 | 0.05589 |

### Implementation

- Added config: `configs/01_main_submission_blend.yaml`.
- Added script: `scripts/run_submission_blend.py`.
- Inputs:
  - `submissions/01_main/pearson_topk/submission_best.csv`
  - `submissions/01_main/stability_blend/submission_best.csv`
  - matching holdout and rolling validation predictions from prior runs.
- Evaluated raw and z-score blends on:
  - `holdout_80_100`
  - `fold_50_60`
  - `fold_60_70`
  - `fold_70_80`
  - `fold_80_90`

### Findings

- Source submission prediction correlation: `0.782205`.
- Pure top200 is still best on the private-like 80/20 holdout.
- A small stability component improves the worst rolling fold with minimal holdout loss.
- The best balanced validation objective uses a larger stability component, but it sacrifices more holdout Pearson.

### Generated Candidates

| Candidate | Formula | Holdout Pearson | Rolling Mean | Rolling Std | Rolling Min | Linear Public Proxy | Linear Private Proxy |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| private_safe | `0.85 * top200 + 0.15 * stability` | 0.124352 | 0.153129 | 0.062887 | 0.091246 | 0.033377 | 0.064569 |
| balanced | `0.45 * top200 + 0.55 * stability` | 0.117788 | 0.145881 | 0.036770 | 0.106824 | 0.047021 | 0.060485 |
| public_probe | `0.20 * top200 + 0.80 * stability` | 0.110363 | 0.138804 | 0.024292 | 0.108925 | 0.055548 | 0.057932 |

Default submission:

```text
submissions/01_main/submission_blend/submission_best.csv
```

This is the `private_safe` candidate. It was selected because its holdout Pearson is only about `0.0007` below pure top200 while the rolling minimum improves from `0.081128` to `0.091246`.

Suggested submission order:

1. `submission_best.csv` / `submission_private_safe.csv`
2. `submission_balanced.csv` only if the first candidate does not improve private score and more public/private tradeoff probing is acceptable.
3. `submission_public_probe.csv` only as a diagnostic public-heavy probe, not as a final private-score candidate.

### Kaggle Result

Submitted `submissions/01_main/submission_blend/submission_best.csv`, the `private_safe` candidate:

```text
0.85 * Pearson top200 + 0.15 * stability blend
```

| Split | Score |
| --- | ---: |
| Public | 0.03365 |
| Private | 0.06628 |

Interpretation: public score stayed low, but private score slightly improved over pure Pearson top200 (`0.06628` vs `0.06610`). This confirms that the private split prefers a model very close to top200, while a small stability component can still help. Public score should not be used as the main selection signal for the final report.

Next modeling implication: probe a narrow neighborhood around the private-safe blend, such as top200 weights `0.80`, `0.90`, and `0.95`, rather than moving toward public-heavy blends.
