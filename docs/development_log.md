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

## 2026-06-17: Beta1 Narrow Blend Probe

### Stage Decision

Baseline1 is closed. The main task now enters Beta1, focused on small, submission-level private-score probes around the current best blend.

Current private best before Beta1:

```text
0.85 * Pearson top200 + 0.15 * stability blend
```

Leaderboard:

| Split | Score |
| --- | ---: |
| Public | 0.03365 |
| Private | 0.06628 |

### Implementation

- Added config: `configs/01_main_beta1_blend_probe.yaml`.
- Reused script: `scripts/run_submission_blend.py`.
- No retraining was performed.
- Input endpoints:
  - Pearson top200 submission
  - stability blend submission
- Output directory:
  - `runs/01_main/beta1_blend_probe/`
  - `submissions/01_main/beta1_blend_probe/`

### Candidate Grid

| Candidate | Top200 Weight | Stability Weight | Holdout Pearson | Rolling Mean | Rolling Std | Rolling Min |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| beta1_w080 | 0.800 | 0.200 | 0.123910 | 0.152618 | 0.059448 | 0.094150 |
| beta1_w085 | 0.850 | 0.150 | 0.124352 | 0.153129 | 0.062887 | 0.091246 |
| beta1_w0875 | 0.875 | 0.125 | 0.124534 | 0.153335 | 0.064603 | 0.089698 |
| beta1_w090 | 0.900 | 0.100 | 0.124691 | 0.153508 | 0.066314 | 0.088091 |
| beta1_w0925 | 0.925 | 0.075 | 0.124822 | 0.153647 | 0.068016 | 0.086426 |
| beta1_w095 | 0.950 | 0.050 | 0.124928 | 0.153753 | 0.069708 | 0.084709 |

Default Beta1 submission:

```text
submissions/01_main/beta1_blend_probe/submission_best.csv
```

This is `beta1_w090`:

```text
0.90 * Pearson top200 + 0.10 * stability blend
```

Rationale: `w090` moves closer to the private-favored top200 endpoint than `w085`, improves holdout Pearson from `0.124352` to `0.124691`, and keeps a small stability correction. If `w090` underperforms private, test `w080`; if it improves, test `w0925` or `w095`.

### Kaggle Result

Submitted `submissions/01_main/beta1_blend_probe/submission_best.csv`, the `beta1_w090` candidate:

```text
0.90 * Pearson top200 + 0.10 * stability blend
```

| Split | Score |
| --- | ---: |
| Public | 0.03182 |
| Private | 0.06627 |

Interpretation: `w090` is essentially tied with `w085` but does not beat it (`0.06627` vs `0.06628`). Moving closer to pure top200 did not improve private score despite better holdout Pearson. The best observed private blend remains `w085`.

Next beta1 action: test `beta1_w080`, not `w0925` or `w095`, because the observed private optimum is not moving toward the pure top200 endpoint.

### Kaggle Result: beta1_w080

Submitted `submissions/01_main/beta1_blend_probe/submission_beta1_w080.csv`, the `beta1_w080` candidate:

```text
0.80 * Pearson top200 + 0.20 * stability blend
```

| Split | Score |
| --- | ---: |
| Public | 0.03551 |
| Private | 0.06623 |

Interpretation: adding more stability weight than `w085` also fails to improve private score. The observed private ordering is:

| Candidate | Public | Private |
| --- | ---: | ---: |
| beta1_w080 | 0.03551 | 0.06623 |
| beta1_w085 | 0.03365 | 0.06628 |
| beta1_w090 | 0.03182 | 0.06627 |

Conclusion: the useful one-dimensional blend region is exhausted for now. The best observed point remains `w085`, and further probing on this axis is unlikely to give meaningful gain unless submission budget is very loose.

Next beta1 action: move from endpoint-weight probing to a different optimization mechanism, such as prediction distribution calibration, rank-based blending, or a new model signal.

## 2026-06-17: Beta1 Rank and Normal-Score Calibration

### Goal

Test a different optimization mechanism after raw endpoint-weight probing saturated. Instead of changing only the endpoint weights, transform prediction distributions before blending:

- `rank`: percentile rank scores, standardized.
- `normal_score`: rank-to-Gaussian normal scores, standardized.
- outputs are restored to the top200 submission scale for submission sanity.

### Implementation

- Extended `scripts/run_submission_blend.py` with `rank` and `normal_score` modes.
- Added config: `configs/01_main_beta1_rank_calibration.yaml`.
- No model retraining was performed.
- Output directory:
  - `runs/01_main/beta1_rank_calibration/`
  - `submissions/01_main/beta1_rank_calibration/`

### Candidate Results

| Candidate | Mode | Top200 Weight | Holdout Pearson | Rolling Mean | Rolling Std | Rolling Min |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| raw_w085 | raw | 0.85 | 0.124352 | 0.153129 | 0.062887 | 0.091246 |
| zscore_w085 | zscore | 0.85 | 0.124345 | 0.153701 | 0.065089 | 0.089119 |
| rank_w080 | rank | 0.80 | 0.124315 | 0.145268 | 0.026661 | 0.108466 |
| rank_w085 | rank | 0.85 | 0.124248 | 0.145012 | 0.028825 | 0.106015 |
| rank_w090 | rank | 0.90 | 0.124061 | 0.144629 | 0.030960 | 0.103448 |
| normal_w080 | normal_score | 0.80 | 0.126394 | 0.152866 | 0.043338 | 0.103557 |
| normal_w085 | normal_score | 0.85 | 0.126615 | 0.152885 | 0.045816 | 0.101175 |
| normal_w090 | normal_score | 0.90 | 0.126724 | 0.152782 | 0.048224 | 0.098691 |

Default submission:

```text
submissions/01_main/beta1_rank_calibration/submission_best.csv
```

This is `normal_w085`:

```text
normal_score(0.85 * top200_rank_signal + 0.15 * stability_rank_signal), restored to top200 scale
```

Rationale: `normal_w085` keeps the empirically best raw weight `0.85`, improves holdout Pearson from `0.124352` to `0.126615`, and keeps rolling behavior stronger than the raw blend. `normal_w090` has slightly higher holdout, but prior raw `w090` did not improve private score, so `normal_w085` is the safer first probe.

Suggested submission order:

1. `submission_best.csv` / `submission_normal_w085.csv`.
2. If private improves, test `submission_normal_w090.csv`.
3. If private falls but rolling robustness looks useful, test `submission_normal_w080.csv`.

### Kaggle Result

Submitted:

- `submissions/01_main/beta1_rank_calibration/submission_normal_w085.csv`
- `submissions/01_main/beta1_rank_calibration/submission_normal_w080.csv`

| Candidate | Public | Private |
| --- | ---: | ---: |
| normal_w085 | 0.03707 | 0.06217 |
| normal_w080 | 0.03902 | 0.06231 |

Interpretation: normal-score calibration improved offline holdout Pearson, but it materially hurt private leaderboard score. This is another validation mismatch and should not be used as the final direction. The current best remains raw `w085` with private `0.06628`.

Next beta1 action: stop rank/normal-score submission calibration for the main score. Move to a genuinely new signal, such as another top-k model family, a time-slice-specific model, or a separate feature-selection/dimensionality-reduction experiment that can add diversity without destroying the private-favored top200 signal.

## 2026-06-17: Beta1 New Signal Blend

### Goal

Move beyond two-endpoint calibration by introducing a genuinely different signal into the current best raw `w085` submission. Candidate signals were selected from existing validation predictions first, then only the lightweight missing final signal was trained.

Current base:

```text
raw_w085 = 0.85 * Pearson top200 + 0.15 * stability blend
```

### Implementation

- Added config: `configs/01_main_beta1_signal_blend.yaml`.
- Added script: `scripts/run_beta1_signal_blend.py`.
- Generated a final Ridge submission for `top100_ridge`.
- Reused existing final submissions for:
  - `full_ensemble`
  - `full_ridge`
- Blended each signal into the raw `w085` base with small weights.

### Candidate Results

| Candidate | Signal | Signal Weight | Holdout Pearson | Rolling Mean | Rolling Std | Rolling Min | Mean Corr With Base |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| top100_ridge_w020 | top100_ridge | 0.20 | 0.126226 | 0.151209 | 0.062663 | 0.092158 | 0.778503 |
| top100_ridge_w030 | top100_ridge | 0.30 | 0.126511 | 0.149044 | 0.061897 | 0.091906 | 0.778503 |
| full_ensemble_w020 | full_ensemble | 0.20 | 0.125491 | 0.151189 | 0.046383 | 0.105264 | 0.711032 |
| full_ensemble_w030 | full_ensemble | 0.30 | 0.124295 | 0.148130 | 0.038810 | 0.107463 | 0.711032 |
| full_ridge_w015 | full_ridge | 0.15 | 0.125249 | 0.143882 | 0.042254 | 0.098323 | 0.627808 |

Default submission:

```text
submissions/01_main/beta1_signal_blend/submission_best.csv
```

This is `top100_ridge_w030`:

```text
0.70 * raw_w085 + 0.30 * top100_ridge
```

Rationale: `top100_ridge_w030` gives the best holdout Pearson among the tested new-signal blends while keeping rolling minimum slightly above raw `w085`. If it fails on private, the next safer probe is `full_ensemble_w020`, which has a lower holdout gain but much better rolling minimum.

### Kaggle Result

Submitted:

- `submissions/01_main/beta1_signal_blend/submission_best.csv` / `submission_top100_ridge_w030.csv`
- `submissions/01_main/beta1_signal_blend/submission_full_ensemble_w020.csv`
- `submissions/01_main/beta1_signal_blend/submission_top100_ridge_w0325.csv`
- `submissions/01_main/beta1_signal_blend/submission_top100_ridge_w035.csv`
- `submissions/01_main/beta1_signal_blend/submission_top100_ridge_w040.csv`
- `submissions/01_main/beta1_signal_blend/submission_top100_ridge_w045.csv`
- `submissions/01_main/beta1_signal_blend/submission_top100_ridge_w050.csv`

| Candidate | Public | Private |
| --- | ---: | ---: |
| full_ensemble_w020 | 0.03625 | 0.06443 |
| top100_ridge_w030 | 0.03651 | 0.07539 |
| top100_ridge_w0325 | 0.03672 | 0.07615 |
| top100_ridge_w035 | 0.03692 | 0.07692 |
| top100_ridge_w040 | 0.03730 | 0.07842 |
| top100_ridge_w045 | 0.03763 | 0.07989 |
| top100_ridge_w050 | 0.03793 | 0.08131 |
| top100_ridge_w060 | 0.03834 | 0.08393 |
| top100_ridge_w080 | 0.03837 | 0.08781 |
| top100_ridge_w095 | 0.03758 | 0.08900 |
| top100_ridge_w100 | 0.03716 | 0.08901 |
| top100_ridge_w105 | 0.03666 | 0.08881 |
| top100_ridge_w110 | 0.03610 | 0.08842 |
| top100_ridge_w125 | 0.03407 | 0.08612 |

Interpretation: the private score continues to improve from `w030` through pure `top100_ridge`, making `top100_ridge_w100` the new best private result. It improves over raw `w085` from `0.06628` to `0.08901`. The weak `full_ensemble_w020` result confirms that not every diverse signal helps; the useful direction in this round is specifically the top100 Ridge signal.

Conclusion: the one-dimensional `raw_w085` to `top100_ridge` weight axis has reached a local bottleneck. `w095` and `w100` are effectively tied, while `w105`, `w110`, and `w125` all decline. `submission_best.csv` now points to `top100_ridge_w100`. The next improvement should come from a new direction, such as a different top-k Ridge signal, a Ridge-only top-k ensemble, or another feature-selection family, rather than further scanning this same blend axis.

## 2026-06-17: Beta2 Ridge Top-k Family

### Goal

Close Beta1 and move to a new signal direction. Since pure `top100_ridge` became the Beta1 best, Beta2 tests whether neighboring Pearson top-k Ridge models or Ridge-only blends can improve the score.

### Implementation

- Added config: `configs/01_main_beta2_ridge_topk.yaml`.
- Added script: `scripts/run_beta2_ridge_topk.py`.
- Generated final Ridge submissions for:
  - `top50_ridge`
  - `top100_ridge`
  - `top200_ridge`
  - `top300_ridge`
  - `top500_ridge`
  - `full_ridge`
- Generated Ridge-only blends, focusing on `top50_ridge` and `top100_ridge` after `top200_ridge` failed on the leaderboard.

### Kaggle Result

| Candidate | Public | Private |
| --- | ---: | ---: |
| top100_ridge | 0.03716 | 0.08901 |
| ridge_top100_top200_avg | 0.03321 | 0.08099 |
| top200_ridge | 0.02629 | 0.06625 |
| top50_ridge | 0.05162 | 0.08821 |
| ridge_top100_080_top50_020 | 0.04092 | 0.09117 |
| ridge_top100_070_top50_030 | 0.04275 | 0.09197 |
| ridge_top100_060_top50_040 | 0.04450 | 0.09253 |
| ridge_top50_top100_avg | 0.04614 | 0.09278 |
| ridge_top100_045_top50_055 | 0.04690 | 0.09279 |
| ridge_top100_040_top50_060 | 0.04763 | 0.09269 |
| ridge_top100_0475_top50_0525 | 0.04652 | 0.09280 |

Interpretation: `top200_ridge` has the best offline holdout Pearson but fails badly on the leaderboard, confirming another validation mismatch. `top50_ridge` has much better public score and nearly matches `top100_ridge` private score. Blending `top50_ridge` with `top100_ridge` gives a stable improvement, with a broad plateau around `45%` to `50%` top100 weight.

Conclusion: current best is `ridge_top100_0475_top50_0525`:

```text
0.475 * top100_ridge + 0.525 * top50_ridge
```

`submission_best.csv` now points to this candidate. The local top50/top100 blend axis is close to saturated because `45/55`, `47.5/52.5`, and `50/50` are nearly tied. The next Beta2 direction should test whether intermediate direct feature widths such as top75/top80/top90 Ridge can replace or improve this blend.

## 2026-06-18: Beta2 Intermediate Ridge Widths

### Goal

Test whether the `top50_ridge` / `top100_ridge` blend improvement can be replaced by a single direct intermediate feature width.

### Implementation

- Added config: `configs/01_main_beta2_intermediate_ridge.yaml`.
- Added script: `scripts/run_beta2_intermediate_ridge.py`.
- Reused the train-split Pearson feature ranking from `runs/01_main/pearson_topk/pearson_feature_ranking.csv`.
- Trained Ridge-only submissions for `top60`, `top70`, `top75`, `top80`, and `top90`.
- Searched Ridge alpha over `[10, 100, 1000, 3000, 10000]`.

### Kaggle Result

| Candidate | Alpha | Public | Private |
| --- | ---: | ---: | ---: |
| top90_ridge | 10 | 0.04789 | 0.09063 |
| top80_ridge | 3000 | 0.05112 | 0.09050 |

Offline validation ranked `top90_ridge` highest among the intermediate widths, but both submitted intermediate Ridge models are below the current top50/top100 blend best.

Conclusion: direct intermediate feature width does not replace the blend. The current best remains:

```text
0.475 * top100_ridge + 0.525 * top50_ridge
```

Next direction: test whether alpha variants for `top50_ridge` and `top100_ridge`, or a different feature selection criterion, can produce better components before blending.

## 2026-06-18: Beta2 Alpha Variant Ridge Refinement

### Goal

Tune the already validated `top50_ridge` and `top100_ridge` components instead of continuing to scan feature widths. The previous best before this run was:

```text
0.475 * top100_ridge(alpha=1000) + 0.525 * top50_ridge(alpha=1000)
```

with private score `0.09280`.

### Implementation

- Added config: `configs/01_main_beta2_alpha_variants.yaml`.
- Added script: `scripts/run_beta2_alpha_variants.py`.
- Fixed feature sets to the existing Pearson `top50` and `top100` selected features.
- Trained Ridge alpha variants for each component:
  - `[10, 30, 100, 300, 1000, 3000, 10000]`
- Evaluated all `top50_alphaA + top100_alphaB` blends over top50 weights:
  - `[0.45, 0.475, 0.50, 0.525, 0.55, 0.60]`
- Also generated an alpha ensemble over `[100, 1000, 10000]` for each component.
- Materialized only three selected submission files to avoid writing hundreds of large CSVs.

### Kaggle Result

| Candidate | Public | Private | Notes |
| --- | ---: | ---: | --- |
| Previous best: `top50/top100 alpha1000 blend` | 0.04652 | 0.09280 | Baseline for this run |
| `0.45 * top50_alpha10000 + 0.55 * top100_alpha30` | 0.04808 | 0.09385 | New best |
| `0.45 * top50_alpha10 + 0.55 * top100_alpha10` | 0.04745 | 0.09307 | Lower-correlation backup, improves old best |
| `alpha_ensemble_top50w525_top100w475` | 0.04831 | 0.09360 | Smoother, improves old best |

Interpretation: alpha tuning is useful. The best result moves away from the previous alpha1000/alpha1000 pair toward a strongly regularized `top50` component and a lightly regularized `top100` component. The alpha ensemble also improves the old best, but does not beat the selected alpha pair.

Current best:

```text
0.45 * top50_ridge(alpha=10000) + 0.55 * top100_ridge(alpha=30)
```

The corresponding file is:

```text
submissions/01_main/beta2_alpha_variants/submission_best.csv
```

Next direction: refine around this alpha pair and weight region, especially `top50_alpha10000` with `top100_alpha10/30/100` and top50 weights near `0.40` to `0.50`.

## 2026-06-18: Beta2 High-Alpha Local Ridge Refinement

### Goal

Continue the proven Ridge-only direction after `top50_alpha10000/top100_alpha30` improved the private score to `0.09385`. This run tests whether stronger Ridge regularization and narrower blend weights can improve the `top50` / `top100` component pair.

### Implementation

- Extended `scripts/run_beta2_alpha_variants.py` so each component can define its own alpha grid.
- Added local refinement configs:
  - `configs/01_main_beta2_local_alpha_refine.yaml`
  - `configs/01_main_beta2_high_alpha_refine.yaml`
  - `configs/01_main_beta2_high_alpha_fine.yaml`
  - `configs/01_main_beta2_low_top100_fine.yaml`
  - `configs/01_main_beta2_high_top50_tail.yaml`
- Kept the same row-order 80/20 validation split and train-only preprocessing logic.
- Materialized only selected submission candidates plus manually selected high-potential candidates around the current best.

### Kaggle Result

| Candidate | Public | Private | Notes |
| --- | ---: | ---: | --- |
| Previous best: `0.45 * top50_alpha10000 + 0.55 * top100_alpha30` | 0.04808 | 0.09385 | Baseline for this run |
| `0.35 * top50_alpha50000 + 0.65 * top100_alpha20` | 0.04779 | 0.09366 | Offline best, did not transfer |
| `0.525 * top50_alpha50000 + 0.475 * top100_alpha300` | 0.05004 | 0.09536 | Boundary candidate improved |
| `0.525 * top50_alpha100000 + 0.475 * top100_alpha150` | 0.05124 | 0.09619 | Higher top50 alpha improved |
| `0.5 * top50_alpha125000 + 0.5 * top100_alpha50` | 0.05156 | 0.09619 | Public improved, private tied |
| `0.475 * top50_alpha100000 + 0.525 * top100_alpha20` | 0.05139 | 0.09574 | Lower top100 alpha over-correction |
| `0.5 * top50_alpha200000 + 0.5 * top100_alpha50` | 0.05160 | 0.09638 | Current best |

Current best:

```text
0.5 * top50_ridge(alpha=200000) + 0.5 * top100_ridge(alpha=50)
```

The corresponding local file is:

```text
submissions/01_main/beta2_high_alpha_fine/submission_best.csv
```

Interpretation: the useful direction was not the highest offline holdout candidate. It came from lower-correlation boundary probes with stronger `top50` regularization. A final high-top50 tail scan showed that increasing `top50_alpha` beyond `200000` reduces holdout along the current best axis, while decreasing `top100_alpha` to `20` also hurts the leaderboard. This suggests the Ridge alpha/weight refinement line is now near a local bottleneck.

Next direction: move away from further alpha/weight micro-tuning. Better candidates are a new feature-selection signal, rolling-stability-aware top-k selection, or studying high-ranking solutions for new assumptions before adding more submissions.

## 2026-06-18: Beta3 New Feature-Selection Signals

### Goal

Move beyond Pearson top-k alpha/weight micro-tuning and test whether different train-only feature-selection criteria can create new Ridge signals that complement the current best:

```text
0.5 * top50_ridge(alpha=200000) + 0.5 * top100_ridge(alpha=50)
```

with private score `0.09638`.

### Implementation

- Added config: `configs/01_main_beta3_new_feature_signals.yaml`.
- Added script: `scripts/run_beta3_new_feature_signals.py`.
- Extended `src/drw_crypto/feature_selection.py` with reusable ranking utilities:
  - arbitrary-target Pearson/Spearman ranking;
  - Spearman top-k ranking;
  - stable Pearson ranking across train-only time windows.
- Tested three new signal families:
  - Spearman top-k Ridge: `top50`, `top100`, `top200`;
  - rolling-stability-aware Pearson top-k Ridge: `top50`, `top100`, `top200`;
  - residual-correlation top-k Ridge: residual Pearson `top50/top100/top200` and residual Spearman `top100`.
- Residual ranking target:

```text
residual = label - current_best_train_prediction
```

computed only on the 80% training split.
- Each signal was blended into the current best with signal weights:
  - `0.05`, `0.10`, `0.15`, `0.20`.
- Candidate selection recorded holdout Pearson, RMSE, prediction std, signal correlation with current best, blend correlation, and validation-bin stability.

### Kaggle Result

| Candidate | Public | Private | Notes |
| --- | ---: | ---: | --- |
| Previous best: `top50_alpha200000/top100_alpha50` | 0.05160 | 0.09638 | Beta2 best |
| `0.80 * current_best + 0.20 * residual_pearson_top100` | 0.05835 | 0.09929 | Strong new residual signal |
| `0.80 * current_best + 0.20 * spearman_top50` | 0.05427 | 0.09998 | Current best |
| `0.95 * current_best + 0.05 * stable_pearson_top200` | 0.04996 | 0.09408 | Stable ranking did not transfer |

Current best:

```text
0.80 * beta2_current_best + 0.20 * spearman_top50_ridge
```

where:

```text
beta2_current_best = 0.5 * top50_ridge(alpha=200000) + 0.5 * top100_ridge(alpha=50)
```

The corresponding local file is:

```text
submissions/01_main/beta3_new_feature_signals/submission_best.csv
```

Interpretation: the new feature-selection direction works. The useful gains came from feature rankings that are not identical to Pearson top-k: Spearman top50 and residual Pearson top100. Rolling-stability-aware selection, despite being reasonable for analysis, failed on the leaderboard and should not be the next main optimization path.

Next direction: test a controlled blend of the two successful Beta3 signals:

```text
current_best + residual_pearson_top100 + spearman_top50
```

with small weight grids, instead of adding more stable-top-k variants.
