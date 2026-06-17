# Main Baseline Results

## Data

- Train file: `data/raw/train.parquet`
- Test file: `data/raw/test.parquet`
- Sample submission: `data/raw/sample_submission.csv`
- Train shape: 525,886 rows x 786 columns
- Test shape: 538,150 rows x 786 columns
- Target column: `label`
- Submission columns: `ID`, `prediction`
- Feature count: 785

The `label` column in `test.parquet` is all zeros and is treated as a target placeholder, not as a feature.

## Validation Protocol

- Split: chronological 80/20 split by row order
- Validation rows: 105,178
- Primary metric: Pearson correlation
- Preprocessing fit scope: training split only
- Runtime environment: `D:\PythonEnvs\envs\py313-science`

## Full-Feature Baseline Metrics

| Model | Pearson | RMSE |
| --- | ---: | ---: |
| Ridge + LightGBM weighted ensemble | 0.100263 | 1.126039 |
| Mean ensemble | 0.099955 | 1.107722 |
| Ridge | 0.097166 | 1.235401 |
| LightGBM | 0.063721 | 1.079047 |
| XGBoost | 0.047563 | 1.227862 |
| CatBoost | 0.040489 | 1.106226 |

The selected first submission model is `ensemble_ridge_lgbm` with Ridge weight `0.60`.

## Artifacts

- Full validation metrics: `runs/01_main/baseline_full/metrics_all_models_validation.csv`
- Main run metrics: `runs/01_main/baseline_full/metrics_main_models.csv`
- Validation predictions: `runs/01_main/baseline_full/valid_predictions/`
- Best submission: `submissions/01_main/baseline_full/submission_best.csv`
- Ridge submission: `submissions/01_main/baseline_full/submission_ridge.csv`
- LightGBM submission: `submissions/01_main/baseline_full/submission_lightgbm.csv`
- Ridge + LightGBM submission: `submissions/01_main/baseline_full/submission_ensemble_ridge_lgbm.csv`

## Reference

- First-place writeup for later study: https://www.kaggle.com/competitions/drw-crypto-market-prediction/writeups/drw-solution-1st

## Main Optimization Result: Pearson Top-k

After the full-feature baseline, a train-only Pearson top-k feature selection experiment was run with Ridge, LightGBM, and their weighted ensemble.

| Scheme | Model | Features | Pearson | RMSE |
| --- | --- | ---: | ---: | ---: |
| top200 | Ridge + LightGBM weighted ensemble | 200 | 0.125069 | 1.079047 |
| top200 | Ridge | 200 | 0.124899 | 1.081521 |
| top100 | Ridge + LightGBM weighted ensemble | 100 | 0.114977 | 1.071005 |
| full | Ridge + LightGBM weighted ensemble | 785 | 0.100263 | 1.126039 |

Selected optimized submission:

```text
submissions/01_main/pearson_topk/submission_best.csv
```

This file has 538,150 rows and columns `ID`, `prediction`.

Kaggle result for this optimized submission:

| Split | Score |
| --- | ---: |
| Public | 0.02826 |
| Private | 0.06610 |

The leaderboard result is materially below the offline validation Pearson `0.125069`, so the next validation improvement should focus on rolling/time-stability checks rather than trusting a single 80/20 split.

## Rolling Validation Check

After the Kaggle result, a four-fold expanding rolling validation was run for `full`, `top50`, `top100`, `top200`, `top300`, and `top500`.

Ridge + LightGBM ensemble summary:

| Scheme | Mean Pearson | Std | Min | Max |
| --- | ---: | ---: | ---: | ---: |
| full | 0.115468 | 0.019837 | 0.098538 | 0.147126 |
| top50 | 0.121328 | 0.035865 | 0.078946 | 0.170520 |
| top200 | 0.153862 | 0.073044 | 0.081128 | 0.272860 |
| top300 | 0.162315 | 0.051951 | 0.103448 | 0.239494 |

Main conclusion: `top200` and `top300` still look strong offline, but their rolling variance is high. The `full` scheme is the most stable reference. Future submissions should not use the single 80/20 top-k validation score as the only selection rule.

Rolling artifacts:

- `runs/03_temporal/rolling_validation/metrics_rolling_validation.csv`
- `runs/03_temporal/rolling_validation/scheme_stability_summary.csv`
- `runs/03_temporal/rolling_validation/lightgbm_importance_stability.csv`
