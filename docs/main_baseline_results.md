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
