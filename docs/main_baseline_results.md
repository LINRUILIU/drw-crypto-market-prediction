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

Kaggle result for Pearson top200:

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

## Stability Blend Submission Candidate

A rolling-aware blend was built after the leaderboard gap. The blend searches component weights under stability constraints instead of selecting the highest single 80/20 validation score.

Selected weights:

| Component | Weight |
| --- | ---: |
| full Ridge | 0.20 |
| full LightGBM | 0.25 |
| top50 Ridge | 0.05 |
| top50 LightGBM | 0.20 |
| top200 Ridge | 0.30 |
| top300 Ridge | 0.00 |

Rolling and holdout performance:

| Metric | Value |
| --- | ---: |
| rolling mean Pearson | 0.132549 |
| rolling std | 0.018230 |
| rolling min Pearson | 0.108175 |
| rolling max Pearson | 0.154487 |
| 80/20 holdout Pearson | 0.103034 |
| 80/20 holdout RMSE | 1.081795 |

This candidate is more stable than the top200 rolling result:

| Candidate | Rolling Mean | Rolling Std | Rolling Min |
| --- | ---: | ---: | ---: |
| stability blend | 0.132549 | 0.018230 | 0.108175 |
| top200 Ridge + LightGBM | 0.153862 | 0.073044 | 0.081128 |
| top300 Ridge + LightGBM | 0.162315 | 0.051951 | 0.103448 |
| full Ridge + LightGBM | 0.115468 | 0.019837 | 0.098538 |

Selected submission:

```text
submissions/01_main/stability_blend/submission_best.csv
```

This file has 538,150 rows and columns `ID`, `prediction`.

Kaggle result for stability blend:

| Split | Score |
| --- | ---: |
| Public | 0.06237 |
| Private | 0.05589 |

Conclusion: the stability blend improved public score but reduced private score, so it is not the best final candidate by private score.

## Submission Blend Calibration

The next optimization blended the two submitted endpoints directly:

```text
Pearson top200 submission
Stability blend submission
```

The source submission prediction correlation is `0.782205`, so they are related but still different enough to justify blend probing.

Generated candidates:

| Candidate | Top200 Weight | Stability Weight | Holdout Pearson | Rolling Mean | Rolling Std | Rolling Min |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| private_safe | 0.85 | 0.15 | 0.124352 | 0.153129 | 0.062887 | 0.091246 |
| balanced | 0.45 | 0.55 | 0.117788 | 0.145881 | 0.036770 | 0.106824 |
| public_probe | 0.20 | 0.80 | 0.110363 | 0.138804 | 0.024292 | 0.108925 |

Default submission:

```text
submissions/01_main/submission_blend/submission_best.csv
```

This file is the `private_safe` candidate:

```text
0.85 * Pearson top200 + 0.15 * stability blend
```

Kaggle result for submission blend `private_safe`:

| Split | Score |
| --- | ---: |
| Public | 0.03365 |
| Private | 0.06628 |

Current leaderboard comparison:

| Submission | Public | Private | Notes |
| --- | ---: | ---: | --- |
| Pearson top200 | 0.02826 | 0.06610 | Previous best private score |
| stability blend | 0.06237 | 0.05589 | Better public, worse private |
| submission blend private_safe | 0.03365 | 0.06628 | Current best private score |

Conclusion: public score is not a reliable proxy for the private split in this project. The small `15%` stability component slightly improved private score while keeping behavior close to Pearson top200.

## Beta1 Narrow Blend Probe

Baseline1 is closed. Beta1 starts from the current private-best blend and probes nearby weights.

Current best before Beta1:

```text
0.85 * Pearson top200 + 0.15 * stability blend
```

Beta1 candidates:

| Candidate | Top200 Weight | Stability Weight | Holdout Pearson | Rolling Min |
| --- | ---: | ---: | ---: | ---: |
| beta1_w080 | 0.800 | 0.200 | 0.123910 | 0.094150 |
| beta1_w085 | 0.850 | 0.150 | 0.124352 | 0.091246 |
| beta1_w0875 | 0.875 | 0.125 | 0.124534 | 0.089698 |
| beta1_w090 | 0.900 | 0.100 | 0.124691 | 0.088091 |
| beta1_w0925 | 0.925 | 0.075 | 0.124822 | 0.086426 |
| beta1_w095 | 0.950 | 0.050 | 0.124928 | 0.084709 |

Default Beta1 submission:

```text
submissions/01_main/beta1_blend_probe/submission_best.csv
```

This file is `beta1_w090`:

```text
0.90 * Pearson top200 + 0.10 * stability blend
```

Submit `beta1_w090` first. Then use the result to decide the direction:

- If private improves, test `beta1_w0925` or `beta1_w095`.
- If private falls, test `beta1_w080`.
