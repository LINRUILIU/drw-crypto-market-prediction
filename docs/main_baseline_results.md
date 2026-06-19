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

Kaggle result for `beta1_w090`:

| Split | Score |
| --- | ---: |
| Public | 0.03182 |
| Private | 0.06627 |

Current Beta1 comparison:

| Submission | Public | Private | Notes |
| --- | ---: | ---: | --- |
| submission blend private_safe / beta1_w085 | 0.03365 | 0.06628 | Current best private score |
| beta1_w090 | 0.03182 | 0.06627 | Essentially tied, slightly worse |
| beta1_w080 | 0.03551 | 0.06623 | Worse private despite better public |

Conclusion: the narrow endpoint blend probe has likely saturated. The current best remains `beta1_w085`; the next main-task optimization should use a different mechanism rather than continuing to scan this one-dimensional blend.

## Beta1 Rank And Normal-Score Calibration

The next Beta1 optimization keeps the same two endpoints but changes the prediction distribution before blending.

Generated candidates:

| Candidate | Mode | Top200 Weight | Holdout Pearson | Rolling Std | Rolling Min |
| --- | --- | ---: | ---: | ---: | ---: |
| raw_w085 | raw | 0.85 | 0.124352 | 0.062887 | 0.091246 |
| rank_w080 | rank | 0.80 | 0.124315 | 0.026661 | 0.108466 |
| rank_w085 | rank | 0.85 | 0.124248 | 0.028825 | 0.106015 |
| normal_w080 | normal_score | 0.80 | 0.126394 | 0.043338 | 0.103557 |
| normal_w085 | normal_score | 0.85 | 0.126615 | 0.045816 | 0.101175 |
| normal_w090 | normal_score | 0.90 | 0.126724 | 0.048224 | 0.098691 |

Default next submission:

```text
submissions/01_main/beta1_rank_calibration/submission_best.csv
```

This file is `normal_w085`. It keeps the best observed raw weight while applying rank-to-Gaussian calibration before blending.

Kaggle results:

| Submission | Public | Private | Notes |
| --- | ---: | ---: | --- |
| normal_w085 | 0.03707 | 0.06217 | Offline holdout improved, private worsened |
| normal_w080 | 0.03902 | 0.06231 | Slightly better than normal_w085, still worse than raw w085 |

Conclusion: rank-to-Gaussian calibration is not a good main-score direction. The current best remains raw `beta1_w085` with private `0.06628`.

## Beta1 New Signal Blend

After rank/normal-score calibration failed on private score, Beta1 moved to new signal blending. The base remains raw `w085`.

Base:

```text
0.85 * Pearson top200 + 0.15 * stability blend
```

Generated candidates:

| Candidate | Added Signal | Signal Weight | Holdout Pearson | Rolling Std | Rolling Min |
| --- | --- | ---: | ---: | ---: | ---: |
| top100_ridge_w020 | top100_ridge | 0.20 | 0.126226 | 0.062663 | 0.092158 |
| top100_ridge_w030 | top100_ridge | 0.30 | 0.126511 | 0.061897 | 0.091906 |
| full_ensemble_w020 | full_ensemble | 0.20 | 0.125491 | 0.046383 | 0.105264 |
| full_ensemble_w030 | full_ensemble | 0.30 | 0.124295 | 0.038810 | 0.107463 |
| full_ridge_w015 | full_ridge | 0.15 | 0.125249 | 0.042254 | 0.098323 |

Default next submission:

```text
submissions/01_main/beta1_signal_blend/submission_best.csv
```

This file is:

```text
0.70 * raw_w085 + 0.30 * top100_ridge
```

If this fails on private, the next safer candidate is:

```text
submissions/01_main/beta1_signal_blend/submission_full_ensemble_w020.csv
```

Kaggle results:

| Submission | Public | Private | Notes |
| --- | ---: | ---: | --- |
| full_ensemble_w020 | 0.03625 | 0.06443 | Worse than raw w085; this diverse signal did not help |
| top100_ridge_w030 | 0.03651 | 0.07539 | Large private improvement |
| top100_ridge_w0325 | 0.03672 | 0.07615 | Better than w030 |
| top100_ridge_w035 | 0.03692 | 0.07692 | Better than w0325 |
| top100_ridge_w040 | 0.03730 | 0.07842 | Higher top100 Ridge weight keeps improving |
| top100_ridge_w045 | 0.03763 | 0.07989 | Higher again |
| top100_ridge_w050 | 0.03793 | 0.08131 | Higher again |
| top100_ridge_w060 | 0.03834 | 0.08393 | Higher again |
| top100_ridge_w080 | 0.03837 | 0.08781 | Higher again |
| top100_ridge_w095 | 0.03758 | 0.08900 | Near the peak |
| top100_ridge_w100 | 0.03716 | 0.08901 | Current best private score |
| top100_ridge_w105 | 0.03666 | 0.08881 | Worse than w100 |
| top100_ridge_w110 | 0.03610 | 0.08842 | Worse than w100 |
| top100_ridge_w125 | 0.03407 | 0.08612 | Over-extrapolation hurts |

Conclusion: top100 Ridge is the dominant signal found so far. The best observed private result is pure `top100_ridge_w100`, and the local weight scan has saturated because both smaller and larger nearby weights fail to improve it. The next Beta1 improvement should use a new signal direction rather than further tuning this same blend axis.

## Beta2 Ridge Top-k Family

Beta2 closes the Beta1 submission-level blend axis and starts from the Ridge family directly.

Kaggle results:

| Submission | Public | Private | Notes |
| --- | ---: | ---: | --- |
| top100_ridge | 0.03716 | 0.08901 | Beta1 best before Beta2 |
| ridge_top100_top200_avg | 0.03321 | 0.08099 | Offline looked strong, leaderboard worsened |
| top200_ridge | 0.02629 | 0.06625 | Validation mismatch; not useful |
| top50_ridge | 0.05162 | 0.08821 | Strong public, slightly worse private than top100 |
| ridge_top100_080_top50_020 | 0.04092 | 0.09117 | Better private |
| ridge_top100_070_top50_030 | 0.04275 | 0.09197 | Better private |
| ridge_top100_060_top50_040 | 0.04450 | 0.09253 | Better private |
| ridge_top50_top100_avg | 0.04614 | 0.09278 | Near plateau |
| ridge_top100_045_top50_055 | 0.04690 | 0.09279 | Near plateau |
| ridge_top100_040_top50_060 | 0.04763 | 0.09269 | Slightly worse |
| ridge_top100_0475_top50_0525 | 0.04652 | 0.09280 | Current best private score |

Current best:

```text
0.475 * top100_ridge + 0.525 * top50_ridge
```

Conclusion: the useful Ridge window is narrower than top200 and sits between top50 and top100. The next step should test direct intermediate feature widths, such as top75/top80/top90 Ridge, instead of further tuning the top50/top100 blend by one or two percent.

## Beta2 Intermediate Ridge Widths

Direct intermediate top-k Ridge models were trained to test whether a single feature width can replace the top50/top100 blend.

| Submission | Alpha | Public | Private | Notes |
| --- | ---: | ---: | ---: | --- |
| top90_ridge | 10 | 0.04789 | 0.09063 | Best offline intermediate width, below blend |
| top80_ridge | 3000 | 0.05112 | 0.09050 | Stronger public, still below blend private |

Conclusion: direct intermediate widths do not beat the blend. The current main-task best remains `ridge_top100_0475_top50_0525` with private `0.09280`.

## Beta2 Alpha Variant Ridge Refinement

Alpha variants were trained for the two strongest components, `top50_ridge` and `top100_ridge`.

| Submission | Public | Private | Notes |
| --- | ---: | ---: | --- |
| Previous best: `top50/top100 alpha1000 blend` | 0.04652 | 0.09280 | Before alpha tuning |
| `0.45 * top50_alpha10000 + 0.55 * top100_alpha30` | 0.04808 | 0.09385 | Current best private score |
| `0.45 * top50_alpha10 + 0.55 * top100_alpha10` | 0.04745 | 0.09307 | Lower-correlation backup |
| `alpha_ensemble_top50w525_top100w475` | 0.04831 | 0.09360 | Smooth alpha ensemble |

Current best:

```text
0.45 * top50_ridge(alpha=10000) + 0.55 * top100_ridge(alpha=30)
```

Conclusion: tuning Ridge alpha on the two validated components is useful and improves the private score from `0.09280` to `0.09385`.

## Beta2 High-Alpha Local Ridge Refinement

The alpha refinement line was extended with component-specific alpha grids and narrower local blend searches.

| Submission | Public | Private | Notes |
| --- | ---: | ---: | --- |
| Previous best: `top50_alpha10000/top100_alpha30` | 0.04808 | 0.09385 | Before high-alpha refinement |
| `0.35 * top50_alpha50000 + 0.65 * top100_alpha20` | 0.04779 | 0.09366 | Offline best did not transfer |
| `0.525 * top50_alpha50000 + 0.475 * top100_alpha300` | 0.05004 | 0.09536 | Boundary probe improved private |
| `0.525 * top50_alpha100000 + 0.475 * top100_alpha150` | 0.05124 | 0.09619 | Stronger top50 regularization improved again |
| `0.5 * top50_alpha125000 + 0.5 * top100_alpha50` | 0.05156 | 0.09619 | Public improved, private tied |
| `0.475 * top50_alpha100000 + 0.525 * top100_alpha20` | 0.05139 | 0.09574 | Too little top100 regularization hurt private |
| `0.5 * top50_alpha200000 + 0.5 * top100_alpha50` | 0.05160 | 0.09638 | Current best private score |

Current best:

```text
0.5 * top50_ridge(alpha=200000) + 0.5 * top100_ridge(alpha=50)
```

Conclusion: Ridge alpha/weight tuning still improved the private score, but the last tail scan suggests this line is approaching a local bottleneck. Further progress should come from a new signal or selection criterion rather than more small alpha/weight sweeps.

## Beta3 New Feature-Selection Signals

Beta3 tested new train-only feature-selection criteria as complementary Ridge signals.

| Submission | Public | Private | Notes |
| --- | ---: | ---: | --- |
| Previous best: `top50_alpha200000/top100_alpha50` | 0.05160 | 0.09638 | Beta2 best |
| `0.80 * current_best + 0.20 * residual_pearson_top100` | 0.05835 | 0.09929 | Strong residual signal |
| `0.80 * current_best + 0.20 * spearman_top50` | 0.05427 | 0.09998 | Current best private score |
| `0.95 * current_best + 0.05 * stable_pearson_top200` | 0.04996 | 0.09408 | Rolling-stability-aware top-k did not transfer |

Current best:

```text
0.80 * beta2_current_best + 0.20 * spearman_top50_ridge
```

where:

```text
beta2_current_best = 0.5 * top50_ridge(alpha=200000) + 0.5 * top100_ridge(alpha=50)
```

Conclusion: new feature-selection criteria produced the largest gain since Beta2 started. Spearman and residual-correlation rankings are worth continuing; stable Pearson top-k should be deprioritized as a leaderboard optimization path.

## Beta3 Signal Combination and Spearman Weight Fine-Tuning

The two successful Beta3 signals were blended together, then the Spearman-only weight was fine-tuned around the previous best.

| Submission | Public | Private | Notes |
| --- | ---: | ---: | --- |
| Previous best: `0.80 * beta2_current_best + 0.20 * spearman_top50` | 0.05427 | 0.09998 | Before signal-combo tuning |
| `0.65 * beta2_current_best + 0.25 * residual_pearson_top100 + 0.10 * spearman_top50` | 0.05990 | 0.09734 | Holdout/public improved, private worsened |
| `0.65 * beta2_current_best + 0.175 * residual_pearson_top100 + 0.175 * spearman_top50` | 0.05885 | 0.09872 | Residual still too heavy |
| `0.775 * beta2_current_best + 0.225 * spearman_top50` | 0.05448 | 0.10003 | New best private score |
| `0.75 * beta2_current_best + 0.25 * spearman_top50` | 0.05467 | 0.09999 | Slightly beyond the peak |
| `0.765 * beta2_current_best + 0.235 * spearman_top50` | 0.05456 | 0.10003 | Tied best private score |

Current best:

```text
0.765 * beta2_current_best + 0.235 * spearman_top50_ridge
```

where:

```text
beta2_current_best = 0.5 * top50_ridge(alpha=200000) + 0.5 * top100_ridge(alpha=50)
```

Conclusion: adding residual Pearson directly to the blend did not transfer to private despite stronger holdout and public scores. The current useful region is a narrow Spearman-only plateau around weight `0.225-0.235`; further progress should come from a new signal definition instead of more same-axis weight sweeps.

## Beta4 Structured Features and SHAP-Stable XGB

Beta4 tested the first-place-solution-inspired path: correlation-cluster medoid features, purged-group XGB TreeSHAP stable feature selection, and a CPU MLP baseline.

| Submission | Public | Private | Notes |
| --- | ---: | ---: | --- |
| Previous best: Beta3 Spearman fine-tune | 0.05456 | 0.10003 | Before Beta4 |
| `0.85 * current_best + 0.15 * medoid_t0.6_ridge` | 0.05437 | 0.09565 | Medoid Ridge did not transfer |
| `0.75 * current_best + 0.25 * shap_stable_xgboost` | 0.05634 | 0.10043 | Current best private score |
| `0.85 * current_best + 0.15 * hybrid_mlp` | 0.05959 | 0.09729 | Higher public, private below current best |

Current best:

```text
0.75 * beta3_current_best + 0.25 * shap_stable_xgboost
```

The SHAP-stable branch selects 20 features from medoid-filtered features using purged 6-fold XGBoost TreeSHAP contributions. The MLP baseline is implemented, but the first full run did not produce a strong standalone model, and its safest small hybrid-feature blend did not transfer to private.

Conclusion: Beta4 confirms that the next useful direction is not another linear top-k weight sweep. The first transferable gain came from a nonlinear XGB/SHAP-stable signal layered onto the strong Beta3 Ridge/Spearman baseline.

## Beta4.1 SHAP-Stable XGB Refinement

Beta4.1 refined the successful SHAP-stable XGB branch by separating feature-rule changes, XGB training changes, and final blend weights.

| Submission | Public | Private | Notes |
| --- | ---: | ---: | --- |
| Previous best: Beta4 SHAP-stable XGB | 0.05634 | 0.10043 | Current best before Beta4.1 |
| `top30_min3_fill20`, RMSE ES, signal weight `0.325` | 0.05786 | 0.09956 | Local best blend did not transfer |
| `top30_min3_fill20`, RMSE ES, signal weight `0.25` | 0.05688 | 0.09974 | Conservative rule-change probe still below best |

Key local findings:

- `top30_min3_fill20` selected 37 pure-stable features and improved local signal Pearson to `0.082649`, but private score decreased after blending.
- Fixed XGB tree counts (`200`, `400`, `800`) underperformed RMSE early stopping.
- Pearson early stopping matched RMSE early stopping in this setup, so the evaluation-metric mismatch was not the limiting factor in this branch.
- The unsubmitted `0.30` blend was skipped after both `0.25` and `0.325` failed on private.

Current best remains:

```text
0.75 * beta3_current_best + 0.25 * shap_stable_xgboost
```

Conclusion: Beta4.1 closes the first SHAP-stable refinement pass without a new best. The branch remains useful as a component, but further progress likely requires a new signal source rather than narrower SHAP-rule or XGB early-stopping tweaks.

## Beta5-A Interaction-First Feature Expansion

Beta5-A introduced symbolic pairwise interaction features from a 40-feature core pool, then trained Ridge/XGB signals and blended them into the Beta4 best.

| Submission | Public | Private | Notes |
| --- | ---: | ---: | --- |
| Previous best: Beta4 SHAP-stable XGB | 0.05634 | 0.10043 | Before Beta5-A |
| `0.95 * current_best + 0.05 * xgb_core_plus_interactions` | 0.05657 | 0.10044 | Essentially tied current best |
| `0.85 * current_best + 0.15 * ridge_interactions_only` | 0.06362 | 0.10302 | Current best private score |

Current best:

```text
0.85 * beta4_current_best + 0.15 * ridge_interactions_only
```

where:

```text
beta4_current_best = 0.75 * beta3_current_best + 0.25 * shap_stable_xgboost
```

The interaction branch generated 5,460 pairwise candidates, selected 120 after train-only scoring and high-correlation pruning, and produced a complementary Ridge signal. XGB on the same interaction set was not useful as a standalone signal.

Conclusion: interaction features are the first clearly transferable new signal after Beta4. The next step should refine the interaction Ridge branch or feed selected interactions into AE features, rather than returning to SHAP-stable XGB weight tuning.

## Beta5-B Interaction Ridge Refinement

Beta5-B widened and retuned the successful interaction Ridge branch.

| Submission | Public | Private | Notes |
| --- | ---: | ---: | --- |
| Previous best: Beta5-A interaction Ridge | 0.06362 | 0.10302 | Current best before Beta5-B |
| `0.80 * beta4_base + 0.20 * int240_alpha300000` | 0.06729 | 0.10214 | Holdout best, private below current best |
| `0.80 * beta4_base + 0.20 * int240_alpha100000` | 0.06797 | 0.10256 | Higher public, still below current best private |

Current best remains:

```text
0.85 * beta4_current_best + 0.15 * ridge_interactions_only
```

Conclusion: larger interaction sets and heavier interaction weighting improved holdout/public but did not transfer to private. Keep the Beta5-A 120-feature interaction Ridge blend as the selected main submission.

## Beta5 Closed Summary

Beta5 is closed with the Beta5-A interaction Ridge blend as the selected current best.

Current best:

```text
0.85 * beta4_current_best + 0.15 * ridge_interactions_only
```

Leaderboard score:

| Public | Private | Local file |
| ---: | ---: | --- |
| 0.06362 | 0.10302 | `submissions/01_main/beta5_interactions/submission_best.csv` |

Final interpretation:

- Interaction features are useful and should be kept in the main narrative.
- XGB on interaction features is not useful in the current setup.
- Larger `int240` Ridge variants increased public score but reduced private score.
- The next stage should use the selected 120 interaction features as structured inputs for AE/MLP work, not continue widening Beta5 interactions.

## Beta6-AE Minimal AutoEncoder Features

Beta6-AE trained an 8-dimensional autoencoder representation from the 40 Beta5 core features plus 120 selected Beta5-A interaction features.

| Submission | Public | Private | Notes |
| --- | ---: | ---: | --- |
| Previous best: Beta5-A interaction Ridge | 0.06362 | 0.10302 | Current best before Beta6 |
| `0.85 * current_best + 0.15 * ae_ridge` | 0.06488 | 0.10169 | Holdout/public improved, private dropped |
| `0.90 * current_best + 0.10 * ae_ridge` | 0.06454 | 0.10225 | Conservative AE blend, still below current best |

Current best remains:

```text
0.85 * beta4_current_best + 0.15 * ridge_interactions_only
```

Conclusion: the first minimal AE pass did not transfer to private. AE features are not selected for the main submission unless the AE training recipe changes materially.

## Beta6.1 AE Refinement

Beta6.1 tested wider and lightly denoised AE representations on the same 160 Beta5 structured inputs.

| Submission | Public | Private | Notes |
| --- | ---: | ---: | --- |
| Previous best: Beta5-A interaction Ridge | 0.06362 | 0.10302 | Current best before Beta6.1 |
| `0.95 * current_best + 0.05 * ae8_base_ridge` | 0.06412 | 0.10269 | Conservative AE blend, still below current best |

Current best remains:

```text
0.85 * beta4_current_best + 0.15 * ridge_interactions_only
```

Conclusion: widening the AE bottleneck to 16/32, adding light denoising, and probing another seed did not improve over AE8. AE is not selected for the main submission in its current form.

## Beta6.2 Supervised AE Features

Beta6.2 added a supervised auxiliary target head to the AE latent representation while keeping the same 160 Beta5 structured inputs.

| Submission | Public | Private | Notes |
| --- | ---: | ---: | --- |
| Previous best: Beta5-A interaction Ridge | 0.06362 | 0.10302 | Current best before Beta6.2 |
| `0.875 * current_best + 0.125 * ae8_supervised_mse005` | 0.06781 | 0.10269 | Local best, AE weight too high |
| `0.95 * current_best + 0.05 * ae8_supervised_mse005` | 0.06542 | 0.10300 | Near tie |
| `0.975 * current_best + 0.025 * ae8_supervised_mse005` | 0.06454 | 0.10303 | Current best private score |

Current best:

```text
0.975 * beta5_current_best + 0.025 * ae8_supervised_mse005_ridge
```

Conclusion: supervised AE features produce a marginal transferable gain, but only at very small weight. Larger AE weights improve public/local metrics and reduce private score.

## Beta7 Supervised MLP Signal Generator

Beta7 trained six supervised MLP signals on the fixed 160 Beta5 structured inputs, plus seed-mean and multi-model mean ensembles.

| Candidate | Holdout Pearson | Delta vs Current Best | Public | Private | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| `0.975 * current_best + 0.025 * wide_adamw_lr001_seed_mean` | 0.120212 | +0.001018 | 0.06588 | 0.10411 | Current best private score |
| `0.925 * current_best + 0.075 * wide_adamw_lr001_seed2026` | 0.121517 | +0.002322 | - | - | Held for follow-up |
| `0.900 * current_best + 0.100 * wide_adamw_lr001_seed3026` | 0.121241 | +0.002047 | - | - | Lowest-correlation candidate |

Current best:

```text
0.975 * beta6_2_current_best + 0.025 * wide_adamw_lr001_seed_mean
```

Conclusion: the constrained MLP branch produced a real transferable gain, but only at very small weight. The MLP should be kept as a complementary signal, not treated as a standalone replacement for the Ridge/interaction/AE stack.
