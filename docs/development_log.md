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
