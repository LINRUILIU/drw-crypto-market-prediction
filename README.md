# DRW Crypto Market Prediction Modeling

This repository is organized as a reproducible course-project pipeline for the Kaggle DRW Crypto Market Prediction task and its extension analyses.

## Directory Layout

```text
configs/                 Experiment configuration files, grouped by task number
data/raw/                Kaggle input files, not committed
data/processed/          Derived intermediate data, not committed
docs/                    Report notes and audit records
reports/report.md        Final Markdown report
reports/figures/         Stable figures used by the report and PPT
runs/01_main/            Main prediction task outputs
runs/02_feature/         Feature selection and dimensionality outputs
runs/03_temporal/        Temporal stability outputs
runs/04_signal/          Signal interpretation outputs
scripts/                 Command-line entry points
src/drw_crypto/          Shared reusable project code
submissions/01_main/     Kaggle submission files for the main task
```

## Expected Data Files

Place Kaggle files under `data/raw/`:

```text
data/raw/train.parquet
data/raw/test.parquet
data/raw/sample_submission.csv
```

CSV files with the same base names are also supported. The baseline runner will auto-detect common file names.

## Final Report

The polished Markdown report is `reports/report.md`. It references stable figures under `reports/figures/` and supporting evidence under `docs/`.

## Main Baseline

Activate your Python environment and run from the repository root:

```powershell
python.exe scripts/run_main_baseline.py --config configs/01_main_baseline_full.yaml
```

The runner always supports a NumPy Ridge baseline. ElasticNet, LightGBM, XGBoost, and CatBoost are enabled when their packages are installed.

## Pearson Top-k Optimization

```powershell
python.exe scripts/run_pearson_topk.py --config configs/01_main_pearson_topk.yaml
```

This trains Ridge, LightGBM, and their weighted ensemble on train-only Pearson top-k feature sets.
