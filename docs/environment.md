# Environment

Use a dedicated Python environment with the dependencies from `requirements.txt`.

Do not install project dependencies into the system Python environment. Do not create an in-project `.venv` unless the environment policy changes.

After activating the environment, install dependencies from the repository root:

```powershell
python.exe -m pip install -r requirements.txt
```

Run project scripts through the same interpreter:

```powershell
python.exe scripts/run_main_baseline.py --config configs/01_main_baseline_full.yaml
```

Verified core packages:

| Package | Version |
| --- | --- |
| Python | 3.13.9 |
| numpy | 2.2.6 |
| pandas | 3.0.3 |
| scipy | 1.17.1 |
| pyarrow | 24.0.0 |
| PyYAML | 6.0.3 |
| scikit-learn | 1.9.0 |
| lightgbm | 4.6.0 |
| xgboost | 3.2.0 |
| catboost | 1.2.10 |
| matplotlib | 3.11.0 |
| tqdm | 4.68.2 |
| torch | 2.12.1+cpu |
| shap | 0.52.0 |

Beta4 uses CPU-only PyTorch. The verified environment reports `torch.cuda.is_available() == False`, which is expected for this project.
