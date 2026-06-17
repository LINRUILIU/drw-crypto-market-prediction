from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drw_crypto.io import (  # noqa: E402
    SAMPLE_SUBMISSION_CANDIDATES,
    TEST_CANDIDATES,
    TRAIN_CANDIDATES,
    ensure_dir,
    find_first_existing,
    read_table,
    write_json,
    write_lines,
)
from drw_crypto.metrics import pearson_corr, rmse  # noqa: E402
from drw_crypto.models import NumpyRidgeRegressor  # noqa: E402
from drw_crypto.preprocessing import (  # noqa: E402
    TabularPreprocessor,
    infer_feature_columns,
    infer_id_column,
    infer_target_column,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run main DRW crypto baseline models.")
    parser.add_argument("--config", default="configs/main_baseline.yaml", help="Path to YAML config.")
    parser.add_argument("--max-train-rows", type=int, default=None, help="Optional tail sample for quick tests.")
    parser.add_argument("--no-final", action="store_true", help="Skip fitting final models and submission output.")
    parser.add_argument(
        "--only-models",
        default=None,
        help="Comma-separated model subset, e.g. ridge,lightgbm. Ensembles use available predictions.",
    )
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def has_module(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def split_time_ordered(df: pd.DataFrame, validation_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")
    split_at = int(len(df) * (1.0 - validation_fraction))
    if split_at <= 0 or split_at >= len(df):
        raise ValueError("Validation split produced an empty train or validation set.")
    return df.iloc[:split_at], df.iloc[split_at:]


def choose_prediction_column(
    sample_submission: pd.DataFrame | None,
    configured: str | None,
    target_col: str,
) -> str:
    if configured:
        return configured
    if sample_submission is None:
        return "prediction"
    if target_col in sample_submission.columns:
        return target_col
    if "prediction" in sample_submission.columns:
        return "prediction"
    if len(sample_submission.columns) >= 2:
        return str(sample_submission.columns[-1])
    return str(sample_submission.columns[0])


def save_valid_predictions(path: Path, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "row_index": np.arange(len(y_true), dtype=np.int64),
            "y_true": y_true,
            "prediction": y_pred,
        }
    ).to_csv(path, index=False)


def make_submission(
    sample_submission: pd.DataFrame | None,
    test_df: pd.DataFrame,
    pred: np.ndarray,
    prediction_col: str,
    id_col: str | None,
) -> pd.DataFrame:
    if sample_submission is not None:
        if len(sample_submission) != len(pred):
            raise ValueError(
                f"Sample submission has {len(sample_submission)} rows, but prediction has {len(pred)} rows."
            )
        submission = sample_submission.copy()
        if prediction_col not in submission.columns:
            raise ValueError(
                f"Prediction column {prediction_col!r} is not in sample submission columns: "
                f"{list(submission.columns)}"
            )
        submission[prediction_col] = pred
        return submission

    if id_col and id_col in test_df.columns:
        return pd.DataFrame({id_col: test_df[id_col].to_numpy(), prediction_col: pred})
    return pd.DataFrame({"row_id": np.arange(len(pred), dtype=np.int64), prediction_col: pred})


def train_ridge_search(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    alphas: list[float],
    out_dir: Path,
) -> tuple[dict[str, Any], np.ndarray]:
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_pred: np.ndarray | None = None

    for alpha in alphas:
        start = time.perf_counter()
        model = NumpyRidgeRegressor(alpha=float(alpha)).fit(x_train, y_train)
        pred = model.predict(x_valid)
        elapsed = time.perf_counter() - start
        row = {
            "model": "ridge",
            "alpha": float(alpha),
            "pearson": pearson_corr(y_valid, pred),
            "rmse": rmse(y_valid, pred),
            "train_seconds": elapsed,
        }
        rows.append(row)
        if best is None or row["pearson"] > best["pearson"]:
            best = row
            best_pred = pred

    pd.DataFrame(rows).to_csv(out_dir / "ridge_alpha_search.csv", index=False)
    assert best is not None and best_pred is not None
    return best, best_pred


def train_elasticnet(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    params: dict[str, Any],
) -> tuple[dict[str, Any], np.ndarray] | None:
    if not has_module("sklearn"):
        return None

    from sklearn.linear_model import ElasticNet

    start = time.perf_counter()
    model = ElasticNet(
        alpha=float(params.get("alpha", 0.0005)),
        l1_ratio=float(params.get("l1_ratio", 0.2)),
        max_iter=int(params.get("max_iter", 3000)),
        random_state=42,
    )
    model.fit(x_train, y_train)
    pred = model.predict(x_valid)
    elapsed = time.perf_counter() - start
    return (
        {
            "model": "elasticnet",
            "pearson": pearson_corr(y_valid, pred),
            "rmse": rmse(y_valid, pred),
            "train_seconds": elapsed,
            "params": json.dumps(params, sort_keys=True),
        },
        pred,
    )


def train_lightgbm(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    params: dict[str, Any],
) -> tuple[dict[str, Any], np.ndarray, dict[str, Any]] | None:
    if not has_module("lightgbm"):
        return None

    import lightgbm as lgb

    def lgb_pearson_eval(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[str, float, bool]:
        return "pearson", pearson_corr(y_true, y_pred), True

    model_params = {
        "n_estimators": int(params.get("n_estimators", 2000)),
        "learning_rate": float(params.get("learning_rate", 0.03)),
        "num_leaves": int(params.get("num_leaves", 63)),
        "subsample": float(params.get("subsample", 0.85)),
        "colsample_bytree": float(params.get("colsample_bytree", 0.85)),
        "reg_alpha": float(params.get("reg_alpha", 0.0)),
        "reg_lambda": float(params.get("reg_lambda", 1.0)),
        "objective": "regression",
        "metric": "None",
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": -1,
    }
    callbacks = [
        lgb.early_stopping(int(params.get("early_stopping_rounds", 100)), verbose=False),
        lgb.log_evaluation(period=100),
    ]

    start = time.perf_counter()
    model = lgb.LGBMRegressor(**model_params)
    model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], eval_metric=lgb_pearson_eval, callbacks=callbacks)
    pred = model.predict(x_valid)
    elapsed = time.perf_counter() - start

    best_iter = getattr(model, "best_iteration_", None)
    final_params = dict(model_params)
    if best_iter:
        final_params["n_estimators"] = int(best_iter)

    return (
        {
            "model": "lightgbm",
            "pearson": pearson_corr(y_valid, pred),
            "rmse": rmse(y_valid, pred),
            "train_seconds": elapsed,
            "best_iteration": best_iter,
            "params": json.dumps(final_params, sort_keys=True),
        },
        pred,
        final_params,
    )


def train_xgboost(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    params: dict[str, Any],
) -> tuple[dict[str, Any], np.ndarray, dict[str, Any]] | None:
    if not has_module("xgboost"):
        return None

    import xgboost as xgb

    model_params = {
        "n_estimators": int(params.get("n_estimators", 1200)),
        "learning_rate": float(params.get("learning_rate", 0.03)),
        "max_depth": int(params.get("max_depth", 6)),
        "subsample": float(params.get("subsample", 0.85)),
        "colsample_bytree": float(params.get("colsample_bytree", 0.85)),
        "reg_lambda": float(params.get("reg_lambda", 1.0)),
        "objective": "reg:squarederror",
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",
        "eval_metric": "rmse",
    }
    early_stopping_rounds = int(params.get("early_stopping_rounds", 100))
    if early_stopping_rounds > 0:
        model_params["early_stopping_rounds"] = early_stopping_rounds

    start = time.perf_counter()
    model = xgb.XGBRegressor(**model_params)
    try:
        model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=False)
    except TypeError:
        model_params.pop("early_stopping_rounds", None)
        model = xgb.XGBRegressor(**model_params)
        model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=False)
    pred = model.predict(x_valid)
    elapsed = time.perf_counter() - start

    best_iter = getattr(model, "best_iteration", None) if "early_stopping_rounds" in model_params else None
    final_params = dict(model_params)
    if best_iter is not None:
        final_params["n_estimators"] = int(best_iter) + 1
        final_params.pop("early_stopping_rounds", None)

    return (
        {
            "model": "xgboost",
            "pearson": pearson_corr(y_valid, pred),
            "rmse": rmse(y_valid, pred),
            "train_seconds": elapsed,
            "best_iteration": best_iter,
            "params": json.dumps(final_params, sort_keys=True),
        },
        pred,
        final_params,
    )


def train_catboost(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    params: dict[str, Any],
) -> tuple[dict[str, Any], np.ndarray, dict[str, Any]] | None:
    if not has_module("catboost"):
        return None

    from catboost import CatBoostRegressor

    model_params = {
        "iterations": int(params.get("iterations", 1500)),
        "learning_rate": float(params.get("learning_rate", 0.03)),
        "depth": int(params.get("depth", 6)),
        "l2_leaf_reg": float(params.get("l2_leaf_reg", 3.0)),
        "loss_function": "RMSE",
        "random_seed": 42,
        "allow_writing_files": False,
        "verbose": 100,
    }

    start = time.perf_counter()
    model = CatBoostRegressor(**model_params)
    early_stopping_rounds = int(params.get("early_stopping_rounds", 100))
    if early_stopping_rounds > 0:
        model.fit(
            x_train,
            y_train,
            eval_set=(x_valid, y_valid),
            use_best_model=True,
            early_stopping_rounds=early_stopping_rounds,
        )
    else:
        model.fit(x_train, y_train)
    pred = model.predict(x_valid)
    elapsed = time.perf_counter() - start

    best_iter = model.get_best_iteration() if early_stopping_rounds > 0 else None
    final_params = dict(model_params)
    if best_iter is not None:
        final_params["iterations"] = int(best_iter) + 1
    final_params["verbose"] = False

    return (
        {
            "model": "catboost",
            "pearson": pearson_corr(y_valid, pred),
            "rmse": rmse(y_valid, pred),
            "train_seconds": elapsed,
            "best_iteration": best_iter,
            "params": json.dumps(final_params, sort_keys=True),
        },
        pred,
        final_params,
    )


def fit_final_model(name: str, params: dict[str, Any], x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    if name == "ridge":
        return NumpyRidgeRegressor(alpha=float(params["alpha"])).fit(x_train, y_train).predict(x_test)
    if name == "elasticnet":
        from sklearn.linear_model import ElasticNet

        model = ElasticNet(
            alpha=float(params.get("alpha", 0.0005)),
            l1_ratio=float(params.get("l1_ratio", 0.2)),
            max_iter=int(params.get("max_iter", 3000)),
            random_state=42,
        )
        model.fit(x_train, y_train)
        return model.predict(x_test)
    if name == "lightgbm":
        import lightgbm as lgb

        model = lgb.LGBMRegressor(**params)
        model.fit(x_train, y_train)
        return model.predict(x_test)
    if name == "xgboost":
        import xgboost as xgb

        params = dict(params)
        params.pop("early_stopping_rounds", None)
        model = xgb.XGBRegressor(**params)
        model.fit(x_train, y_train, verbose=False)
        return model.predict(x_test)
    if name == "catboost":
        from catboost import CatBoostRegressor

        params = dict(params)
        params["verbose"] = False
        model = CatBoostRegressor(**params)
        model.fit(x_train, y_train)
        return model.predict(x_test)
    raise ValueError(f"Unsupported final model: {name}")


def main() -> int:
    args = parse_args()
    config = load_config(ROOT / args.config if not Path(args.config).is_absolute() else args.config)

    data_cfg = config["data"]
    output_cfg = config["output"]
    model_cfg = config["models"]
    prep_cfg = config["preprocessing"]

    if args.only_models:
        selected_models = {name.strip().lower() for name in args.only_models.split(",") if name.strip()}
        for model_name, params in model_cfg.items():
            params["enabled"] = model_name.lower() in selected_models

    run_dir = ensure_dir(ROOT / output_cfg["run_dir"])
    pred_dir = ensure_dir(run_dir / "valid_predictions")
    submission_dir = ensure_dir(ROOT / output_cfg["submission_dir"])

    raw_dir = ROOT / data_cfg["raw_dir"]
    train_path = find_first_existing(raw_dir, data_cfg.get("train_file"), TRAIN_CANDIDATES)
    test_path = find_first_existing(raw_dir, data_cfg.get("test_file"), TEST_CANDIDATES)
    sample_path = find_first_existing(
        raw_dir,
        data_cfg.get("sample_submission_file"),
        SAMPLE_SUBMISSION_CANDIDATES,
    )

    if train_path is None or (test_path is None and not args.no_final):
        raise FileNotFoundError(
            "Missing train/test files. Put Kaggle data under data/raw/, for example "
            "data/raw/train.parquet and data/raw/test.parquet."
        )

    train_df = read_table(train_path)
    test_df = read_table(test_path) if test_path is not None and not args.no_final else None
    sample_submission = read_table(sample_path) if sample_path is not None else None

    max_train_rows = args.max_train_rows if args.max_train_rows is not None else data_cfg.get("max_train_rows")
    if max_train_rows and max_train_rows < len(train_df):
        train_df = train_df.tail(int(max_train_rows)).reset_index(drop=True)

    target_col = infer_target_column(train_df, test_df, data_cfg.get("target_col"))
    id_col = infer_id_column(train_df, test_df, sample_submission, target_col, data_cfg.get("id_col"))
    feature_cols = infer_feature_columns(
        train_df,
        target_col=target_col,
        id_col=id_col,
        drop_columns=data_cfg.get("drop_columns") or [],
        numeric_only=bool(prep_cfg.get("numeric_only", True)),
    )
    prediction_col = choose_prediction_column(sample_submission, output_cfg.get("prediction_col"), target_col)

    write_json(
        run_dir / "data_summary.json",
        {
            "train_path": str(train_path.relative_to(ROOT)),
            "test_path": str(test_path.relative_to(ROOT)) if test_path else None,
            "sample_submission_path": str(sample_path.relative_to(ROOT)) if sample_path else None,
            "train_shape": list(train_df.shape),
            "test_shape": list(test_df.shape) if test_df is not None else None,
            "target_col": target_col,
            "id_col": id_col,
            "prediction_col": prediction_col,
            "feature_count": len(feature_cols),
            "validation_fraction": data_cfg["validation_fraction"],
            "max_train_rows": max_train_rows,
        },
    )
    write_lines(run_dir / "feature_columns.txt", feature_cols)
    write_json(run_dir / "resolved_config.json", config)

    train_part, valid_part = split_time_ordered(train_df, float(data_cfg["validation_fraction"]))
    y_train = train_part[target_col].to_numpy(dtype=np.float64)
    y_valid = valid_part[target_col].to_numpy(dtype=np.float64)

    preprocessor = TabularPreprocessor(feature_cols).fit(
        train_part,
        missing_fill=str(prep_cfg.get("missing_fill", "median")),
        standardize=bool(prep_cfg.get("standardize", True)),
    )
    x_train = preprocessor.transform(train_part)
    x_valid = preprocessor.transform(valid_part)

    metrics_rows: list[dict[str, Any]] = []
    valid_predictions: dict[str, np.ndarray] = {}
    final_params_by_model: dict[str, dict[str, Any]] = {}
    skipped: dict[str, str] = {}

    if model_cfg["ridge"].get("enabled", True):
        ridge_best, ridge_pred = train_ridge_search(
            x_train,
            y_train,
            x_valid,
            y_valid,
            [float(alpha) for alpha in model_cfg["ridge"].get("alphas", [1.0])],
            run_dir,
        )
        metrics_rows.append(
            {
                "model": "ridge",
                "pearson": ridge_best["pearson"],
                "rmse": ridge_best["rmse"],
                "train_seconds": ridge_best["train_seconds"],
                "n_features": len(feature_cols),
                "notes": f"alpha={ridge_best['alpha']}",
            }
        )
        valid_predictions["ridge"] = ridge_pred
        final_params_by_model["ridge"] = {"alpha": ridge_best["alpha"]}
        save_valid_predictions(pred_dir / "valid_ridge.csv", y_valid, ridge_pred)

    if model_cfg["elasticnet"].get("enabled", True):
        result = train_elasticnet(x_train, y_train, x_valid, y_valid, model_cfg["elasticnet"])
        if result is None:
            skipped["elasticnet"] = "scikit-learn is not installed"
        else:
            row, pred = result
            row["n_features"] = len(feature_cols)
            row["notes"] = row.pop("params")
            metrics_rows.append(row)
            valid_predictions["elasticnet"] = pred
            final_params_by_model["elasticnet"] = dict(model_cfg["elasticnet"])
            save_valid_predictions(pred_dir / "valid_elasticnet.csv", y_valid, pred)

    if model_cfg["lightgbm"].get("enabled", True):
        result = train_lightgbm(x_train, y_train, x_valid, y_valid, model_cfg["lightgbm"])
        if result is None:
            skipped["lightgbm"] = "lightgbm is not installed"
        else:
            row, pred, final_params = result
            row["n_features"] = len(feature_cols)
            row["notes"] = row.pop("params")
            metrics_rows.append(row)
            valid_predictions["lightgbm"] = pred
            final_params_by_model["lightgbm"] = final_params
            save_valid_predictions(pred_dir / "valid_lightgbm.csv", y_valid, pred)

    if model_cfg["xgboost"].get("enabled", True):
        result = train_xgboost(x_train, y_train, x_valid, y_valid, model_cfg["xgboost"])
        if result is None:
            skipped["xgboost"] = "xgboost is not installed"
        else:
            row, pred, final_params = result
            row["n_features"] = len(feature_cols)
            row["notes"] = row.pop("params")
            metrics_rows.append(row)
            valid_predictions["xgboost"] = pred
            final_params_by_model["xgboost"] = final_params
            save_valid_predictions(pred_dir / "valid_xgboost.csv", y_valid, pred)

    if model_cfg["catboost"].get("enabled", True):
        result = train_catboost(x_train, y_train, x_valid, y_valid, model_cfg["catboost"])
        if result is None:
            skipped["catboost"] = "catboost is not installed"
        else:
            row, pred, final_params = result
            row["n_features"] = len(feature_cols)
            row["notes"] = row.pop("params")
            metrics_rows.append(row)
            valid_predictions["catboost"] = pred
            final_params_by_model["catboost"] = final_params
            save_valid_predictions(pred_dir / "valid_catboost.csv", y_valid, pred)

    if config.get("ensemble", {}).get("enabled", True) and len(valid_predictions) >= 2:
        mean_pred = np.mean(np.column_stack(list(valid_predictions.values())), axis=1)
        metrics_rows.append(
            {
                "model": "ensemble_mean_all",
                "pearson": pearson_corr(y_valid, mean_pred),
                "rmse": rmse(y_valid, mean_pred),
                "train_seconds": 0.0,
                "n_features": len(feature_cols),
                "notes": ",".join(valid_predictions.keys()),
            }
        )
        valid_predictions["ensemble_mean_all"] = mean_pred
        save_valid_predictions(pred_dir / "valid_ensemble_mean_all.csv", y_valid, mean_pred)

        if "ridge" in valid_predictions and "lightgbm" in valid_predictions:
            step = float(config["ensemble"].get("ridge_lgbm_grid_step", 0.05))
            weights = np.arange(0.0, 1.0 + step / 2.0, step)
            best_weight = 0.0
            best_score = -np.inf
            best_pred = None
            grid_rows = []
            for weight in weights:
                pred = weight * valid_predictions["ridge"] + (1.0 - weight) * valid_predictions["lightgbm"]
                score = pearson_corr(y_valid, pred)
                grid_rows.append({"ridge_weight": weight, "pearson": score, "rmse": rmse(y_valid, pred)})
                if score > best_score:
                    best_weight = float(weight)
                    best_score = score
                    best_pred = pred
            pd.DataFrame(grid_rows).to_csv(run_dir / "ensemble_ridge_lgbm_grid.csv", index=False)
            assert best_pred is not None
            metrics_rows.append(
                {
                    "model": "ensemble_ridge_lgbm",
                    "pearson": pearson_corr(y_valid, best_pred),
                    "rmse": rmse(y_valid, best_pred),
                    "train_seconds": 0.0,
                    "n_features": len(feature_cols),
                    "notes": f"ridge_weight={best_weight:.4f}",
                }
            )
            valid_predictions["ensemble_ridge_lgbm"] = best_pred
            save_valid_predictions(pred_dir / "valid_ensemble_ridge_lgbm.csv", y_valid, best_pred)

    metrics_df = pd.DataFrame(metrics_rows).sort_values("pearson", ascending=False)
    metrics_df.to_csv(run_dir / "metrics_main_models.csv", index=False)
    write_json(run_dir / "skipped_models.json", skipped)

    if args.no_final:
        print(metrics_df.to_string(index=False))
        print(f"\nSkipped models: {skipped}")
        return 0

    full_preprocessor = TabularPreprocessor(feature_cols).fit(
        train_df,
        missing_fill=str(prep_cfg.get("missing_fill", "median")),
        standardize=bool(prep_cfg.get("standardize", True)),
    )
    x_full = full_preprocessor.transform(train_df)
    y_full = train_df[target_col].to_numpy(dtype=np.float64)
    x_test = full_preprocessor.transform(test_df)

    test_predictions: dict[str, np.ndarray] = {}
    for model_name, params in final_params_by_model.items():
        if model_name in {"elasticnet", "lightgbm", "xgboost", "catboost"} and not has_module(
            "sklearn" if model_name == "elasticnet" else model_name
        ):
            continue
        test_predictions[model_name] = fit_final_model(model_name, params, x_full, y_full, x_test)

    if "ensemble_mean_all" in valid_predictions and len(test_predictions) >= 2:
        test_predictions["ensemble_mean_all"] = np.mean(np.column_stack(list(test_predictions.values())), axis=1)

    if "ensemble_ridge_lgbm" in valid_predictions and {"ridge", "lightgbm"}.issubset(test_predictions):
        note = metrics_df.loc[metrics_df["model"] == "ensemble_ridge_lgbm", "notes"].iloc[0]
        ridge_weight = float(str(note).split("=")[1])
        test_predictions["ensemble_ridge_lgbm"] = (
            ridge_weight * test_predictions["ridge"] + (1.0 - ridge_weight) * test_predictions["lightgbm"]
        )

    best_model = str(metrics_df.iloc[0]["model"])
    if best_model not in test_predictions:
        best_model = next(iter(test_predictions))

    for model_name, pred in test_predictions.items():
        submission = make_submission(sample_submission, test_df, pred, prediction_col, id_col)
        submission.to_csv(submission_dir / f"submission_{model_name}.csv", index=False)

    best_submission = make_submission(sample_submission, test_df, test_predictions[best_model], prediction_col, id_col)
    best_submission.to_csv(submission_dir / "submission_best.csv", index=False)

    write_json(
        run_dir / "final_submission_summary.json",
        {
            "best_validation_model": best_model,
            "submission_files": sorted(str(path.relative_to(ROOT)) for path in submission_dir.glob("submission_*.csv")),
        },
    )

    print(metrics_df.to_string(index=False))
    print(f"\nBest validation model: {best_model}")
    print(f"Submission directory: {submission_dir.relative_to(ROOT)}")
    print(f"Skipped models: {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
