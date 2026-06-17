from __future__ import annotations

import argparse
import gc
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

from drw_crypto.feature_selection import pearson_feature_ranking  # noqa: E402
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
    parser = argparse.ArgumentParser(description="Run Pearson top-k Ridge/LightGBM experiments.")
    parser.add_argument("--config", default="configs/01_main_pearson_topk.yaml", help="Path to YAML config.")
    parser.add_argument("--max-train-rows", type=int, default=None, help="Optional tail sample for smoke runs.")
    parser.add_argument("--no-final", action="store_true", help="Skip fitting final model and submission output.")
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def split_time_ordered(df: pd.DataFrame, validation_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    return str(sample_submission.columns[-1])


def train_ridge_search(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    alphas: list[float],
) -> tuple[dict[str, Any], np.ndarray, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_pred: np.ndarray | None = None

    for alpha in alphas:
        start = time.perf_counter()
        model = NumpyRidgeRegressor(alpha=alpha).fit(x_train, y_train)
        pred = model.predict(x_valid)
        elapsed = time.perf_counter() - start
        row = {
            "alpha": alpha,
            "pearson": pearson_corr(y_valid, pred),
            "rmse": rmse(y_valid, pred),
            "train_seconds": elapsed,
        }
        rows.append(row)
        if best is None or row["pearson"] > best["pearson"]:
            best = row
            best_pred = pred

    assert best is not None and best_pred is not None
    return best, best_pred, pd.DataFrame(rows)


def train_lightgbm(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    params: dict[str, Any],
) -> tuple[dict[str, Any], np.ndarray, dict[str, Any]]:
    import lightgbm as lgb

    def lgb_pearson_eval(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[str, float, bool]:
        return "pearson", pearson_corr(y_true, y_pred), True

    model_params = {
        "n_estimators": int(params.get("n_estimators", 1200)),
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

    row = {
        "pearson": pearson_corr(y_valid, pred),
        "rmse": rmse(y_valid, pred),
        "train_seconds": elapsed,
        "best_iteration": best_iter,
    }
    return row, pred, final_params


def save_valid_prediction(path: Path, y_true: np.ndarray, y_pred: np.ndarray) -> None:
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
            raise ValueError("Prediction length does not match sample submission length.")
        submission = sample_submission.copy()
        submission[prediction_col] = pred
        return submission
    if id_col and id_col in test_df.columns:
        return pd.DataFrame({id_col: test_df[id_col].to_numpy(), prediction_col: pred})
    return pd.DataFrame({"row_id": np.arange(len(pred), dtype=np.int64), prediction_col: pred})


def fit_ridge(alpha: float, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    return NumpyRidgeRegressor(alpha=alpha).fit(x_train, y_train).predict(x_test)


def fit_lightgbm(params: dict[str, Any], x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    import lightgbm as lgb

    model = lgb.LGBMRegressor(**params)
    model.fit(x_train, y_train)
    return model.predict(x_test)


def main() -> int:
    args = parse_args()
    config_path = ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)
    config = load_config(config_path)

    data_cfg = config["data"]
    prep_cfg = config["preprocessing"]
    feature_cfg = config["feature_selection"]
    model_cfg = config["models"]
    output_cfg = config["output"]

    run_dir = ensure_dir(ROOT / output_cfg["run_dir"])
    pred_dir = ensure_dir(run_dir / "valid_predictions")
    submission_dir = ensure_dir(ROOT / output_cfg["submission_dir"])

    raw_dir = ROOT / data_cfg["raw_dir"]
    train_path = find_first_existing(raw_dir, data_cfg.get("train_file"), TRAIN_CANDIDATES)
    test_path = find_first_existing(raw_dir, data_cfg.get("test_file"), TEST_CANDIDATES)
    sample_path = find_first_existing(raw_dir, data_cfg.get("sample_submission_file"), SAMPLE_SUBMISSION_CANDIDATES)
    if train_path is None:
        raise FileNotFoundError("Missing train file under data/raw.")

    train_df = read_table(train_path)
    sample_submission = read_table(sample_path) if sample_path is not None else None

    max_train_rows = args.max_train_rows if args.max_train_rows is not None else data_cfg.get("max_train_rows")
    if max_train_rows and max_train_rows < len(train_df):
        train_df = train_df.tail(int(max_train_rows)).reset_index(drop=True)

    target_col = infer_target_column(train_df, None, data_cfg.get("target_col"))
    id_col = infer_id_column(train_df, None, sample_submission, target_col, data_cfg.get("id_col"))
    feature_cols = infer_feature_columns(
        train_df,
        target_col=target_col,
        id_col=id_col,
        drop_columns=data_cfg.get("drop_columns") or [],
        numeric_only=bool(prep_cfg.get("numeric_only", True)),
    )
    prediction_col = choose_prediction_column(sample_submission, output_cfg.get("prediction_col"), target_col)

    train_part, valid_part = split_time_ordered(train_df, float(data_cfg["validation_fraction"]))
    y_train = train_part[target_col].to_numpy(dtype=np.float64)
    y_valid = valid_part[target_col].to_numpy(dtype=np.float64)

    ranking = pearson_feature_ranking(train_part, feature_cols, target_col)
    ranking.to_csv(run_dir / "pearson_feature_ranking.csv", index=False)

    top_k_values = [int(k) for k in feature_cfg.get("top_k", [])]
    schemes: list[tuple[str, list[str]]] = []
    for k in top_k_values:
        selected = ranking["feature"].head(min(k, len(feature_cols))).tolist()
        scheme = f"top{k}"
        write_lines(run_dir / f"selected_features_{scheme}.txt", selected)
        schemes.append((scheme, selected))
    if bool(feature_cfg.get("include_full", True)):
        write_lines(run_dir / "selected_features_full.txt", feature_cols)
        schemes.append(("full", feature_cols))

    write_json(
        run_dir / "data_summary.json",
        {
            "train_path": str(train_path.relative_to(ROOT)),
            "test_path": str(test_path.relative_to(ROOT)) if test_path else None,
            "sample_submission_path": str(sample_path.relative_to(ROOT)) if sample_path else None,
            "train_shape": list(train_df.shape),
            "target_col": target_col,
            "id_col": id_col,
            "prediction_col": prediction_col,
            "feature_count": len(feature_cols),
            "validation_fraction": data_cfg["validation_fraction"],
            "max_train_rows": max_train_rows,
            "ranking_fit_scope": "train_split_only",
        },
    )
    write_json(run_dir / "resolved_config.json", config)

    metrics_rows: list[dict[str, Any]] = []
    final_state_by_scheme: dict[str, dict[str, Any]] = {}

    for scheme, selected_features in schemes:
        print(f"\n=== Running scheme: {scheme} ({len(selected_features)} features) ===")
        preprocessor = TabularPreprocessor(selected_features).fit(
            train_part,
            missing_fill=str(prep_cfg.get("missing_fill", "median")),
            standardize=bool(prep_cfg.get("standardize", True)),
        )
        x_train = preprocessor.transform(train_part)
        x_valid = preprocessor.transform(valid_part)

        ridge_best, ridge_pred, ridge_grid = train_ridge_search(
            x_train,
            y_train,
            x_valid,
            y_valid,
            [float(alpha) for alpha in model_cfg["ridge"].get("alphas", [1.0])],
        )
        ridge_grid.insert(0, "scheme", scheme)
        ridge_grid.to_csv(run_dir / f"ridge_alpha_search_{scheme}.csv", index=False)
        save_valid_prediction(pred_dir / f"valid_{scheme}_ridge.csv", y_valid, ridge_pred)
        metrics_rows.append(
            {
                "scheme": scheme,
                "feature_count": len(selected_features),
                "model": "ridge",
                "pearson": ridge_best["pearson"],
                "rmse": ridge_best["rmse"],
                "train_seconds": ridge_best["train_seconds"],
                "best_iteration": "",
                "notes": f"alpha={ridge_best['alpha']}",
            }
        )

        lgbm_row, lgbm_pred, lgbm_params = train_lightgbm(
            x_train,
            y_train,
            x_valid,
            y_valid,
            model_cfg["lightgbm"],
        )
        save_valid_prediction(pred_dir / f"valid_{scheme}_lightgbm.csv", y_valid, lgbm_pred)
        metrics_rows.append(
            {
                "scheme": scheme,
                "feature_count": len(selected_features),
                "model": "lightgbm",
                "pearson": lgbm_row["pearson"],
                "rmse": lgbm_row["rmse"],
                "train_seconds": lgbm_row["train_seconds"],
                "best_iteration": lgbm_row["best_iteration"],
                "notes": json.dumps(lgbm_params, sort_keys=True),
            }
        )

        step = float(config["ensemble"].get("ridge_lgbm_grid_step", 0.05))
        grid_rows: list[dict[str, Any]] = []
        best_weight = 0.0
        best_score = -np.inf
        best_pred: np.ndarray | None = None
        for weight in np.arange(0.0, 1.0 + step / 2.0, step):
            pred = weight * ridge_pred + (1.0 - weight) * lgbm_pred
            score = pearson_corr(y_valid, pred)
            grid_rows.append(
                {
                    "scheme": scheme,
                    "ridge_weight": float(weight),
                    "pearson": score,
                    "rmse": rmse(y_valid, pred),
                }
            )
            if score > best_score:
                best_weight = float(weight)
                best_score = score
                best_pred = pred

        assert best_pred is not None
        pd.DataFrame(grid_rows).to_csv(run_dir / f"ensemble_ridge_lgbm_grid_{scheme}.csv", index=False)
        save_valid_prediction(pred_dir / f"valid_{scheme}_ensemble_ridge_lgbm.csv", y_valid, best_pred)
        metrics_rows.append(
            {
                "scheme": scheme,
                "feature_count": len(selected_features),
                "model": "ensemble_ridge_lgbm",
                "pearson": pearson_corr(y_valid, best_pred),
                "rmse": rmse(y_valid, best_pred),
                "train_seconds": 0.0,
                "best_iteration": "",
                "notes": f"ridge_weight={best_weight:.4f}",
            }
        )

        final_state_by_scheme[scheme] = {
            "features": selected_features,
            "ridge_alpha": ridge_best["alpha"],
            "lightgbm_params": lgbm_params,
            "ensemble_ridge_weight": best_weight,
        }

        metrics_df = pd.DataFrame(metrics_rows).sort_values("pearson", ascending=False)
        metrics_df.to_csv(run_dir / "metrics_pearson_topk.csv", index=False)
        del preprocessor, x_train, x_valid, ridge_pred, lgbm_pred, best_pred
        gc.collect()

    metrics_df = pd.DataFrame(metrics_rows).sort_values("pearson", ascending=False)
    metrics_df.to_csv(run_dir / "metrics_pearson_topk.csv", index=False)

    best_row = metrics_df.iloc[0].to_dict()
    baseline_pearson = float(feature_cfg.get("baseline_pearson", 0.100263))
    summary: dict[str, Any] = {
        "best_validation_row": best_row,
        "baseline_pearson": baseline_pearson,
        "beats_baseline": bool(best_row["pearson"] > baseline_pearson),
        "submission_generated": False,
    }

    if not args.no_final and best_row["pearson"] > baseline_pearson:
        if test_path is None:
            raise FileNotFoundError("Best scheme beat baseline, but test file is missing.")

        scheme = str(best_row["scheme"])
        model = str(best_row["model"])
        state = final_state_by_scheme[scheme]
        selected_features = state["features"]
        test_df = read_table(test_path)

        preprocessor = TabularPreprocessor(selected_features).fit(
            train_df,
            missing_fill=str(prep_cfg.get("missing_fill", "median")),
            standardize=bool(prep_cfg.get("standardize", True)),
        )
        x_full = preprocessor.transform(train_df)
        y_full = train_df[target_col].to_numpy(dtype=np.float64)
        x_test = preprocessor.transform(test_df)

        if model == "ridge":
            test_pred = fit_ridge(float(state["ridge_alpha"]), x_full, y_full, x_test)
        elif model == "lightgbm":
            test_pred = fit_lightgbm(state["lightgbm_params"], x_full, y_full, x_test)
        elif model == "ensemble_ridge_lgbm":
            ridge_pred = fit_ridge(float(state["ridge_alpha"]), x_full, y_full, x_test)
            lgbm_pred = fit_lightgbm(state["lightgbm_params"], x_full, y_full, x_test)
            weight = float(state["ensemble_ridge_weight"])
            test_pred = weight * ridge_pred + (1.0 - weight) * lgbm_pred
        else:
            raise ValueError(f"Unsupported best model for submission: {model}")

        submission = make_submission(sample_submission, test_df, test_pred, prediction_col, id_col)
        submission.to_csv(submission_dir / "submission_best.csv", index=False)
        summary["submission_generated"] = True
        summary["submission_path"] = str((submission_dir / "submission_best.csv").relative_to(ROOT))
        summary["submission_shape"] = list(submission.shape)

    write_json(run_dir / "final_summary.json", summary)
    print(metrics_df.to_string(index=False))
    print(f"\nBest validation row: {best_row}")
    print(f"Beats baseline {baseline_pearson:.6f}: {summary['beats_baseline']}")
    print(f"Submission generated: {summary['submission_generated']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

