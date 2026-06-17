from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable

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


STABLE_COMPONENTS = ("full_ridge", "full_lightgbm", "top50_ridge", "top50_lightgbm")
SIGNAL_COMPONENTS = ("top200_ridge", "top300_ridge")
COMPONENTS = (*STABLE_COMPONENTS, *SIGNAL_COMPONENTS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run rolling-aware stable blend optimization.")
    parser.add_argument("--config", default="configs/01_main_stability_blend.yaml", help="Path to YAML config.")
    parser.add_argument("--max-train-rows", type=int, default=None, help="Optional tail sample for smoke runs.")
    parser.add_argument("--no-final", action="store_true", help="Skip final test prediction and submission output.")
    parser.add_argument("--final-only", action="store_true", help="Reuse existing blend artifacts and only generate submission.")
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def slice_by_fraction(df: pd.DataFrame, start: float, end: float) -> pd.DataFrame:
    n_rows = len(df)
    start_idx = int(n_rows * start)
    end_idx = int(n_rows * end)
    if start_idx < 0 or end_idx > n_rows or start_idx >= end_idx:
        raise ValueError(f"Invalid fraction slice: start={start}, end={end}, rows={n_rows}")
    return df.iloc[start_idx:end_idx]


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
        if prediction_col not in submission.columns:
            raise ValueError(f"Prediction column {prediction_col!r} not found in sample submission.")
        submission[prediction_col] = pred
        return submission
    if id_col and id_col in test_df.columns:
        return pd.DataFrame({id_col: test_df[id_col].to_numpy(), prediction_col: pred})
    return pd.DataFrame({"row_id": np.arange(len(pred), dtype=np.int64), prediction_col: pred})


def make_lightgbm_params(params: dict[str, Any], n_estimators: int | None = None) -> dict[str, Any]:
    return {
        "n_estimators": int(n_estimators if n_estimators is not None else params.get("n_estimators", 1200)),
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
        model = NumpyRidgeRegressor(alpha=float(alpha)).fit(x_train, y_train)
        pred = model.predict(x_valid)
        elapsed = time.perf_counter() - start
        row = {
            "alpha": float(alpha),
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


def train_ridge_fixed(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    alpha: float,
) -> tuple[dict[str, Any], np.ndarray]:
    start = time.perf_counter()
    model = NumpyRidgeRegressor(alpha=float(alpha)).fit(x_train, y_train)
    pred = model.predict(x_valid)
    elapsed = time.perf_counter() - start
    return (
        {
            "alpha": float(alpha),
            "pearson": pearson_corr(y_valid, pred),
            "rmse": rmse(y_valid, pred),
            "train_seconds": elapsed,
        },
        pred,
    )


def train_lightgbm(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    params: dict[str, Any],
    n_estimators: int | None = None,
    early_stopping: bool = True,
) -> tuple[dict[str, Any], np.ndarray, dict[str, Any]]:
    import lightgbm as lgb

    def lgb_pearson_eval(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[str, float, bool]:
        return "pearson", pearson_corr(y_true, y_pred), True

    model_params = make_lightgbm_params(params, n_estimators=n_estimators)
    callbacks = []
    eval_set = None
    eval_metric = None
    if early_stopping:
        callbacks = [
            lgb.early_stopping(int(params.get("early_stopping_rounds", 100)), verbose=False),
            lgb.log_evaluation(period=100),
        ]
        eval_set = [(x_valid, y_valid)]
        eval_metric = lgb_pearson_eval

    start = time.perf_counter()
    model = lgb.LGBMRegressor(**model_params)
    model.fit(x_train, y_train, eval_set=eval_set, eval_metric=eval_metric, callbacks=callbacks)
    pred = model.predict(x_valid)
    elapsed = time.perf_counter() - start
    best_iter = getattr(model, "best_iteration_", None) if early_stopping else None
    final_params = dict(model_params)
    if best_iter:
        final_params["n_estimators"] = int(best_iter)

    return (
        {
            "pearson": pearson_corr(y_valid, pred),
            "rmse": rmse(y_valid, pred),
            "train_seconds": elapsed,
            "best_iteration": best_iter if best_iter is not None else "",
        },
        pred,
        final_params,
    )


def fit_ridge_predict(alpha: float, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    return NumpyRidgeRegressor(alpha=float(alpha)).fit(x_train, y_train).predict(x_test)


def fit_lightgbm_predict(params: dict[str, Any], x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    import lightgbm as lgb

    model = lgb.LGBMRegressor(**params)
    model.fit(x_train, y_train)
    return model.predict(x_test)


def resolve_feature_sets(
    train_part: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    stable_top_k: int,
    signal_top_k: list[int],
    ranking_path: Path,
    selected_dir: Path,
    split_name: str,
) -> dict[str, list[str]]:
    ranking = pearson_feature_ranking(train_part, feature_cols, target_col)
    ranking.to_csv(ranking_path, index=False)

    feature_sets: dict[str, list[str]] = {"full": feature_cols}
    top_k_values = sorted({stable_top_k, *signal_top_k})
    for k in top_k_values:
        selected = ranking["feature"].head(min(int(k), len(feature_cols))).tolist()
        scheme = f"top{k}"
        feature_sets[scheme] = selected
        write_lines(selected_dir / f"selected_features_{split_name}_{scheme}.txt", selected)
    write_lines(selected_dir / f"selected_features_{split_name}_full.txt", feature_cols)
    return feature_sets


def train_split_components(
    split_name: str,
    train_part: pd.DataFrame,
    valid_part: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    config: dict[str, Any],
    ranking_dir: Path,
    selected_dir: Path,
    fixed_ridge_alphas: dict[str, float] | None = None,
    fixed_lgbm_estimators: dict[str, int] | None = None,
    tune_lightgbm: bool = True,
) -> tuple[dict[str, np.ndarray], np.ndarray, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    prep_cfg = config["preprocessing"]
    model_cfg = config["models"]
    feature_cfg = config["feature_selection"]
    stable_top_k = int(feature_cfg.get("stable_top_k", 50))
    signal_top_k = [int(k) for k in feature_cfg.get("signal_top_k", [200, 300])]
    y_train = train_part[target_col].to_numpy(dtype=np.float64)
    y_valid = valid_part[target_col].to_numpy(dtype=np.float64)

    feature_sets = resolve_feature_sets(
        train_part,
        feature_cols,
        target_col,
        stable_top_k,
        signal_top_k,
        ranking_dir / f"pearson_feature_ranking_{split_name}.csv",
        selected_dir,
        split_name,
    )

    predictions: dict[str, np.ndarray] = {}
    metric_rows: list[dict[str, Any]] = []
    alpha_rows: list[dict[str, Any]] = []
    lgbm_rows: list[dict[str, Any]] = []

    ridge_schemes = ["full", f"top{stable_top_k}", *(f"top{k}" for k in signal_top_k)]
    lgbm_schemes = ["full", f"top{stable_top_k}"]

    for scheme in ridge_schemes:
        print(f"--- {split_name}: {scheme} Ridge")
        selected_features = feature_sets[scheme]
        preprocessor = TabularPreprocessor(selected_features).fit(
            train_part,
            missing_fill=str(prep_cfg.get("missing_fill", "median")),
            standardize=bool(prep_cfg.get("standardize", True)),
        )
        x_train = preprocessor.transform(train_part)
        x_valid = preprocessor.transform(valid_part)
        fixed_alpha = None if fixed_ridge_alphas is None else fixed_ridge_alphas.get(scheme)
        if fixed_alpha is None:
            best, pred, grid = train_ridge_search(
                x_train,
                y_train,
                x_valid,
                y_valid,
                [float(alpha) for alpha in model_cfg["ridge"].get("alphas", [1.0])],
            )
            for row in grid.to_dict("records"):
                row.update({"split": split_name, "scheme": scheme})
                alpha_rows.append(row)
        else:
            best, pred = train_ridge_fixed(x_train, y_train, x_valid, y_valid, fixed_alpha)
            alpha_rows.append({**best, "split": split_name, "scheme": scheme})

        component = f"{scheme}_ridge"
        predictions[component] = pred
        metric_rows.append(
            {
                "split": split_name,
                "scheme": scheme,
                "component": component,
                "model": "ridge",
                "feature_count": len(selected_features),
                "pearson": best["pearson"],
                "rmse": best["rmse"],
                "train_seconds": best["train_seconds"],
                "best_iteration": "",
                "notes": f"alpha={best['alpha']}",
            }
        )
        del preprocessor, x_train, x_valid, pred
        gc.collect()

    for scheme in lgbm_schemes:
        print(f"--- {split_name}: {scheme} LightGBM")
        selected_features = feature_sets[scheme]
        preprocessor = TabularPreprocessor(selected_features).fit(
            train_part,
            missing_fill=str(prep_cfg.get("missing_fill", "median")),
            standardize=bool(prep_cfg.get("standardize", True)),
        )
        x_train = preprocessor.transform(train_part)
        x_valid = preprocessor.transform(valid_part)
        n_estimators = None if fixed_lgbm_estimators is None else fixed_lgbm_estimators.get(scheme)
        row, pred, params = train_lightgbm(
            x_train,
            y_train,
            x_valid,
            y_valid,
            model_cfg["lightgbm"],
            n_estimators=n_estimators,
            early_stopping=tune_lightgbm and n_estimators is None,
        )
        component = f"{scheme}_lightgbm"
        predictions[component] = pred
        metric_rows.append(
            {
                "split": split_name,
                "scheme": scheme,
                "component": component,
                "model": "lightgbm",
                "feature_count": len(selected_features),
                "pearson": row["pearson"],
                "rmse": row["rmse"],
                "train_seconds": row["train_seconds"],
                "best_iteration": row["best_iteration"],
                "notes": json.dumps(params, sort_keys=True),
            }
        )
        lgbm_rows.append(
            {
                "split": split_name,
                "scheme": scheme,
                "best_iteration": row["best_iteration"],
                "pearson": row["pearson"],
                "rmse": row["rmse"],
                "params": json.dumps(params, sort_keys=True),
            }
        )
        del preprocessor, x_train, x_valid, pred
        gc.collect()

    missing = [component for component in COMPONENTS if component not in predictions]
    if missing:
        raise RuntimeError(f"Missing expected components for {split_name}: {missing}")

    return predictions, y_valid, metric_rows, alpha_rows, lgbm_rows


def prediction_covariance_stats(y_true: np.ndarray, predictions: dict[str, np.ndarray], components: list[str]) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=np.float64)
    x = np.column_stack([np.asarray(predictions[name], dtype=np.float64) for name in components])
    y_centered = y - np.mean(y)
    x_centered = x - np.mean(x, axis=0)
    return {
        "xtx": x_centered.T @ x_centered,
        "xty": x_centered.T @ y_centered,
        "yty": float(y_centered @ y_centered),
    }


def pearson_from_stats(weights: np.ndarray, stats: dict[str, Any]) -> float:
    numerator = float(weights @ stats["xty"])
    pred_ss = float(weights @ stats["xtx"] @ weights)
    denom = np.sqrt(pred_ss * float(stats["yty"]))
    if not np.isfinite(denom) or denom == 0:
        return 0.0
    return numerator / denom


def compositions(total: int, parts: int) -> Iterable[tuple[int, ...]]:
    if parts == 1:
        yield (total,)
        return
    for value in range(total + 1):
        for rest in compositions(total - value, parts - 1):
            yield (value, *rest)


def generate_weight_grid(
    components: list[str],
    step: float,
    min_stable_weight: float,
    max_signal_weight: float,
) -> pd.DataFrame:
    total = int(round(1.0 / step))
    rows: list[dict[str, Any]] = []
    stable_idx = [components.index(name) for name in STABLE_COMPONENTS]
    signal_idx = [components.index(name) for name in SIGNAL_COMPONENTS]
    for units in compositions(total, len(components)):
        weights = np.asarray(units, dtype=np.float64) * step
        stable_weight = float(weights[stable_idx].sum())
        signal_weight = float(weights[signal_idx].sum())
        if stable_weight + 1e-12 < min_stable_weight:
            continue
        if signal_weight - 1e-12 > max_signal_weight:
            continue
        row = {component: float(weight) for component, weight in zip(components, weights)}
        row["stable_weight"] = stable_weight
        row["signal_weight"] = signal_weight
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_weight_grid(
    weight_grid: pd.DataFrame,
    fold_stats: dict[str, dict[str, Any]],
    components: list[str],
    stability_penalty: float,
    min_fold_weight: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in weight_grid.to_dict("records"):
        weights = np.asarray([row[component] for component in components], dtype=np.float64)
        fold_scores = {fold: pearson_from_stats(weights, stats) for fold, stats in fold_stats.items()}
        scores = np.asarray(list(fold_scores.values()), dtype=np.float64)
        mean = float(np.mean(scores))
        std = float(np.std(scores, ddof=0))
        min_score = float(np.min(scores))
        objective = mean - stability_penalty * std + min_fold_weight * min_score
        out = dict(row)
        out.update(
            {
                "pearson_mean": mean,
                "pearson_std": std,
                "pearson_min": min_score,
                "pearson_max": float(np.max(scores)),
                "objective": float(objective),
            }
        )
        out.update({f"pearson_{fold}": score for fold, score in fold_scores.items()})
        rows.append(out)
    return pd.DataFrame(rows).sort_values(["objective", "pearson_mean"], ascending=[False, False])


def weighted_prediction(predictions: dict[str, np.ndarray], weights: dict[str, float], components: list[str]) -> np.ndarray:
    output: np.ndarray | None = None
    for component in components:
        weight = float(weights.get(component, 0.0))
        if weight == 0.0:
            continue
        contribution = weight * np.asarray(predictions[component], dtype=np.float64)
        output = contribution if output is None else output + contribution
    if output is None:
        raise ValueError("All blend weights are zero.")
    return output


def save_prediction(path: Path, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "row_index": np.arange(len(y_true), dtype=np.int64),
            "y_true": y_true,
            "prediction": y_pred,
        }
    ).to_csv(path, index=False)


def select_ridge_alphas(alpha_df: pd.DataFrame, stability_penalty: float, min_fold_weight: float) -> tuple[dict[str, float], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for (scheme, alpha), group in alpha_df.groupby(["scheme", "alpha"]):
        scores = group["pearson"].astype(float).to_numpy()
        rows.append(
            {
                "scheme": scheme,
                "alpha": float(alpha),
                "pearson_mean": float(np.mean(scores)),
                "pearson_std": float(np.std(scores, ddof=0)),
                "pearson_min": float(np.min(scores)),
                "objective": float(np.mean(scores) - stability_penalty * np.std(scores, ddof=0) + min_fold_weight * np.min(scores)),
            }
        )
    summary = pd.DataFrame(rows).sort_values(["scheme", "objective"], ascending=[True, False])
    selected: dict[str, float] = {}
    for scheme, group in summary.groupby("scheme", sort=False):
        selected[str(scheme)] = float(group.iloc[0]["alpha"])
    return selected, summary


def select_lgbm_estimators(lgbm_df: pd.DataFrame, default_n_estimators: int) -> tuple[dict[str, int], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    selected: dict[str, int] = {}
    for scheme, group in lgbm_df.groupby("scheme"):
        values = pd.to_numeric(group["best_iteration"], errors="coerce").dropna().astype(int)
        if values.empty:
            n_estimators = int(default_n_estimators)
        else:
            n_estimators = max(1, int(round(float(values.median()))))
        selected[str(scheme)] = n_estimators
        rows.append(
            {
                "scheme": scheme,
                "selected_n_estimators": n_estimators,
                "median_best_iteration": float(values.median()) if not values.empty else np.nan,
                "min_best_iteration": int(values.min()) if not values.empty else "",
                "max_best_iteration": int(values.max()) if not values.empty else "",
            }
        )
    return selected, pd.DataFrame(rows)


def load_best_weights(run_dir: Path, components: list[str]) -> dict[str, float]:
    path = run_dir / "best_blend_weights.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing best weights file: {path}")
    row = pd.read_csv(path).iloc[0]
    return {component: float(row.get(component, 0.0)) for component in components}


def load_selected_ridge_alphas(run_dir: Path) -> dict[str, float]:
    path = run_dir / "ridge_alpha_stability_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing Ridge alpha summary: {path}")
    summary = pd.read_csv(path).sort_values(["scheme", "objective"], ascending=[True, False])
    selected: dict[str, float] = {}
    for scheme, group in summary.groupby("scheme", sort=False):
        selected[str(scheme)] = float(group.iloc[0]["alpha"])
    return selected


def load_selected_lgbm_estimators(run_dir: Path) -> dict[str, int]:
    path = run_dir / "lightgbm_iteration_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing LightGBM iteration summary: {path}")
    summary = pd.read_csv(path)
    return {str(row["scheme"]): int(row["selected_n_estimators"]) for row in summary.to_dict("records")}


def read_first_record(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    if frame.empty:
        return None
    return frame.iloc[0].to_dict()


def read_metric_summary(path: Path) -> dict[str, float] | None:
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    if frame.empty or "pearson" not in frame.columns:
        return None
    return {
        "pearson_mean": float(frame["pearson"].mean()),
        "pearson_std": float(frame["pearson"].std(ddof=0)),
        "pearson_min": float(frame["pearson"].min()),
        "pearson_max": float(frame["pearson"].max()),
    }


def fit_final_components(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    config: dict[str, Any],
    weights: dict[str, float],
    ridge_alphas: dict[str, float],
    lgbm_estimators: dict[str, int],
    run_dir: Path,
) -> dict[str, np.ndarray]:
    prep_cfg = config["preprocessing"]
    model_cfg = config["models"]
    feature_cfg = config["feature_selection"]
    stable_top_k = int(feature_cfg.get("stable_top_k", 50))
    signal_top_k = [int(k) for k in feature_cfg.get("signal_top_k", [200, 300])]
    selected_dir = ensure_dir(run_dir / "final_selected_features")

    ranking = pearson_feature_ranking(train_df, feature_cols, target_col)
    ranking.to_csv(run_dir / "final_pearson_feature_ranking.csv", index=False)
    feature_sets: dict[str, list[str]] = {"full": feature_cols}
    for k in sorted({stable_top_k, *signal_top_k}):
        selected = ranking["feature"].head(min(k, len(feature_cols))).tolist()
        scheme = f"top{k}"
        feature_sets[scheme] = selected
        write_lines(selected_dir / f"selected_features_final_{scheme}.txt", selected)
    write_lines(selected_dir / "selected_features_final_full.txt", feature_cols)

    y_full = train_df[target_col].to_numpy(dtype=np.float64)
    predictions: dict[str, np.ndarray] = {}
    for component, weight in weights.items():
        if float(weight) == 0.0:
            continue
        scheme, model_name = component.rsplit("_", 1)
        selected_features = feature_sets[scheme]
        preprocessor = TabularPreprocessor(selected_features).fit(
            train_df,
            missing_fill=str(prep_cfg.get("missing_fill", "median")),
            standardize=bool(prep_cfg.get("standardize", True)),
        )
        x_full = preprocessor.transform(train_df)
        x_test = preprocessor.transform(test_df)
        if model_name == "ridge":
            predictions[component] = fit_ridge_predict(ridge_alphas[scheme], x_full, y_full, x_test)
        elif model_name == "lightgbm":
            params = make_lightgbm_params(model_cfg["lightgbm"], n_estimators=lgbm_estimators[scheme])
            predictions[component] = fit_lightgbm_predict(params, x_full, y_full, x_test)
        else:
            raise ValueError(f"Unsupported component: {component}")
        del preprocessor, x_full, x_test
        gc.collect()
    return predictions


def main() -> int:
    args = parse_args()
    config_path = ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)
    config = load_config(config_path)

    data_cfg = config["data"]
    output_cfg = config["output"]
    blend_cfg = config["blend"]
    run_dir = ensure_dir(ROOT / output_cfg["run_dir"])
    submission_dir = ensure_dir(ROOT / output_cfg["submission_dir"])
    ranking_dir = ensure_dir(run_dir / "feature_rankings")
    selected_dir = ensure_dir(run_dir / "selected_features")
    pred_dir = ensure_dir(run_dir / "valid_predictions")

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
        numeric_only=bool(config["preprocessing"].get("numeric_only", True)),
    )
    prediction_col = choose_prediction_column(sample_submission, output_cfg.get("prediction_col"), target_col)
    components = list(COMPONENTS)

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
            "max_train_rows": max_train_rows,
            "components": components,
        },
    )
    write_json(run_dir / "resolved_config.json", config)

    if args.final_only:
        if test_path is None:
            raise FileNotFoundError("Missing test file under data/raw.")
        print("\n=== Final-only submission generation ===")
        best_weights = load_best_weights(run_dir, components)
        ridge_alphas = load_selected_ridge_alphas(run_dir)
        lgbm_estimators = load_selected_lgbm_estimators(run_dir)
        test_df = read_table(test_path)
        final_component_predictions = fit_final_components(
            train_df,
            test_df,
            feature_cols,
            target_col,
            config,
            best_weights,
            ridge_alphas,
            lgbm_estimators,
            run_dir,
        )
        test_pred = weighted_prediction(final_component_predictions, best_weights, components)
        submission = make_submission(sample_submission, test_df, test_pred, prediction_col, id_col)
        submission_path = submission_dir / "submission_best.csv"
        submission.to_csv(submission_path, index=False)
        final_summary = {
            "best_candidate": read_first_record(run_dir / "blend_candidates_top.csv"),
            "best_weights": best_weights,
            "rolling_blend_metrics": read_metric_summary(run_dir / "best_blend_fold_metrics.csv"),
            "ridge_alphas": ridge_alphas,
            "lightgbm_estimators": lgbm_estimators,
            "holdout_summary": read_first_record(run_dir / "holdout_stability_blend_metrics.csv"),
            "submission_generated": True,
            "submission_path": str(submission_path.relative_to(ROOT)),
            "submission_shape": list(submission.shape),
            "prediction_mean": float(np.mean(test_pred)),
            "prediction_std": float(np.std(test_pred)),
        }
        write_json(run_dir / "final_summary.json", final_summary)
        print(f"Submission path: {submission_path.relative_to(ROOT)}")
        print(f"Submission shape: {submission.shape}")
        return 0

    split_predictions: dict[str, dict[str, np.ndarray]] = {}
    split_targets: dict[str, np.ndarray] = {}
    fold_stats: dict[str, dict[str, Any]] = {}
    component_metric_rows: list[dict[str, Any]] = []
    ridge_alpha_rows: list[dict[str, Any]] = []
    lgbm_rows: list[dict[str, Any]] = []

    for fold in config["rolling"]["folds"]:
        split_name = fold["name"]
        print(f"\n=== Rolling fold: {split_name} ===")
        train_part = slice_by_fraction(train_df, float(fold["train_start"]), float(fold["train_end"]))
        valid_part = slice_by_fraction(train_df, float(fold["valid_start"]), float(fold["valid_end"]))
        predictions, y_valid, metrics, alpha_rows, lgbm_metric_rows = train_split_components(
            split_name,
            train_part,
            valid_part,
            feature_cols,
            target_col,
            config,
            ranking_dir,
            selected_dir,
        )
        split_predictions[split_name] = predictions
        split_targets[split_name] = y_valid
        fold_stats[split_name] = prediction_covariance_stats(y_valid, predictions, components)
        component_metric_rows.extend(metrics)
        ridge_alpha_rows.extend(alpha_rows)
        lgbm_rows.extend(lgbm_metric_rows)

        pd.DataFrame(component_metric_rows).to_csv(run_dir / "component_metrics_rolling.csv", index=False)
        pd.DataFrame(ridge_alpha_rows).to_csv(run_dir / "ridge_alpha_search_rolling.csv", index=False)
        pd.DataFrame(lgbm_rows).to_csv(run_dir / "lightgbm_iterations_rolling.csv", index=False)

    component_metrics = pd.DataFrame(component_metric_rows)
    component_metrics.to_csv(run_dir / "component_metrics_rolling.csv", index=False)
    ridge_alpha_df = pd.DataFrame(ridge_alpha_rows)
    ridge_alpha_df.to_csv(run_dir / "ridge_alpha_search_rolling.csv", index=False)
    lgbm_df = pd.DataFrame(lgbm_rows)
    lgbm_df.to_csv(run_dir / "lightgbm_iterations_rolling.csv", index=False)

    weight_grid = generate_weight_grid(
        components,
        step=float(blend_cfg.get("step", 0.05)),
        min_stable_weight=float(blend_cfg.get("min_stable_weight", 0.70)),
        max_signal_weight=float(blend_cfg.get("max_signal_weight", 0.30)),
    )
    candidates = evaluate_weight_grid(
        weight_grid,
        fold_stats,
        components,
        stability_penalty=float(blend_cfg.get("stability_penalty", 1.0)),
        min_fold_weight=float(blend_cfg.get("min_fold_weight", 0.25)),
    )
    save_top_n = int(blend_cfg.get("save_top_n", 1000))
    candidates.head(save_top_n).to_csv(run_dir / "blend_candidates_top.csv", index=False)
    best_candidate = candidates.iloc[0].to_dict()
    best_weights = {component: float(best_candidate[component]) for component in components}
    pd.DataFrame([best_weights]).to_csv(run_dir / "best_blend_weights.csv", index=False)

    best_fold_rows: list[dict[str, Any]] = []
    for split_name, predictions in split_predictions.items():
        pred = weighted_prediction(predictions, best_weights, components)
        y_valid = split_targets[split_name]
        save_prediction(pred_dir / f"valid_{split_name}_stability_blend.csv", y_valid, pred)
        best_fold_rows.append(
            {
                "split": split_name,
                "pearson": pearson_corr(y_valid, pred),
                "rmse": rmse(y_valid, pred),
                "prediction_std": float(np.std(pred)),
                "target_std": float(np.std(y_valid)),
            }
        )
    best_fold_metrics = pd.DataFrame(best_fold_rows)
    best_fold_metrics.to_csv(run_dir / "best_blend_fold_metrics.csv", index=False)

    ridge_alphas, ridge_alpha_summary = select_ridge_alphas(
        ridge_alpha_df,
        stability_penalty=float(blend_cfg.get("stability_penalty", 1.0)),
        min_fold_weight=float(blend_cfg.get("min_fold_weight", 0.25)),
    )
    ridge_alpha_summary.to_csv(run_dir / "ridge_alpha_stability_summary.csv", index=False)
    lgbm_estimators, lgbm_summary = select_lgbm_estimators(
        lgbm_df,
        default_n_estimators=int(config["models"]["lightgbm"].get("n_estimators", 1200)),
    )
    lgbm_summary.to_csv(run_dir / "lightgbm_iteration_summary.csv", index=False)

    holdout_summary: dict[str, Any] | None = None
    if bool(config.get("holdout_validation", {}).get("enabled", False)):
        holdout = config["holdout_validation"]
        split_name = holdout.get("name", "holdout")
        print(f"\n=== Holdout validation: {split_name} ===")
        train_part = slice_by_fraction(train_df, float(holdout["train_start"]), float(holdout["train_end"]))
        valid_part = slice_by_fraction(train_df, float(holdout["valid_start"]), float(holdout["valid_end"]))
        predictions, y_valid, metrics, alpha_rows, lgbm_metric_rows = train_split_components(
            split_name,
            train_part,
            valid_part,
            feature_cols,
            target_col,
            config,
            ranking_dir,
            selected_dir,
            fixed_ridge_alphas=ridge_alphas,
            fixed_lgbm_estimators=lgbm_estimators,
            tune_lightgbm=False,
        )
        pd.DataFrame(metrics).to_csv(run_dir / "component_metrics_holdout.csv", index=False)
        pd.DataFrame(alpha_rows).to_csv(run_dir / "ridge_alpha_holdout.csv", index=False)
        pd.DataFrame(lgbm_metric_rows).to_csv(run_dir / "lightgbm_holdout.csv", index=False)
        pred = weighted_prediction(predictions, best_weights, components)
        save_prediction(pred_dir / f"valid_{split_name}_stability_blend.csv", y_valid, pred)
        holdout_summary = {
            "split": split_name,
            "pearson": pearson_corr(y_valid, pred),
            "rmse": rmse(y_valid, pred),
            "prediction_std": float(np.std(pred)),
            "target_std": float(np.std(y_valid)),
        }
        pd.DataFrame([holdout_summary]).to_csv(run_dir / "holdout_stability_blend_metrics.csv", index=False)

    submission_summary: dict[str, Any] = {"submission_generated": False}
    if not args.no_final:
        if test_path is None:
            raise FileNotFoundError("Missing test file under data/raw.")
        print("\n=== Final fit and submission ===")
        test_df = read_table(test_path)
        final_component_predictions = fit_final_components(
            train_df,
            test_df,
            feature_cols,
            target_col,
            config,
            best_weights,
            ridge_alphas,
            lgbm_estimators,
            run_dir,
        )
        test_pred = weighted_prediction(final_component_predictions, best_weights, components)
        submission = make_submission(sample_submission, test_df, test_pred, prediction_col, id_col)
        submission_path = submission_dir / "submission_best.csv"
        submission.to_csv(submission_path, index=False)
        submission_summary = {
            "submission_generated": True,
            "submission_path": str(submission_path.relative_to(ROOT)),
            "submission_shape": list(submission.shape),
            "prediction_mean": float(np.mean(test_pred)),
            "prediction_std": float(np.std(test_pred)),
        }

    final_summary = {
        "best_candidate": best_candidate,
        "best_weights": best_weights,
        "rolling_blend_metrics": {
            "pearson_mean": float(best_fold_metrics["pearson"].mean()),
            "pearson_std": float(best_fold_metrics["pearson"].std(ddof=0)),
            "pearson_min": float(best_fold_metrics["pearson"].min()),
            "pearson_max": float(best_fold_metrics["pearson"].max()),
        },
        "ridge_alphas": ridge_alphas,
        "lightgbm_estimators": lgbm_estimators,
        "holdout_summary": holdout_summary,
        **submission_summary,
    }
    write_json(run_dir / "final_summary.json", final_summary)

    print("\n=== Best stability blend ===")
    print(pd.DataFrame([best_candidate]).to_string(index=False))
    print("\n=== Rolling fold metrics ===")
    print(best_fold_metrics.to_string(index=False))
    if holdout_summary:
        print("\n=== Holdout metrics ===")
        print(pd.DataFrame([holdout_summary]).to_string(index=False))
    print(f"\nSubmission generated: {submission_summary['submission_generated']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
