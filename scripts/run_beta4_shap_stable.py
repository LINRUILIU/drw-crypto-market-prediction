from __future__ import annotations

import argparse
import gc
import json
import shutil
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

from drw_crypto.feature_selection import aggregate_shap_stable_features  # noqa: E402
from drw_crypto.io import (  # noqa: E402
    SAMPLE_SUBMISSION_CANDIDATES,
    TEST_CANDIDATES,
    TRAIN_CANDIDATES,
    ensure_dir,
    find_first_existing,
    read_table,
    write_json,
)
from drw_crypto.metrics import pearson_corr, rmse  # noqa: E402
from drw_crypto.pipeline import (  # noqa: E402
    choose_prediction_column,
    fit_ridge,
    make_submission,
    read_lines,
    safe_name,
    save_valid_prediction,
    split_time_ordered,
    train_ridge_search,
    weighted_ridge_prediction,
    write_selected_features,
)
from drw_crypto.preprocessing import (  # noqa: E402
    TabularPreprocessor,
    infer_feature_columns,
    infer_id_column,
    infer_target_column,
)
from drw_crypto.validation import purged_group_time_series_splits  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Beta4-B XGB SHAP-stable feature experiments.")
    parser.add_argument("--config", default="configs/01_main_beta4_shap_stable.yaml", help="Path to YAML config.")
    parser.add_argument("--max-train-rows", type=int, default=None, help="Optional tail sample for smoke runs.")
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def xgb_params(params: dict[str, Any], random_state: int) -> dict[str, Any]:
    return {
        "n_estimators": int(params.get("n_estimators", 800)),
        "learning_rate": float(params.get("learning_rate", 0.03)),
        "max_depth": int(params.get("max_depth", 5)),
        "subsample": float(params.get("subsample", 0.85)),
        "colsample_bytree": float(params.get("colsample_bytree", 0.85)),
        "reg_lambda": float(params.get("reg_lambda", 1.0)),
        "objective": "reg:squarederror",
        "random_state": int(random_state),
        "n_jobs": -1,
        "tree_method": "hist",
        "eval_metric": "rmse",
    }


def train_xgb_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    params: dict[str, Any],
    random_state: int,
) -> tuple[Any, np.ndarray, dict[str, Any], dict[str, Any]]:
    import xgboost as xgb

    model_params = xgb_params(params, random_state)
    early_stopping_rounds = int(params.get("early_stopping_rounds", 80))
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
    row = {
        "pearson": pearson_corr(y_valid, pred),
        "rmse": rmse(y_valid, pred),
        "train_seconds": elapsed,
        "best_iteration": best_iter,
        "params": json.dumps(final_params, sort_keys=True),
    }
    return model, pred, final_params, row


def fit_xgb_predict(params: dict[str, Any], x_train: np.ndarray, y_train: np.ndarray, x_pred: np.ndarray) -> np.ndarray:
    import xgboost as xgb

    model_params = dict(params)
    model_params.pop("early_stopping_rounds", None)
    model = xgb.XGBRegressor(**model_params)
    model.fit(x_train, y_train, verbose=False)
    return model.predict(x_pred)


def shap_top_features(model: Any, x_valid: np.ndarray, feature_cols: list[str], fold: int, top_n: int) -> pd.DataFrame:
    import xgboost as xgb

    matrix = xgb.DMatrix(x_valid, feature_names=feature_cols)
    contrib = model.get_booster().predict(matrix, pred_contribs=True)
    values = np.abs(contrib[:, : len(feature_cols)]).mean(axis=0)
    rows = (
        pd.DataFrame({"feature": feature_cols, "mean_abs_shap": values})
        .sort_values(["mean_abs_shap", "feature"], ascending=[False, True])
        .head(int(top_n))
        .reset_index(drop=True)
    )
    rows.insert(0, "rank", np.arange(1, len(rows) + 1, dtype=np.int64))
    rows.insert(0, "fold", int(fold))
    return rows


def summarize_delta(pred: np.ndarray, reference: np.ndarray, prefix: str) -> dict[str, float]:
    delta = pred - reference
    return {
        f"{prefix}_corr_current_best": pearson_corr(reference, pred),
        f"{prefix}_delta_std_current_best": float(np.std(delta)),
        f"{prefix}_delta_mae_current_best": float(np.mean(np.abs(delta))),
    }


def main() -> int:
    args = parse_args()
    config_path = ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)
    config = load_config(config_path)

    data_cfg = config["data"]
    prep_cfg = config["preprocessing"]
    current_cfg = config["current_best"]
    feature_cfg = config["features"]
    cv_cfg = config["cv"]
    xgb_cfg = config["xgboost"]
    stable_cfg = config["stable_selection"]
    ridge_cfg = config["ridge"]
    blend_cfg = config["blend"]
    output_cfg = config["output"]

    run_dir = ensure_dir(ROOT / output_cfg["run_dir"])
    pred_dir = ensure_dir(run_dir / "valid_predictions")
    shap_dir = ensure_dir(run_dir / "shap")
    submission_dir = ensure_dir(ROOT / output_cfg["submission_dir"])
    signal_submission_dir = ensure_dir(submission_dir / "signals")

    raw_dir = ROOT / data_cfg["raw_dir"]
    train_path = find_first_existing(raw_dir, data_cfg.get("train_file"), TRAIN_CANDIDATES)
    test_path = find_first_existing(raw_dir, data_cfg.get("test_file"), TEST_CANDIDATES)
    sample_path = find_first_existing(raw_dir, data_cfg.get("sample_submission_file"), SAMPLE_SUBMISSION_CANDIDATES)
    if train_path is None or test_path is None:
        raise FileNotFoundError("Missing train/test files under data/raw.")

    medoid_path = ROOT / feature_cfg["medoid_features_path"]
    if not medoid_path.exists():
        raise FileNotFoundError(f"Missing medoid feature list: {medoid_path}. Run Beta4-A first.")
    medoid_features = read_lines(medoid_path)

    train_df = read_table(train_path)
    if args.max_train_rows is not None:
        train_df = train_df.tail(int(args.max_train_rows)).reset_index(drop=True)
    test_df = read_table(test_path)
    sample_submission = read_table(sample_path) if sample_path is not None else None

    target_col = infer_target_column(train_df, test_df, data_cfg.get("target_col"))
    id_col = infer_id_column(train_df, test_df, sample_submission, target_col, data_cfg.get("id_col"))
    feature_cols = infer_feature_columns(
        train_df,
        target_col,
        id_col,
        data_cfg.get("drop_columns", []),
        bool(prep_cfg.get("numeric_only", True)),
    )
    invalid = set(medoid_features) - set(feature_cols)
    if invalid:
        raise ValueError(f"Medoid feature list contains unknown features: {sorted(invalid)[:5]}")
    prediction_col = choose_prediction_column(sample_submission, output_cfg.get("prediction_col"), target_col)

    train_part, valid_part = split_time_ordered(train_df, float(data_cfg["validation_fraction"]))
    y_train = train_part[target_col].to_numpy(dtype=np.float64)
    y_valid = valid_part[target_col].to_numpy(dtype=np.float64)
    y_full = train_df[target_col].to_numpy(dtype=np.float64)

    current_best_valid = weighted_ridge_prediction(
        train_part,
        y_train,
        valid_part,
        current_cfg["components"],
        prep_cfg,
        ROOT,
    )
    current_best_test = weighted_ridge_prediction(
        train_df,
        y_full,
        test_df,
        current_cfg["components"],
        prep_cfg,
        ROOT,
    )
    current_best_holdout = pearson_corr(y_valid, current_best_valid)

    fold_rows: list[dict[str, Any]] = []
    top_tables: list[pd.DataFrame] = []
    splits = purged_group_time_series_splits(
        len(train_part),
        n_groups=int(cv_cfg.get("n_groups", 6)),
        gap=int(cv_cfg.get("gap", 1)),
    )
    for split in splits:
        print(f"\n=== XGB SHAP fold {split.fold} valid_group={split.valid_group} purged={split.purged_groups} ===")
        fold_train = train_part.iloc[split.train_indices]
        fold_valid = train_part.iloc[split.valid_indices]
        fold_y_train = fold_train[target_col].to_numpy(dtype=np.float64)
        fold_y_valid = fold_valid[target_col].to_numpy(dtype=np.float64)
        preprocessor = TabularPreprocessor(medoid_features).fit(
            fold_train,
            missing_fill=str(prep_cfg.get("missing_fill", "median")),
            standardize=bool(prep_cfg.get("standardize", True)),
        )
        x_fold_train = preprocessor.transform(fold_train)
        x_fold_valid = preprocessor.transform(fold_valid)
        model, pred, final_params, row = train_xgb_model(
            x_fold_train,
            fold_y_train,
            x_fold_valid,
            fold_y_valid,
            xgb_cfg,
            random_state=42 + split.fold,
        )
        fold_rows.append(
            {
                "fold": split.fold,
                "valid_group": split.valid_group,
                "purged_groups": ",".join(str(group) for group in split.purged_groups),
                "train_rows": len(split.train_indices),
                "valid_rows": len(split.valid_indices),
                **row,
            }
        )
        top = shap_top_features(model, x_fold_valid, medoid_features, split.fold, int(stable_cfg["fold_top_n"]))
        top_tables.append(top)
        top.to_csv(shap_dir / f"fold{split.fold}_shap_top{int(stable_cfg['fold_top_n'])}.csv", index=False)
        del preprocessor, x_fold_train, x_fold_valid, model, pred
        gc.collect()

    fold_metrics = pd.DataFrame(fold_rows)
    fold_metrics.to_csv(run_dir / "purged_xgb_fold_metrics.csv", index=False)
    fold_top = pd.concat(top_tables, ignore_index=True)
    fold_top.to_csv(run_dir / "shap_fold_top_features.csv", index=False)
    stable_table = aggregate_shap_stable_features(
        fold_top,
        min_fold_appearances=int(stable_cfg["min_fold_appearances"]),
        fill_to_n=int(stable_cfg["fill_to_n"]),
    )
    stable_table.to_csv(run_dir / "shap_stable_feature_table.csv", index=False)
    stable_features = stable_table[stable_table["selected"]]["feature"].astype(str).tolist()
    write_selected_features(run_dir / "selected_features_shap_stable.txt", stable_features)

    metrics_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    preprocessor = TabularPreprocessor(stable_features).fit(
        train_part,
        missing_fill=str(prep_cfg.get("missing_fill", "median")),
        standardize=bool(prep_cfg.get("standardize", True)),
    )
    x_train = preprocessor.transform(train_part)
    x_valid = preprocessor.transform(valid_part)
    final_preprocessor = TabularPreprocessor(stable_features).fit(
        train_df,
        missing_fill=str(prep_cfg.get("missing_fill", "median")),
        standardize=bool(prep_cfg.get("standardize", True)),
    )
    x_full = final_preprocessor.transform(train_df)
    x_test = final_preprocessor.transform(test_df)

    ridge_best, ridge_valid, ridge_grid = train_ridge_search(
        x_train,
        y_train,
        x_valid,
        y_valid,
        [float(alpha) for alpha in ridge_cfg["alphas"]],
    )
    ridge_grid.to_csv(run_dir / "ridge_alpha_search_shap_stable.csv", index=False)
    save_valid_prediction(pred_dir / "valid_shap_stable_ridge.csv", y_valid, ridge_valid)
    ridge_test = fit_ridge(float(ridge_best["alpha"]), x_full, y_full, x_test)
    signal_submission = make_submission(sample_submission, test_df, ridge_test, prediction_col, id_col)
    signal_submission.to_csv(signal_submission_dir / "submission_shap_stable_ridge.csv", index=False)
    metrics_rows.append(
        {
            "feature_set": "shap_stable",
            "feature_count": len(stable_features),
            "model": "ridge",
            "alpha": float(ridge_best["alpha"]),
            "pearson": pearson_corr(y_valid, ridge_valid),
            "rmse": rmse(y_valid, ridge_valid),
            "holdout_delta_current_best": pearson_corr(y_valid, ridge_valid) - current_best_holdout,
        }
    )

    xgb_model, xgb_valid, xgb_final_params, xgb_row = train_xgb_model(x_train, y_train, x_valid, y_valid, xgb_cfg, 2026)
    save_valid_prediction(pred_dir / "valid_shap_stable_xgboost.csv", y_valid, xgb_valid)
    xgb_test = fit_xgb_predict(xgb_final_params, x_full, y_full, x_test)
    signal_submission = make_submission(sample_submission, test_df, xgb_test, prediction_col, id_col)
    signal_submission.to_csv(signal_submission_dir / "submission_shap_stable_xgboost.csv", index=False)
    metrics_rows.append(
        {
            "feature_set": "shap_stable",
            "feature_count": len(stable_features),
            "model": "xgboost",
            "alpha": "",
            "pearson": pearson_corr(y_valid, xgb_valid),
            "rmse": rmse(y_valid, xgb_valid),
            "holdout_delta_current_best": pearson_corr(y_valid, xgb_valid) - current_best_holdout,
            "best_iteration": xgb_row["best_iteration"],
            "params": json.dumps(xgb_final_params, sort_keys=True),
        }
    )

    signal_map = {
        "ridge": (ridge_valid, ridge_test),
        "xgboost": (xgb_valid, xgb_test),
    }
    for model_name, (valid_pred, test_pred) in signal_map.items():
        for signal_weight in [float(value) for value in blend_cfg["signal_weights"]]:
            base_weight = 1.0 - signal_weight
            blend_valid = base_weight * current_best_valid + signal_weight * valid_pred
            blend_test = base_weight * current_best_test + signal_weight * test_pred
            candidate = f"blend_current_w{safe_name(base_weight)}_shap_stable_{model_name}_w{safe_name(signal_weight)}"
            submission = make_submission(sample_submission, test_df, blend_test, prediction_col, id_col)
            path = submission_dir / f"submission_{candidate}.csv"
            submission.to_csv(path, index=False)
            holdout = pearson_corr(y_valid, blend_valid)
            candidate_rows.append(
                {
                    "candidate": candidate,
                    "feature_count": len(stable_features),
                    "model": model_name,
                    "current_best_weight": base_weight,
                    "signal_weight": signal_weight,
                    "holdout_pearson": holdout,
                    "holdout_rmse": rmse(y_valid, blend_valid),
                    "holdout_delta_current_best": holdout - current_best_holdout,
                    "valid_prediction_std": float(np.std(blend_valid)),
                    "submission_prediction_std": float(np.std(blend_test)),
                    "std_ratio_current_best": float(np.std(blend_test) / np.std(current_best_test)),
                    **summarize_delta(blend_valid, current_best_valid, "valid"),
                    **summarize_delta(blend_test, current_best_test, "submission"),
                    "path": str(path.relative_to(ROOT)),
                }
            )

    metrics = pd.DataFrame(metrics_rows).sort_values("pearson", ascending=False)
    metrics.to_csv(run_dir / "metrics_shap_stable.csv", index=False)
    candidates = pd.DataFrame(candidate_rows)
    candidates["eligible"] = (
        (candidates["holdout_delta_current_best"] >= -float(blend_cfg.get("holdout_tolerance", 0.003)))
        & (candidates["std_ratio_current_best"] <= float(blend_cfg.get("max_std_ratio_current_best", 1.2)))
    )
    candidates = candidates.sort_values(
        ["eligible", "holdout_pearson", "submission_corr_current_best"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    candidates.to_csv(run_dir / "candidate_metrics.csv", index=False)

    selected = candidates[candidates["eligible"]].head(int(blend_cfg.get("max_selected_submissions", 3))).copy()
    if selected.empty:
        selected = candidates.head(int(blend_cfg.get("max_selected_submissions", 3))).copy()
    selected.to_csv(run_dir / "selected_candidates.csv", index=False)
    if not selected.empty:
        shutil.copyfile(ROOT / str(selected.iloc[0]["path"]), submission_dir / "submission_best.csv")

    write_json(
        run_dir / "final_summary.json",
        {
            "current_best": current_cfg,
            "current_best_holdout_pearson": current_best_holdout,
            "medoid_features_path": feature_cfg["medoid_features_path"],
            "medoid_feature_count": len(medoid_features),
            "stable_feature_count": len(stable_features),
            "stable_features": stable_features,
            "shap_backend": "xgboost_pred_contribs",
            "best_metric_row": metrics.iloc[0].to_dict(),
            "best_candidate": selected.iloc[0].to_dict() if not selected.empty else None,
        },
    )

    print("\n=== SHAP Stable Metrics ===")
    print(metrics.to_string(index=False))
    print("\n=== Selected Candidates ===")
    print(selected.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
