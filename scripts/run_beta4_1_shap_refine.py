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
    make_submission,
    read_lines,
    safe_name,
    save_valid_prediction,
    split_time_ordered,
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
    parser = argparse.ArgumentParser(description="Run Beta4.1 SHAP-stable XGB refinement.")
    parser.add_argument("--config", default="configs/01_main_beta4_1_shap_refine.yaml", help="Path to YAML config.")
    parser.add_argument("--max-train-rows", type=int, default=None, help="Optional tail sample for smoke runs.")
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def base_xgb_params(params: dict[str, Any], random_state: int) -> dict[str, Any]:
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


def pearson_error_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return 1.0 - pearson_corr(y_true, y_pred)


def train_xgb_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    params: dict[str, Any],
    variant: dict[str, Any],
    random_state: int,
) -> tuple[Any, np.ndarray, dict[str, Any], dict[str, Any]]:
    import xgboost as xgb

    mode = str(variant.get("mode", "rmse_es"))
    model_params = base_xgb_params(params, random_state)
    model_params["n_estimators"] = int(variant.get("n_estimators", model_params["n_estimators"]))
    eval_metric_name = "rmse"
    if mode == "fixed":
        model_params.pop("early_stopping_rounds", None)
    elif mode == "pearson_es":
        model_params["early_stopping_rounds"] = int(variant.get("early_stopping_rounds", 100))
        model_params["eval_metric"] = pearson_error_metric
        eval_metric_name = "pearson_error"
    else:
        model_params["early_stopping_rounds"] = int(variant.get("early_stopping_rounds", params.get("early_stopping_rounds", 80)))

    start = time.perf_counter()
    model = xgb.XGBRegressor(**model_params)
    try:
        model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=False)
    except TypeError:
        model_params.pop("early_stopping_rounds", None)
        if callable(model_params.get("eval_metric")):
            model_params["eval_metric"] = "rmse"
            eval_metric_name = "rmse_fallback"
        model = xgb.XGBRegressor(**model_params)
        model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=False)
    pred = model.predict(x_valid)
    elapsed = time.perf_counter() - start

    best_iter = getattr(model, "best_iteration", None) if "early_stopping_rounds" in model_params else None
    final_params = dict(model_params)
    if best_iter is not None:
        final_params["n_estimators"] = int(best_iter) + 1
        final_params.pop("early_stopping_rounds", None)
    if callable(final_params.get("eval_metric")):
        final_params["eval_metric"] = "rmse"
    row = {
        "pearson": pearson_corr(y_valid, pred),
        "rmse": rmse(y_valid, pred),
        "train_seconds": elapsed,
        "best_iteration": best_iter,
        "eval_metric": eval_metric_name,
        "params": json.dumps(final_params, sort_keys=True),
    }
    return model, pred, final_params, row


def fit_xgb_predict(params: dict[str, Any], x_train: np.ndarray, y_train: np.ndarray, x_pred: np.ndarray) -> np.ndarray:
    import xgboost as xgb

    model_params = dict(params)
    model_params.pop("early_stopping_rounds", None)
    if callable(model_params.get("eval_metric")):
        model_params["eval_metric"] = "rmse"
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


def add_candidate(
    rows: list[dict[str, Any]],
    stage: str,
    name: str,
    feature_rule: str,
    training_variant: str,
    signal_weight: float,
    base_valid: np.ndarray,
    base_test: np.ndarray,
    signal_valid: np.ndarray,
    signal_test: np.ndarray,
    current_best_valid: np.ndarray,
    current_best_test: np.ndarray,
    y_valid: np.ndarray,
    sample_submission: pd.DataFrame | None,
    test_df: pd.DataFrame,
    prediction_col: str,
    id_col: str | None,
    submission_dir: Path,
) -> None:
    base_weight = 1.0 - float(signal_weight)
    blend_valid = base_weight * base_valid + float(signal_weight) * signal_valid
    blend_test = base_weight * base_test + float(signal_weight) * signal_test
    submission = make_submission(sample_submission, test_df, blend_test, prediction_col, id_col)
    path = submission_dir / f"submission_{name}.csv"
    submission.to_csv(path, index=False)
    holdout = pearson_corr(y_valid, blend_valid)
    rows.append(
        {
            "stage": stage,
            "candidate": name,
            "feature_rule": feature_rule,
            "training_variant": training_variant,
            "signal_weight": float(signal_weight),
            "base_weight": base_weight,
            "holdout_pearson": holdout,
            "holdout_rmse": rmse(y_valid, blend_valid),
            "holdout_delta_current_best": holdout - pearson_corr(y_valid, current_best_valid),
            "valid_prediction_std": float(np.std(blend_valid)),
            "submission_prediction_std": float(np.std(blend_test)),
            "std_ratio_current_best": float(np.std(blend_test) / np.std(current_best_test)),
            **summarize_delta(blend_valid, current_best_valid, "valid"),
            **summarize_delta(blend_test, current_best_test, "submission"),
            "path": str(path.relative_to(ROOT)),
        }
    )


def selected_feature_stats(stable_table: pd.DataFrame) -> dict[str, int]:
    selected = stable_table[stable_table["selected"]]
    return {
        "selected_count": int(len(selected)),
        "stable_count": int((selected["selection_reason"] == "stable").sum()),
        "rank_fill_count": int((selected["selection_reason"] == "rank_fill").sum()),
    }


def main() -> int:
    args = parse_args()
    config_path = ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)
    config = load_config(config_path)

    data_cfg = config["data"]
    prep_cfg = config["preprocessing"]
    base_cfg = config["base_signal"]
    current_cfg = config["current_best"]
    feature_cfg = config["features"]
    cv_cfg = config["cv"]
    xgb_cfg = config["xgboost"]
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
        raise FileNotFoundError(f"Missing medoid feature list: {medoid_path}.")
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

    base_valid = weighted_ridge_prediction(train_part, y_train, valid_part, base_cfg["components"], prep_cfg, ROOT)
    base_test = weighted_ridge_prediction(train_df, y_full, test_df, base_cfg["components"], prep_cfg, ROOT)
    base_holdout = pearson_corr(y_valid, base_valid)

    # Train purged-fold XGB once with top-N large enough for all feature rules.
    max_top_n = max(int(rule["fold_top_n"]) for rule in config["feature_rules"])
    fold_rows: list[dict[str, Any]] = []
    fold_top_tables: list[pd.DataFrame] = []
    default_variant = {
        "name": "rmse_es",
        "mode": "rmse_es",
        "n_estimators": int(xgb_cfg.get("n_estimators", 800)),
        "early_stopping_rounds": int(xgb_cfg.get("early_stopping_rounds", 80)),
    }
    for split in purged_group_time_series_splits(
        len(train_part),
        n_groups=int(cv_cfg.get("n_groups", 6)),
        gap=int(cv_cfg.get("gap", 1)),
    ):
        print(f"\n=== Stage A SHAP fold {split.fold} valid_group={split.valid_group} purged={split.purged_groups} ===")
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
        model, _, _, row = train_xgb_model(
            x_fold_train,
            fold_y_train,
            x_fold_valid,
            fold_y_valid,
            xgb_cfg,
            default_variant,
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
        top = shap_top_features(model, x_fold_valid, medoid_features, split.fold, max_top_n)
        top.to_csv(shap_dir / f"fold{split.fold}_shap_top{max_top_n}.csv", index=False)
        fold_top_tables.append(top)
        del preprocessor, x_fold_train, x_fold_valid, model
        gc.collect()

    pd.DataFrame(fold_rows).to_csv(run_dir / "purged_xgb_fold_metrics.csv", index=False)
    fold_top_all = pd.concat(fold_top_tables, ignore_index=True)
    fold_top_all.to_csv(run_dir / f"shap_fold_top_features_top{max_top_n}.csv", index=False)

    feature_rule_metrics: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    signal_cache: dict[str, dict[str, Any]] = {}
    baseline_signal: dict[str, Any] | None = None

    for rule in config["feature_rules"]:
        rule_name = str(rule["name"])
        fold_top_n = int(rule["fold_top_n"])
        fold_top = fold_top_all[fold_top_all["rank"] <= fold_top_n].copy()
        fold_top.to_csv(run_dir / f"shap_fold_top_features_{rule_name}.csv", index=False)
        stable_table = aggregate_shap_stable_features(
            fold_top,
            min_fold_appearances=int(rule["min_fold_appearances"]),
            fill_to_n=int(rule["fill_to_n"]),
        )
        stable_table.to_csv(run_dir / f"shap_stable_feature_table_{rule_name}.csv", index=False)
        selected_features = stable_table[stable_table["selected"]]["feature"].astype(str).tolist()
        if not selected_features:
            raise ValueError(f"Feature rule {rule_name} selected no features.")
        if int(rule["fill_to_n"]) == 0 and (stable_table["selection_reason"] == "rank_fill").any():
            raise ValueError(f"Feature rule {rule_name} unexpectedly used rank_fill.")
        write_selected_features(run_dir / f"selected_features_{rule_name}.txt", selected_features)

        preprocessor = TabularPreprocessor(selected_features).fit(
            train_part,
            missing_fill=str(prep_cfg.get("missing_fill", "median")),
            standardize=bool(prep_cfg.get("standardize", True)),
        )
        x_train = preprocessor.transform(train_part)
        x_valid = preprocessor.transform(valid_part)
        final_preprocessor = TabularPreprocessor(selected_features).fit(
            train_df,
            missing_fill=str(prep_cfg.get("missing_fill", "median")),
            standardize=bool(prep_cfg.get("standardize", True)),
        )
        x_full = final_preprocessor.transform(train_df)
        x_test = final_preprocessor.transform(test_df)
        model, valid_pred, final_params, row = train_xgb_model(
            x_train,
            y_train,
            x_valid,
            y_valid,
            xgb_cfg,
            default_variant,
            random_state=2026,
        )
        test_pred = fit_xgb_predict(final_params, x_full, y_full, x_test)
        save_valid_prediction(pred_dir / f"valid_stageA_{rule_name}_xgboost.csv", y_valid, valid_pred)
        signal_submission = make_submission(sample_submission, test_df, test_pred, prediction_col, id_col)
        signal_submission.to_csv(signal_submission_dir / f"submission_stageA_{rule_name}_xgboost.csv", index=False)

        stats = selected_feature_stats(stable_table)
        signal_cache[f"stageA_{rule_name}_rmse_es"] = {
            "stage": "stageA",
            "feature_rule": rule_name,
            "training_variant": "rmse_es",
            "features": selected_features,
            "valid": valid_pred,
            "test": test_pred,
            "params": final_params,
            "metric": row,
            **stats,
        }
        feature_rule_metrics.append(
            {
                "feature_rule": rule_name,
                "fold_top_n": fold_top_n,
                "min_fold_appearances": int(rule["min_fold_appearances"]),
                "fill_to_n": int(rule["fill_to_n"]),
                **stats,
                "signal_pearson": pearson_corr(y_valid, valid_pred),
                "signal_rmse": rmse(y_valid, valid_pred),
                "signal_delta_base": pearson_corr(y_valid, valid_pred) - base_holdout,
                "best_iteration": row["best_iteration"],
                "eval_metric": row["eval_metric"],
            }
        )
        if rule_name == str(current_cfg["baseline_rule"]):
            baseline_signal = signal_cache[f"stageA_{rule_name}_rmse_es"]
        del preprocessor, final_preprocessor, x_train, x_valid, x_full, x_test, model
        gc.collect()

    if baseline_signal is None:
        raise ValueError(f"Baseline rule {current_cfg['baseline_rule']!r} was not produced.")
    current_best_valid = (
        float(current_cfg["base_weight"]) * base_valid
        + float(current_cfg["signal_weight"]) * baseline_signal["valid"]
    )
    current_best_test = (
        float(current_cfg["base_weight"]) * base_test
        + float(current_cfg["signal_weight"]) * baseline_signal["test"]
    )

    # Recompute Stage A candidate deltas now that current best is known.
    candidate_rows = []
    for key, signal in signal_cache.items():
        if signal["stage"] != "stageA":
            continue
        add_candidate(
            candidate_rows,
            "stageA",
            f"{key}_w{safe_name(float(blend_cfg['stage_a_signal_weight']))}",
            signal["feature_rule"],
            signal["training_variant"],
            float(blend_cfg["stage_a_signal_weight"]),
            base_valid,
            base_test,
            signal["valid"],
            signal["test"],
            current_best_valid,
            current_best_test,
            y_valid,
            sample_submission,
            test_df,
            prediction_col,
            id_col,
            submission_dir,
        )

    feature_rule_df = pd.DataFrame(feature_rule_metrics)
    feature_rule_df.to_csv(run_dir / "feature_rule_metrics.csv", index=False)
    stage_a_candidates = pd.DataFrame(candidate_rows)
    stage_a_candidates.to_csv(run_dir / "stage_a_candidate_metrics.csv", index=False)

    if (stage_a_candidates["holdout_delta_current_best"] > 0).any():
        best_stage_a = stage_a_candidates.sort_values("holdout_pearson", ascending=False).iloc[0]
    else:
        best_stage_a = stage_a_candidates[stage_a_candidates["feature_rule"] == str(current_cfg["baseline_rule"])].iloc[0]
    best_rule = str(best_stage_a["feature_rule"])
    best_rule_signal = signal_cache[f"stageA_{best_rule}_rmse_es"]

    training_metrics: list[dict[str, Any]] = []
    for variant in config["training_variants"]:
        variant_name = str(variant["name"])
        print(f"\n=== Stage B training variant {variant_name} on {best_rule} ===")
        selected_features = best_rule_signal["features"]
        preprocessor = TabularPreprocessor(selected_features).fit(
            train_part,
            missing_fill=str(prep_cfg.get("missing_fill", "median")),
            standardize=bool(prep_cfg.get("standardize", True)),
        )
        x_train = preprocessor.transform(train_part)
        x_valid = preprocessor.transform(valid_part)
        final_preprocessor = TabularPreprocessor(selected_features).fit(
            train_df,
            missing_fill=str(prep_cfg.get("missing_fill", "median")),
            standardize=bool(prep_cfg.get("standardize", True)),
        )
        x_full = final_preprocessor.transform(train_df)
        x_test = final_preprocessor.transform(test_df)
        model, valid_pred, final_params, row = train_xgb_model(
            x_train,
            y_train,
            x_valid,
            y_valid,
            xgb_cfg,
            variant,
            random_state=3026,
        )
        test_pred = fit_xgb_predict(final_params, x_full, y_full, x_test)
        save_valid_prediction(pred_dir / f"valid_stageB_{best_rule}_{variant_name}.csv", y_valid, valid_pred)
        signal_submission = make_submission(sample_submission, test_df, test_pred, prediction_col, id_col)
        signal_submission.to_csv(signal_submission_dir / f"submission_stageB_{best_rule}_{variant_name}.csv", index=False)
        key = f"stageB_{best_rule}_{variant_name}"
        signal_cache[key] = {
            "stage": "stageB",
            "feature_rule": best_rule,
            "training_variant": variant_name,
            "features": selected_features,
            "valid": valid_pred,
            "test": test_pred,
            "params": final_params,
            "metric": row,
            "selected_count": best_rule_signal["selected_count"],
            "stable_count": best_rule_signal["stable_count"],
            "rank_fill_count": best_rule_signal["rank_fill_count"],
        }
        training_metrics.append(
            {
                "feature_rule": best_rule,
                "training_variant": variant_name,
                "mode": str(variant["mode"]),
                "signal_pearson": pearson_corr(y_valid, valid_pred),
                "signal_rmse": rmse(y_valid, valid_pred),
                "best_iteration": row["best_iteration"],
                "eval_metric": row["eval_metric"],
                "params": row["params"],
            }
        )
        add_candidate(
            candidate_rows,
            "stageB",
            f"{key}_w{safe_name(float(blend_cfg['stage_b_signal_weight']))}",
            best_rule,
            variant_name,
            float(blend_cfg["stage_b_signal_weight"]),
            base_valid,
            base_test,
            valid_pred,
            test_pred,
            current_best_valid,
            current_best_test,
            y_valid,
            sample_submission,
            test_df,
            prediction_col,
            id_col,
            submission_dir,
        )
        del preprocessor, final_preprocessor, x_train, x_valid, x_full, x_test, model
        gc.collect()

    pd.DataFrame(training_metrics).to_csv(run_dir / "xgb_training_metrics.csv", index=False)

    pre_stage_c = pd.DataFrame(candidate_rows)
    signal_candidates = pre_stage_c[pre_stage_c["stage"].isin(["stageA", "stageB"])].copy()
    best_signal_row = signal_candidates.sort_values("holdout_pearson", ascending=False).iloc[0]
    best_key = str(best_signal_row["candidate"]).rsplit("_w", 1)[0]
    if best_key not in signal_cache:
        # Stage A keys have candidate names identical to cache keys plus weight suffix.
        best_key = f"{str(best_signal_row['stage'])}_{str(best_signal_row['feature_rule'])}_{str(best_signal_row['training_variant'])}"
    best_signal = signal_cache[best_key]

    for weight in [float(value) for value in blend_cfg["stage_c_signal_weights"]]:
        add_candidate(
            candidate_rows,
            "stageC",
            f"stageC_{best_signal['feature_rule']}_{best_signal['training_variant']}_w{safe_name(weight)}",
            best_signal["feature_rule"],
            best_signal["training_variant"],
            weight,
            base_valid,
            base_test,
            best_signal["valid"],
            best_signal["test"],
            current_best_valid,
            current_best_test,
            y_valid,
            sample_submission,
            test_df,
            prediction_col,
            id_col,
            submission_dir,
        )

    candidates = pd.DataFrame(candidate_rows)
    candidates["eligible"] = (
        (candidates["holdout_delta_current_best"] >= -float(blend_cfg.get("holdout_tolerance", 0.0015)))
        & (candidates["std_ratio_current_best"] <= float(blend_cfg.get("max_std_ratio_current_best", 1.2)))
    )
    candidates = candidates.sort_values(
        ["eligible", "holdout_pearson", "submission_corr_current_best"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    candidates.to_csv(run_dir / "blend_candidate_metrics.csv", index=False)

    selected_names: list[str] = []
    selected_reasons: dict[str, str] = {}
    eligible = candidates[candidates["eligible"]]
    pools = [
        ("holdout_best", eligible),
        (
            "lower_corr_near_best",
            eligible[
                eligible["holdout_pearson"] >= eligible["holdout_pearson"].max() - float(blend_cfg.get("holdout_tolerance", 0.0015))
            ].sort_values(["submission_corr_current_best", "holdout_pearson"], ascending=[True, False]),
        ),
        (
            "near_w025_stable",
            eligible.assign(weight_gap=(eligible["signal_weight"] - float(current_cfg["signal_weight"])).abs()).sort_values(
                ["weight_gap", "holdout_pearson"],
                ascending=[True, False],
            ),
        ),
    ]
    for reason, pool in pools:
        if len(selected_names) >= int(blend_cfg.get("max_selected_submissions", 3)) or pool.empty:
            continue
        for _, row in pool.iterrows():
            candidate = str(row["candidate"])
            if candidate not in selected_names:
                selected_names.append(candidate)
                selected_reasons[candidate] = reason
                break
    if len(selected_names) < int(blend_cfg.get("max_selected_submissions", 3)):
        for _, row in eligible.iterrows():
            candidate = str(row["candidate"])
            if candidate not in selected_names:
                selected_names.append(candidate)
                selected_reasons[candidate] = "overall_best"
            if len(selected_names) >= int(blend_cfg.get("max_selected_submissions", 3)):
                break

    selected = candidates[candidates["candidate"].isin(selected_names)].copy()
    selected["selection_reason"] = selected["candidate"].map(selected_reasons)
    if selected_names:
        selected = selected.set_index("candidate").loc[selected_names].reset_index()
        shutil.copyfile(ROOT / str(selected.iloc[0]["path"]), submission_dir / "submission_best.csv")
    selected.to_csv(run_dir / "selected_candidates.csv", index=False)

    write_json(
        run_dir / "final_summary.json",
        {
            "base_signal": base_cfg,
            "current_best": current_cfg,
            "base_holdout_pearson": base_holdout,
            "current_best_holdout_pearson": pearson_corr(y_valid, current_best_valid),
            "best_stage_a": best_stage_a.to_dict(),
            "best_signal_for_stage_c": best_signal_row.to_dict(),
            "selected_candidates": selected_names,
            "selected_feature_rule": best_rule,
            "shap_backend": "xgboost_pred_contribs",
        },
    )

    print("\n=== Feature Rule Metrics ===")
    print(feature_rule_df.to_string(index=False))
    print("\n=== XGB Training Metrics ===")
    print(pd.DataFrame(training_metrics).to_string(index=False))
    print("\n=== Selected Candidates ===")
    print(selected.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
