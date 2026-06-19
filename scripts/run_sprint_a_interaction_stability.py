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

from drw_crypto.interaction_features import (  # noqa: E402
    InteractionDefinition,
    definitions_to_frame,
    generate_pairwise_interactions,
    prune_correlated_interactions,
    transform_interaction_block,
    transform_interactions,
)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Sprint-A fold-stable interaction selection.")
    parser.add_argument(
        "--config",
        default="configs/01_main_sprint_a_interaction_stability.yaml",
        help="Path to YAML config.",
    )
    parser.add_argument("--max-train-rows", type=int, default=None, help="Optional tail sample for smoke runs.")
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_prediction_csv(path: Path, prediction_col: str, expected_len: int | None = None) -> np.ndarray | None:
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    if prediction_col in frame.columns:
        pred = frame[prediction_col].to_numpy(dtype=np.float64)
    elif "prediction" in frame.columns:
        pred = frame["prediction"].to_numpy(dtype=np.float64)
    else:
        pred = frame.iloc[:, -1].to_numpy(dtype=np.float64)
    if expected_len is not None and len(pred) != expected_len:
        return None
    return pred


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


def train_xgb_valid_prediction(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    params: dict[str, Any],
    random_state: int,
) -> np.ndarray:
    import xgboost as xgb

    model_params = xgb_params(params, random_state)
    early_stopping_rounds = int(params.get("early_stopping_rounds", 80))
    if early_stopping_rounds > 0:
        model_params["early_stopping_rounds"] = early_stopping_rounds
    model = xgb.XGBRegressor(**model_params)
    try:
        model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=False)
    except TypeError:
        model_params.pop("early_stopping_rounds", None)
        model = xgb.XGBRegressor(**model_params)
        model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=False)
    return model.predict(x_valid)


def build_purged_group_masks(n_rows: int, n_groups: int, gap_groups: int) -> tuple[list[np.ndarray], pd.DataFrame]:
    if n_groups < 2:
        raise ValueError("n_groups must be at least 2.")
    positions = np.arange(n_rows, dtype=np.int64)
    group_ids = np.minimum((positions * int(n_groups)) // int(n_rows), int(n_groups) - 1)
    masks: list[np.ndarray] = []
    rows: list[dict[str, int]] = []
    for fold in range(int(n_groups)):
        excluded = np.abs(group_ids - fold) <= int(gap_groups)
        train_idx = positions[~excluded]
        valid_idx = positions[group_ids == fold]
        masks.append(train_idx)
        rows.append(
            {
                "fold": fold,
                "valid_group": fold,
                "gap_groups": int(gap_groups),
                "train_rows": int(len(train_idx)),
                "valid_rows": int(len(valid_idx)),
                "excluded_rows": int(np.sum(excluded)),
                "train_start": int(train_idx[0]) if len(train_idx) else -1,
                "train_end": int(train_idx[-1]) if len(train_idx) else -1,
            }
        )
    return masks, pd.DataFrame(rows)


def _pearson_columns(values: np.ndarray, target: np.ndarray) -> np.ndarray:
    target64 = target.astype(np.float64, copy=False)
    centered = target64 - target64.mean()
    target_std = float(target64.std())
    mean = values.mean(axis=0, dtype=np.float64)
    mean_sq = np.square(values, dtype=np.float64).mean(axis=0)
    std = np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))
    cov = (values.astype(np.float64).T @ centered) / len(centered)
    denom = std * target_std
    corr = np.divide(cov, denom, out=np.zeros_like(cov, dtype=np.float64), where=denom > 0)
    corr[~np.isfinite(corr)] = 0.0
    return corr


def score_interactions_by_fold(
    base_values: np.ndarray,
    definitions: list[InteractionDefinition],
    label: np.ndarray,
    residual: np.ndarray,
    fold_train_indices: list[np.ndarray],
    eps: float,
    clip_quantiles: tuple[float, float],
    block_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[InteractionDefinition]]:
    n_defs = len(definitions)
    n_folds = len(fold_train_indices)
    abs_label = np.zeros((n_defs, n_folds), dtype=np.float32)
    abs_residual = np.zeros((n_defs, n_folds), dtype=np.float32)
    fitted_definitions: list[InteractionDefinition] = []

    for start in range(0, n_defs, int(block_size)):
        block_defs = definitions[start : start + int(block_size)]
        block_values, fitted_block = transform_interaction_block(
            base_values,
            block_defs,
            eps=eps,
            fit_clips=True,
            clip_quantiles=clip_quantiles,
        )
        for fold, idx in enumerate(fold_train_indices):
            values = block_values[idx]
            abs_label[start : start + len(block_defs), fold] = np.abs(_pearson_columns(values, label[idx])).astype(
                np.float32
            )
            abs_residual[start : start + len(block_defs), fold] = np.abs(
                _pearson_columns(values, residual[idx])
            ).astype(np.float32)
        fitted_definitions.extend(fitted_block)

    label_ranks = np.zeros_like(abs_label, dtype=np.float32)
    residual_ranks = np.zeros_like(abs_residual, dtype=np.float32)
    top_rows: list[dict[str, Any]] = []
    for fold in range(n_folds):
        label_order = np.argsort(-abs_label[:, fold], kind="mergesort")
        residual_order = np.argsort(-abs_residual[:, fold], kind="mergesort")
        label_ranks[label_order, fold] = np.arange(1, n_defs + 1, dtype=np.float32)
        residual_ranks[residual_order, fold] = np.arange(1, n_defs + 1, dtype=np.float32)
        for metric_name, order, values in [
            ("abs_pearson_label", label_order, abs_label[:, fold]),
            ("abs_pearson_residual", residual_order, abs_residual[:, fold]),
        ]:
            for rank, idx in enumerate(order[: min(300, n_defs)], start=1):
                definition = fitted_definitions[int(idx)]
                top_rows.append(
                    {
                        "fold": fold,
                        "metric": metric_name,
                        "rank": rank,
                        "feature": definition.name,
                        "left": definition.left,
                        "right": definition.right,
                        "operation": definition.operation,
                        "score": float(values[int(idx)]),
                    }
                )

    aggregate_rows: list[dict[str, Any]] = []
    for idx, definition in enumerate(fitted_definitions):
        best_fold_scores = np.maximum(abs_label[idx], abs_residual[idx])
        best_fold_ranks = np.minimum(label_ranks[idx], residual_ranks[idx])
        aggregate_rows.append(
            {
                "feature": definition.name,
                "left": definition.left,
                "right": definition.right,
                "operation": definition.operation,
                "left_index": definition.left_index,
                "right_index": definition.right_index,
                "clip_low": definition.clip_low,
                "clip_high": definition.clip_high,
                "mean_abs_pearson_label": float(np.mean(abs_label[idx])),
                "std_abs_pearson_label": float(np.std(abs_label[idx])),
                "min_abs_pearson_label": float(np.min(abs_label[idx])),
                "mean_abs_pearson_residual": float(np.mean(abs_residual[idx])),
                "std_abs_pearson_residual": float(np.std(abs_residual[idx])),
                "min_abs_pearson_residual": float(np.min(abs_residual[idx])),
                "mean_best_fold_score": float(np.mean(best_fold_scores)),
                "min_best_fold_score": float(np.min(best_fold_scores)),
                "mean_best_fold_rank": float(np.mean(best_fold_ranks)),
                "min_best_fold_rank": float(np.min(best_fold_ranks)),
                "best_metric": (
                    "mean_abs_pearson_label"
                    if float(np.mean(abs_label[idx])) >= float(np.mean(abs_residual[idx]))
                    else "mean_abs_pearson_residual"
                ),
                "max_abs_score": float(np.mean(best_fold_scores)),
                "best_rank": float(np.mean(best_fold_ranks)),
            }
        )

    scores = pd.DataFrame(aggregate_rows)
    scores["global_rank"] = scores["mean_best_fold_rank"].rank(method="min", ascending=True)
    scores = scores.sort_values(
        ["mean_best_fold_rank", "mean_best_fold_score", "feature"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    return scores, pd.DataFrame(top_rows), fitted_definitions


def apply_feature_rule(
    scores: pd.DataFrame,
    rule: dict[str, Any],
    max_selected: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    fold_top_n = int(rule["fold_top_n"])
    min_appear = int(rule["min_fold_appearances"])
    fill_to_n = int(rule.get("fill_to_n", 0))
    rule_scores = scores.copy()
    label_cols = [col for col in rule_scores.columns if col.startswith("label_rank_fold")]
    residual_cols = [col for col in rule_scores.columns if col.startswith("residual_rank_fold")]
    if not label_cols or not residual_cols:
        raise ValueError("Rule scores must contain per-fold rank columns.")
    appearance = np.zeros(len(rule_scores), dtype=np.int32)
    for label_col, residual_col in zip(label_cols, residual_cols):
        appearance += (
            np.minimum(rule_scores[label_col].to_numpy(dtype=np.float64), rule_scores[residual_col].to_numpy(dtype=np.float64))
            <= fold_top_n
        ).astype(np.int32)
    rule_scores["fold_appearances"] = appearance
    rule_scores["selected_by"] = ""
    stable = rule_scores["fold_appearances"] >= min_appear
    rule_scores.loc[stable, "selected_by"] = "stable"
    selected = rule_scores[stable].sort_values(
        ["mean_best_fold_rank", "mean_best_fold_score", "feature"],
        ascending=[True, False, True],
    )
    stable_count = int(len(selected))
    fill_count = 0
    if fill_to_n > 0 and len(selected) < fill_to_n:
        needed = min(fill_to_n, max_selected) - len(selected)
        if needed > 0:
            fill = rule_scores[~stable].sort_values(
                ["mean_best_fold_rank", "mean_best_fold_score", "feature"],
                ascending=[True, False, True],
            ).head(needed)
            rule_scores.loc[fill.index, "selected_by"] = "rank_fill"
            selected = pd.concat([selected, fill], axis=0)
            fill_count = int(len(fill))
    selected = selected.head(int(max_selected)).reset_index(drop=True)
    summary = {
        "rule": str(rule["name"]),
        "fold_top_n": fold_top_n,
        "min_fold_appearances": min_appear,
        "fill_to_n": fill_to_n,
        "stable_count": stable_count,
        "rank_fill_count": fill_count,
        "pre_prune_count": int(len(selected)),
    }
    return selected, summary


def add_rank_columns(scores: pd.DataFrame, abs_label: np.ndarray, abs_residual: np.ndarray) -> pd.DataFrame:
    out = scores.copy()
    n_defs, n_folds = abs_label.shape
    for fold in range(n_folds):
        label_order = np.argsort(-abs_label[:, fold], kind="mergesort")
        residual_order = np.argsort(-abs_residual[:, fold], kind="mergesort")
        label_rank = np.empty(n_defs, dtype=np.float32)
        residual_rank = np.empty(n_defs, dtype=np.float32)
        label_rank[label_order] = np.arange(1, n_defs + 1, dtype=np.float32)
        residual_rank[residual_order] = np.arange(1, n_defs + 1, dtype=np.float32)
        out[f"label_rank_fold{fold}"] = label_rank
        out[f"residual_rank_fold{fold}"] = residual_rank
    return out


def reconstruct_structured_base_fit_valid(
    train_part: pd.DataFrame,
    valid_part: pd.DataFrame,
    y_train: np.ndarray,
    y_valid: np.ndarray,
    current_cfg: dict[str, Any],
    prep_cfg: dict[str, Any],
    xgb_fallback_cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    beta4_cfg = current_cfg["beta4_base"]
    beta5_cfg = current_cfg["beta5_interaction"]
    beta3_fit = weighted_ridge_prediction(train_part, y_train, train_part, beta4_cfg["beta3_components"], prep_cfg, ROOT)
    beta3_valid = weighted_ridge_prediction(train_part, y_train, valid_part, beta4_cfg["beta3_components"], prep_cfg, ROOT)

    shap_features = read_lines(ROOT / beta4_cfg["shap_stable_features_path"])
    shap_preprocessor = TabularPreprocessor(shap_features).fit(
        train_part,
        missing_fill=str(prep_cfg.get("missing_fill", "median")),
        standardize=bool(prep_cfg.get("standardize", True)),
    )
    x_shap_train = shap_preprocessor.transform(train_part)
    x_shap_valid = shap_preprocessor.transform(valid_part)
    shap_valid = load_prediction_csv(ROOT / beta4_cfg["shap_stable_valid_prediction_path"], "prediction", len(y_valid))
    if shap_valid is None:
        shap_valid = train_xgb_valid_prediction(x_shap_train, y_train, x_shap_valid, y_valid, xgb_fallback_cfg, 2026)
    import xgboost as xgb

    shap_params = xgb_params(xgb_fallback_cfg, 2026)
    shap_params["early_stopping_rounds"] = int(xgb_fallback_cfg.get("early_stopping_rounds", 80))
    shap_model = xgb.XGBRegressor(**shap_params)
    try:
        shap_model.fit(x_shap_train, y_train, eval_set=[(x_shap_valid, y_valid)], verbose=False)
    except TypeError:
        shap_params.pop("early_stopping_rounds", None)
        shap_model = xgb.XGBRegressor(**shap_params)
        shap_model.fit(x_shap_train, y_train, eval_set=[(x_shap_valid, y_valid)], verbose=False)
    shap_fit = shap_model.predict(x_shap_train)
    beta4_fit = float(beta4_cfg["base_weight"]) * beta3_fit + float(beta4_cfg["signal_weight"]) * shap_fit
    beta4_valid = float(beta4_cfg["base_weight"]) * beta3_valid + float(beta4_cfg["signal_weight"]) * shap_valid

    interaction_valid = load_prediction_csv(ROOT / beta5_cfg["valid_prediction_path"], "prediction", len(y_valid))
    interaction_fit = beta4_fit
    if interaction_valid is None:
        interaction_valid = beta4_valid
    structured_fit = (1.0 - float(beta5_cfg["weight"])) * beta4_fit + float(beta5_cfg["weight"]) * interaction_fit
    structured_valid = (1.0 - float(beta5_cfg["weight"])) * beta4_valid + float(beta5_cfg["weight"]) * interaction_valid
    return structured_fit, structured_valid


def reconstruct_current_best_valid(
    train_part: pd.DataFrame,
    valid_part: pd.DataFrame,
    y_train: np.ndarray,
    y_valid: np.ndarray,
    current_cfg: dict[str, Any],
    prep_cfg: dict[str, Any],
    xgb_fallback_cfg: dict[str, Any],
    smoke_mode: bool,
) -> tuple[np.ndarray, np.ndarray, str]:
    structured_fit, structured_valid = reconstruct_structured_base_fit_valid(
        train_part,
        valid_part,
        y_train,
        y_valid,
        current_cfg,
        prep_cfg,
        xgb_fallback_cfg,
    )
    beta6_cfg = current_cfg["beta6_2_supervised_ae"]
    mlp_cfg = current_cfg["sprint_mlp"]
    ae_valid = load_prediction_csv(ROOT / beta6_cfg["valid_prediction_path"], "prediction", len(y_valid))
    mlp_valid = load_prediction_csv(ROOT / mlp_cfg["valid_prediction_path"], "prediction", len(y_valid))
    if ae_valid is None or mlp_valid is None:
        reference = "structured_base_fallback" if smoke_mode else "structured_base_fallback_missing_representation"
        return structured_fit, structured_valid, reference
    ae_weight = float(beta6_cfg["weight"])
    beta6_valid = (1.0 - ae_weight) * structured_valid + ae_weight * ae_valid
    mlp_weight = float(mlp_cfg["weight"])
    current_valid = (1.0 - mlp_weight) * beta6_valid + mlp_weight * mlp_valid
    return structured_fit, current_valid, "sprint_current_best"


def summarize_delta(pred: np.ndarray, reference: np.ndarray, prefix: str) -> dict[str, float]:
    delta = pred - reference
    return {
        f"{prefix}_corr_current_best": pearson_corr(reference, pred),
        f"{prefix}_delta_std_current_best": float(np.std(delta)),
        f"{prefix}_delta_mae_current_best": float(np.mean(np.abs(delta))),
    }


def add_blend_candidate(
    rows: list[dict[str, Any]],
    rule_name: str,
    signal_name: str,
    signal_weight: float,
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
    current_weight = 1.0 - float(signal_weight)
    blend_valid = current_weight * current_best_valid + float(signal_weight) * signal_valid
    blend_test = current_weight * current_best_test + float(signal_weight) * signal_test
    candidate = f"{rule_name}_{signal_name}_w{safe_name(signal_weight)}"
    submission = make_submission(sample_submission, test_df, blend_test, prediction_col, id_col)
    path = submission_dir / f"submission_{candidate}.csv"
    submission.to_csv(path, index=False)
    holdout = pearson_corr(y_valid, blend_valid)
    rows.append(
        {
            "candidate": candidate,
            "rule": rule_name,
            "signal": signal_name,
            "current_best_weight": current_weight,
            "signal_weight": float(signal_weight),
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


def select_candidates(candidates: pd.DataFrame, max_selected: int) -> pd.DataFrame:
    eligible = candidates[candidates["eligible"]].copy()
    if eligible.empty:
        return eligible
    selected_names: list[str] = []
    reasons: dict[str, str] = {}

    holdout_pool = eligible.sort_values(["holdout_pearson", "submission_corr_current_best"], ascending=[False, True])
    if not holdout_pool.empty:
        candidate = str(holdout_pool.iloc[0]["candidate"])
        selected_names.append(candidate)
        reasons[candidate] = "holdout_best"

    low_corr_pool = eligible[~eligible["candidate"].isin(selected_names)].sort_values(
        ["submission_corr_current_best", "holdout_pearson"],
        ascending=[True, False],
    )
    if len(selected_names) < int(max_selected) and not low_corr_pool.empty:
        candidate = str(low_corr_pool.iloc[0]["candidate"])
        selected_names.append(candidate)
        reasons[candidate] = "lower_corr_backup"

    selected = eligible[eligible["candidate"].isin(selected_names)].copy()
    selected["selection_reason"] = selected["candidate"].map(reasons)
    return selected.set_index("candidate").loc[selected_names].reset_index()


def main() -> int:
    args = parse_args()
    config_path = ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)
    config = load_config(config_path)

    data_cfg = config["data"]
    prep_cfg = config["preprocessing"]
    current_cfg = config["current_best"]
    input_cfg = config["input"]
    interaction_cfg = config["interactions"]
    fold_cfg = config["fold_stability"]
    ridge_cfg = config["ridge"]
    blend_cfg = config["blend"]
    output_cfg = config["output"]
    xgb_fallback_cfg = config["xgboost_fallback"]

    run_dir = ensure_dir(ROOT / output_cfg["run_dir"])
    pred_dir = ensure_dir(run_dir / "valid_predictions")
    submission_dir = ensure_dir(ROOT / output_cfg["submission_dir"])
    signal_submission_dir = ensure_dir(submission_dir / "signals")

    raw_dir = ROOT / data_cfg["raw_dir"]
    train_path = find_first_existing(raw_dir, data_cfg.get("train_file"), TRAIN_CANDIDATES)
    test_path = find_first_existing(raw_dir, data_cfg.get("test_file"), TEST_CANDIDATES)
    sample_path = find_first_existing(raw_dir, data_cfg.get("sample_submission_file"), SAMPLE_SUBMISSION_CANDIDATES)
    if train_path is None or test_path is None:
        raise FileNotFoundError("Missing train/test files under data/raw.")

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
    prediction_col = choose_prediction_column(sample_submission, output_cfg.get("prediction_col"), target_col)

    base_features = read_lines(ROOT / input_cfg["base_feature_pool_path"])
    invalid = sorted(set(base_features) - set(feature_cols))
    if invalid:
        raise ValueError(f"Base feature pool contains unknown/non-feature columns: {invalid[:10]}")
    if target_col in base_features or (id_col and id_col in base_features):
        raise ValueError("Base feature pool must not include target or id columns.")
    write_selected_features(run_dir / "base_feature_pool.txt", base_features)

    train_part, valid_part = split_time_ordered(train_df, float(data_cfg["validation_fraction"]))
    y_train = train_part[target_col].to_numpy(dtype=np.float64)
    y_valid = valid_part[target_col].to_numpy(dtype=np.float64)
    y_full = train_df[target_col].to_numpy(dtype=np.float64)

    print("Reconstructing current best validation reference and structured residual reference.")
    residual_reference_fit, current_best_valid, current_reference = reconstruct_current_best_valid(
        train_part,
        valid_part,
        y_train,
        y_valid,
        current_cfg,
        prep_cfg,
        xgb_fallback_cfg,
        smoke_mode=args.max_train_rows is not None,
    )
    current_best_test = load_prediction_csv(ROOT / current_cfg["test_submission_path"], prediction_col, len(test_df))
    if current_best_test is None:
        raise FileNotFoundError(f"Could not load current-best test predictions from {current_cfg['test_submission_path']}.")
    current_best_holdout = pearson_corr(y_valid, current_best_valid)
    residual = y_train - residual_reference_fit

    base_preprocessor = TabularPreprocessor(base_features).fit(
        train_part,
        missing_fill=str(prep_cfg.get("missing_fill", "median")),
        standardize=bool(prep_cfg.get("standardize", True)),
    )
    x_base_train = base_preprocessor.transform(train_part)
    x_base_valid = base_preprocessor.transform(valid_part)
    x_base_full = base_preprocessor.transform(train_df)
    x_base_test = base_preprocessor.transform(test_df)

    definitions = generate_pairwise_interactions(base_features, [str(op) for op in interaction_cfg["operations"]])
    fold_indices, fold_summary = build_purged_group_masks(
        len(train_part),
        int(fold_cfg["n_groups"]),
        int(fold_cfg["gap_groups"]),
    )
    fold_summary.to_csv(run_dir / "fold_group_summary.csv", index=False)

    print(f"Scoring {len(definitions)} interactions across {len(fold_indices)} purged groups.")
    start = time.perf_counter()
    scores, fold_top, fitted_definitions = score_interactions_by_fold(
        x_base_train,
        definitions,
        y_train,
        residual,
        fold_indices,
        eps=float(interaction_cfg.get("eps", 1e-6)),
        clip_quantiles=(float(interaction_cfg["clip_quantiles"][0]), float(interaction_cfg["clip_quantiles"][1])),
        block_size=int(interaction_cfg.get("score_block_size", 64)),
    )
    elapsed = time.perf_counter() - start

    # Reconstruct rank columns from saved fold-top-independent arrays by recomputing from top table is lossy,
    # so compute rule-specific appearances using per-fold top tables below.
    fold_top.to_csv(run_dir / "interaction_fold_top_features.csv", index=False)
    scores.to_csv(run_dir / "interaction_stability_scores.csv", index=False)

    definition_lookup = {definition.name: definition for definition in fitted_definitions}
    score_lookup = scores.set_index("feature")
    top_feature_sets: dict[tuple[int, int], set[str]] = {}
    for rule in fold_cfg["rules"]:
        top_n = int(rule["fold_top_n"])
        for fold in range(int(fold_cfg["n_groups"])):
            key = (fold, top_n)
            if key in top_feature_sets:
                continue
            subset = fold_top[(fold_top["fold"] == fold) & (fold_top["rank"] <= top_n)]
            top_feature_sets[key] = set(subset["feature"].astype(str).tolist())

    selected_rule_rows: list[dict[str, Any]] = []
    signal_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    rule_metric_rows: list[dict[str, Any]] = []

    for rule in fold_cfg["rules"]:
        rule_name = str(rule["name"])
        top_n = int(rule["fold_top_n"])
        min_appear = int(rule["min_fold_appearances"])
        fill_to_n = int(rule.get("fill_to_n", 0))
        rule_scores = scores.copy()
        appearances: list[int] = []
        for feature in rule_scores["feature"].astype(str):
            count = sum(feature in top_feature_sets[(fold, top_n)] for fold in range(int(fold_cfg["n_groups"])))
            appearances.append(int(count))
        rule_scores["fold_appearances"] = appearances
        rule_scores["selected_by"] = ""
        stable_mask = rule_scores["fold_appearances"] >= min_appear
        rule_scores.loc[stable_mask, "selected_by"] = "stable"
        selected_scores = rule_scores[stable_mask].sort_values(
            ["mean_best_fold_rank", "mean_best_fold_score", "feature"],
            ascending=[True, False, True],
        )
        stable_count = int(len(selected_scores))
        rank_fill_count = 0
        if fill_to_n > 0 and len(selected_scores) < fill_to_n:
            needed = min(fill_to_n, int(interaction_cfg["max_selected_features"])) - len(selected_scores)
            fill_scores = rule_scores[~stable_mask].sort_values(
                ["mean_best_fold_rank", "mean_best_fold_score", "feature"],
                ascending=[True, False, True],
            ).head(max(needed, 0))
            rule_scores.loc[fill_scores.index, "selected_by"] = "rank_fill"
            selected_scores = pd.concat([selected_scores, fill_scores], axis=0)
            rank_fill_count = int(len(fill_scores))
        selected_scores = selected_scores.head(int(interaction_cfg["max_selected_features"])).reset_index(drop=True)
        selected_scores.to_csv(run_dir / f"interaction_scores_{rule_name}.csv", index=False)

        candidate_definitions = [definition_lookup[str(name)] for name in selected_scores["feature"]]
        selected_definitions, pruning_log = prune_correlated_interactions(
            x_base_train,
            candidate_definitions,
            selected_scores,
            eps=float(interaction_cfg.get("eps", 1e-6)),
            corr_threshold=float(interaction_cfg["corr_prune_threshold"]),
            max_features=int(interaction_cfg["max_selected_features"]),
        )
        pruning_log.to_csv(run_dir / f"interaction_correlation_pruning_{rule_name}.csv", index=False)
        selected_frame = definitions_to_frame(selected_definitions)
        selected_frame.to_csv(run_dir / f"selected_interaction_definitions_{rule_name}.csv", index=False)
        write_selected_features(
            run_dir / f"selected_interaction_features_{rule_name}.txt",
            selected_frame["feature"].astype(str).tolist(),
        )

        selected_rule_rows.append(
            {
                "rule": rule_name,
                "fold_top_n": top_n,
                "min_fold_appearances": min_appear,
                "fill_to_n": fill_to_n,
                "stable_count": stable_count,
                "rank_fill_count": rank_fill_count,
                "pre_prune_count": int(len(selected_scores)),
                "selected_count": int(len(selected_definitions)),
            }
        )

        if not selected_definitions:
            continue

        print(f"Training Ridge for {rule_name} with {len(selected_definitions)} selected interactions.")
        x_int_train = transform_interactions(
            x_base_train,
            selected_definitions,
            eps=float(interaction_cfg.get("eps", 1e-6)),
            block_size=int(interaction_cfg.get("transform_block_size", 128)),
        )
        x_int_valid = transform_interactions(
            x_base_valid,
            selected_definitions,
            eps=float(interaction_cfg.get("eps", 1e-6)),
            block_size=int(interaction_cfg.get("transform_block_size", 128)),
        )
        x_int_full = transform_interactions(
            x_base_full,
            selected_definitions,
            eps=float(interaction_cfg.get("eps", 1e-6)),
            block_size=int(interaction_cfg.get("transform_block_size", 128)),
        )
        x_int_test = transform_interactions(
            x_base_test,
            selected_definitions,
            eps=float(interaction_cfg.get("eps", 1e-6)),
            block_size=int(interaction_cfg.get("transform_block_size", 128)),
        )

        ridge_best, ridge_valid, ridge_grid = train_ridge_search(
            x_int_train,
            y_train,
            x_int_valid,
            y_valid,
            [float(alpha) for alpha in ridge_cfg["alphas"]],
        )
        ridge_grid.to_csv(run_dir / f"ridge_alpha_search_{rule_name}.csv", index=False)
        ridge_test = fit_ridge(float(ridge_best["alpha"]), x_int_full, y_full, x_int_test)
        signal_name = f"ridge_interactions_{rule_name}"
        save_valid_prediction(pred_dir / f"valid_{signal_name}.csv", y_valid, ridge_valid)
        make_submission(sample_submission, test_df, ridge_test, prediction_col, id_col).to_csv(
            signal_submission_dir / f"submission_{signal_name}.csv",
            index=False,
        )
        signal_pearson = pearson_corr(y_valid, ridge_valid)
        signal_rows.append(
            {
                "rule": rule_name,
                "signal": signal_name,
                "alpha": float(ridge_best["alpha"]),
                "selected_interaction_count": int(len(selected_definitions)),
                "pearson": signal_pearson,
                "rmse": rmse(y_valid, ridge_valid),
                "holdout_delta_current_best": signal_pearson - current_best_holdout,
                "valid_prediction_std": float(np.std(ridge_valid)),
                "submission_prediction_std": float(np.std(ridge_test)),
                "valid_corr_current_best": pearson_corr(current_best_valid, ridge_valid),
                "submission_corr_current_best": pearson_corr(current_best_test, ridge_test),
            }
        )

        for signal_weight in [float(value) for value in blend_cfg["signal_weights"]]:
            add_blend_candidate(
                candidate_rows,
                rule_name,
                "ridge_interactions",
                signal_weight,
                ridge_valid,
                ridge_test,
                current_best_valid,
                current_best_test,
                y_valid,
                sample_submission,
                test_df,
                prediction_col,
                id_col,
                submission_dir,
            )

        rule_metric_rows.append(
            {
                "rule": rule_name,
                "best_alpha": float(ridge_best["alpha"]),
                "signal_pearson": signal_pearson,
                "signal_rmse": rmse(y_valid, ridge_valid),
                "selected_interaction_count": int(len(selected_definitions)),
                "stable_count": stable_count,
                "rank_fill_count": rank_fill_count,
            }
        )
        del x_int_train, x_int_valid, x_int_full, x_int_test
        gc.collect()

    pd.DataFrame(selected_rule_rows).to_csv(run_dir / "feature_rule_metrics.csv", index=False)
    signal_metrics = pd.DataFrame(signal_rows).sort_values(["pearson", "submission_corr_current_best"], ascending=[False, True])
    signal_metrics.to_csv(run_dir / "signal_metrics.csv", index=False)
    pd.DataFrame(rule_metric_rows).to_csv(run_dir / "rule_signal_summary.csv", index=False)

    candidates = pd.DataFrame(candidate_rows)
    candidates["eligible"] = (
        (candidates["holdout_delta_current_best"] >= float(blend_cfg["holdout_floor_delta"]))
        & (candidates["std_ratio_current_best"] >= float(blend_cfg["min_std_ratio_current_best"]))
        & (candidates["std_ratio_current_best"] <= float(blend_cfg["max_std_ratio_current_best"]))
    )
    candidates = candidates.sort_values(
        ["eligible", "holdout_pearson", "submission_corr_current_best"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    candidates.to_csv(run_dir / "blend_candidate_metrics.csv", index=False)
    selected = select_candidates(candidates, int(blend_cfg.get("max_selected_submissions", 2)))
    selected.to_csv(run_dir / "selected_candidates.csv", index=False)
    if not selected.empty:
        shutil.copyfile(ROOT / str(selected.iloc[0]["path"]), submission_dir / "submission_best.csv")

    write_json(
        run_dir / "final_summary.json",
        {
            "current_best": current_cfg,
            "current_best_reference": current_reference,
            "current_best_holdout_pearson": current_best_holdout,
            "residual_reference": "structured_base_fit_approx",
            "base_feature_count": len(base_features),
            "candidate_interaction_count": len(definitions),
            "fold_scoring_seconds": elapsed,
            "feature_rules": selected_rule_rows,
            "best_signal": signal_metrics.iloc[0].to_dict() if not signal_metrics.empty else {},
            "selected_candidates": selected.to_dict(orient="records"),
        },
    )

    print("\n=== Feature Rule Metrics ===")
    print(pd.DataFrame(selected_rule_rows).to_string(index=False))
    print("\n=== Signal Metrics ===")
    print(signal_metrics.to_string(index=False))
    print("\n=== Selected Candidates ===")
    if selected.empty:
        print("No eligible candidates selected for submission.")
    else:
        print(selected.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
