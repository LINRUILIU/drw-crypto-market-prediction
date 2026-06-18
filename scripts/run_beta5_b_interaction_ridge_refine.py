from __future__ import annotations

import argparse
import gc
import shutil
import sys
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
    candidate_pool_from_scores,
    definitions_from_frame,
    definitions_to_frame,
    prune_correlated_interactions,
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
    parser = argparse.ArgumentParser(description="Run Beta5-B interaction Ridge refinement.")
    parser.add_argument("--config", default="configs/01_main_beta5_b_interaction_ridge_refine.yaml", help="Path to YAML config.")
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


def summarize_delta(pred: np.ndarray, reference: np.ndarray, prefix: str) -> dict[str, float]:
    delta = pred - reference
    return {
        f"{prefix}_corr_current_best": pearson_corr(reference, pred),
        f"{prefix}_delta_std_current_best": float(np.std(delta)),
        f"{prefix}_delta_mae_current_best": float(np.mean(np.abs(delta))),
    }


def add_candidate(
    rows: list[dict[str, Any]],
    feature_count: int,
    alpha: float,
    signal_weight: float,
    beta4_valid: np.ndarray,
    beta4_test: np.ndarray,
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
    blend_valid = base_weight * beta4_valid + float(signal_weight) * signal_valid
    blend_test = base_weight * beta4_test + float(signal_weight) * signal_test
    candidate = f"blend_beta4_w{safe_name(base_weight)}_int{feature_count}_a{safe_name(alpha)}_w{safe_name(signal_weight)}"
    submission = make_submission(sample_submission, test_df, blend_test, prediction_col, id_col)
    path = submission_dir / f"submission_{candidate}.csv"
    submission.to_csv(path, index=False)
    holdout = pearson_corr(y_valid, blend_valid)
    rows.append(
        {
            "candidate": candidate,
            "feature_count": int(feature_count),
            "alpha": float(alpha),
            "beta4_weight": base_weight,
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


def select_candidates(candidates: pd.DataFrame, max_selected: int, baseline_feature_count: int, baseline_alpha: float, baseline_weight: float) -> pd.DataFrame:
    eligible = candidates[candidates["eligible"]].copy()
    if eligible.empty:
        return eligible

    selected_names: list[str] = []
    reasons: dict[str, str] = {}
    holdout_pool = eligible.sort_values(["holdout_pearson", "submission_corr_current_best"], ascending=[False, True])
    first = str(holdout_pool.iloc[0]["candidate"])
    selected_names.append(first)
    reasons[first] = "holdout_best"

    near_best = eligible[
        eligible["holdout_pearson"] >= float(holdout_pool.iloc[0]["holdout_pearson"]) - 0.00075
    ].sort_values(["submission_corr_current_best", "holdout_pearson"], ascending=[True, False])
    for _, row in near_best.iterrows():
        candidate = str(row["candidate"])
        if candidate not in selected_names:
            selected_names.append(candidate)
            reasons[candidate] = "lower_corr_near_best"
            break

    baseline_like = eligible[
        (eligible["feature_count"] == int(baseline_feature_count))
        & (eligible["alpha"] == float(baseline_alpha))
        & (np.isclose(eligible["signal_weight"].astype(float), float(baseline_weight)))
    ]
    if len(selected_names) < int(max_selected) and not baseline_like.empty:
        candidate = str(baseline_like.iloc[0]["candidate"])
        if candidate not in selected_names:
            selected_names.append(candidate)
            reasons[candidate] = "baseline_reproduction"

    if len(selected_names) < int(max_selected):
        for _, row in holdout_pool.iterrows():
            candidate = str(row["candidate"])
            if candidate not in selected_names:
                selected_names.append(candidate)
                reasons[candidate] = "overall_best"
            if len(selected_names) >= int(max_selected):
                break

    selected = eligible[eligible["candidate"].isin(selected_names)].copy()
    selected["selection_reason"] = selected["candidate"].map(reasons)
    return selected.set_index("candidate").loc[selected_names].reset_index()


def main() -> int:
    args = parse_args()
    config_path = ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)
    config = load_config(config_path)

    data_cfg = config["data"]
    prep_cfg = config["preprocessing"]
    beta4_cfg = config["beta4_base"]
    current_cfg = config["current_best"]
    pool_cfg = config["feature_pool"]
    interaction_cfg = config["interactions"]
    ridge_cfg = config["ridge"]
    xgb_cfg = config["xgboost"]
    blend_cfg = config["blend"]
    output_cfg = config["output"]

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

    base_features = read_lines(ROOT / pool_cfg["base_feature_pool_path"])
    invalid = sorted(set(base_features) - set(feature_cols))
    if invalid:
        raise ValueError(f"Base feature pool contains unknown/non-feature columns: {invalid[:10]}")
    scores = pd.read_csv(ROOT / pool_cfg["interaction_scores_path"])
    candidate_scores = candidate_pool_from_scores(scores, int(pool_cfg["top_k_each_metric"]))
    definitions = definitions_from_frame(candidate_scores)

    train_part, valid_part = split_time_ordered(train_df, float(data_cfg["validation_fraction"]))
    y_train = train_part[target_col].to_numpy(dtype=np.float64)
    y_valid = valid_part[target_col].to_numpy(dtype=np.float64)
    y_full = train_df[target_col].to_numpy(dtype=np.float64)

    print("Reconstructing Beta4 base and Beta5 current-best predictions.")
    beta3_valid = weighted_ridge_prediction(train_part, y_train, valid_part, beta4_cfg["beta3_components"], prep_cfg, ROOT)
    beta3_test = weighted_ridge_prediction(train_df, y_full, test_df, beta4_cfg["beta3_components"], prep_cfg, ROOT)
    shap_valid = load_prediction_csv(ROOT / beta4_cfg["shap_stable_valid_prediction_path"], "prediction", expected_len=len(y_valid))
    if shap_valid is None:
        print("SHAP-stable valid prediction length mismatch; recomputing for current train scope.")
        shap_features = read_lines(ROOT / beta4_cfg["shap_stable_features_path"])
        shap_preprocessor = TabularPreprocessor(shap_features).fit(
            train_part,
            missing_fill=str(prep_cfg.get("missing_fill", "median")),
            standardize=bool(prep_cfg.get("standardize", True)),
        )
        shap_valid = train_xgb_valid_prediction(
            shap_preprocessor.transform(train_part),
            y_train,
            shap_preprocessor.transform(valid_part),
            y_valid,
            xgb_cfg,
            random_state=2026,
        )
        del shap_preprocessor
        gc.collect()
    beta4_test = load_prediction_csv(ROOT / beta4_cfg["test_submission_path"], prediction_col, expected_len=len(test_df))
    if beta4_test is None:
        raise FileNotFoundError(f"Could not load Beta4 test predictions: {beta4_cfg['test_submission_path']}")
    beta4_valid = float(beta4_cfg["base_weight"]) * beta3_valid + float(beta4_cfg["signal_weight"]) * shap_valid

    preprocessor = TabularPreprocessor(base_features).fit(
        train_part,
        missing_fill=str(prep_cfg.get("missing_fill", "median")),
        standardize=bool(prep_cfg.get("standardize", True)),
    )
    x_base_train = preprocessor.transform(train_part)
    x_base_valid = preprocessor.transform(valid_part)
    x_base_full = preprocessor.transform(train_df)
    x_base_test = preprocessor.transform(test_df)

    old_interaction_valid = load_prediction_csv(
        ROOT / current_cfg["interaction_valid_prediction_path"],
        "prediction",
        expected_len=len(y_valid),
    )
    if old_interaction_valid is None:
        print("Beta5 interaction valid prediction length mismatch; recomputing old 120-feature alpha=200000 signal.")
        old_defs = definitions_from_frame(pd.read_csv(ROOT / "runs/01_main/beta5_interactions/selected_interaction_definitions.csv"))
        x_old_train = transform_interactions(
            x_base_train,
            old_defs,
            eps=float(interaction_cfg.get("eps", 1e-6)),
            block_size=int(interaction_cfg.get("transform_block_size", 128)),
        )
        x_old_valid = transform_interactions(
            x_base_valid,
            old_defs,
            eps=float(interaction_cfg.get("eps", 1e-6)),
            block_size=int(interaction_cfg.get("transform_block_size", 128)),
        )
        old_interaction_valid = fit_ridge(200000.0, x_old_train, y_train, x_old_valid)
        del x_old_train, x_old_valid
        gc.collect()
    current_best_valid = (
        float(current_cfg["beta4_weight"]) * beta4_valid
        + float(current_cfg["interaction_weight"]) * old_interaction_valid
    )
    current_best_test = load_prediction_csv(ROOT / current_cfg["test_submission_path"], prediction_col, expected_len=len(test_df))
    if current_best_test is None:
        raise FileNotFoundError(f"Could not load current-best test predictions: {current_cfg['test_submission_path']}")
    current_best_holdout = pearson_corr(y_valid, current_best_valid)

    print(f"Pruning candidate interactions to {int(interaction_cfg['max_pruned_features'])} features.")
    selected_defs, pruning_log = prune_correlated_interactions(
        x_base_train,
        definitions,
        candidate_scores,
        eps=float(interaction_cfg.get("eps", 1e-6)),
        corr_threshold=float(interaction_cfg["corr_prune_threshold"]),
        max_features=int(interaction_cfg["max_pruned_features"]),
    )
    pruning_log.to_csv(run_dir / "interaction_correlation_pruning.csv", index=False)
    definitions_to_frame(selected_defs).to_csv(run_dir / "selected_interaction_definitions_max.csv", index=False)

    print("Transforming max interaction matrix.")
    x_int_train = transform_interactions(
        x_base_train,
        selected_defs,
        eps=float(interaction_cfg.get("eps", 1e-6)),
        block_size=int(interaction_cfg.get("transform_block_size", 128)),
    )
    x_int_valid = transform_interactions(
        x_base_valid,
        selected_defs,
        eps=float(interaction_cfg.get("eps", 1e-6)),
        block_size=int(interaction_cfg.get("transform_block_size", 128)),
    )
    x_int_full = transform_interactions(
        x_base_full,
        selected_defs,
        eps=float(interaction_cfg.get("eps", 1e-6)),
        block_size=int(interaction_cfg.get("transform_block_size", 128)),
    )
    x_int_test = transform_interactions(
        x_base_test,
        selected_defs,
        eps=float(interaction_cfg.get("eps", 1e-6)),
        block_size=int(interaction_cfg.get("transform_block_size", 128)),
    )
    del x_base_train, x_base_valid, x_base_full, x_base_test
    gc.collect()

    signal_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    signal_cache: dict[tuple[int, float], tuple[np.ndarray, np.ndarray]] = {}
    for feature_count in [int(value) for value in interaction_cfg["feature_counts"]]:
        if feature_count > len(selected_defs):
            raise ValueError(f"feature_count={feature_count} exceeds selected definitions={len(selected_defs)}")
        write_selected_features(
            run_dir / f"selected_interaction_features_{feature_count}.txt",
            [definition.name for definition in selected_defs[:feature_count]],
        )
        for alpha in [float(value) for value in ridge_cfg["alphas"]]:
            print(f"Training Ridge interaction signal: count={feature_count}, alpha={alpha:g}")
            valid_pred = fit_ridge(alpha, x_int_train[:, :feature_count], y_train, x_int_valid[:, :feature_count])
            test_pred = fit_ridge(alpha, x_int_full[:, :feature_count], y_full, x_int_test[:, :feature_count])
            signal_cache[(feature_count, alpha)] = (valid_pred, test_pred)
            save_valid_prediction(pred_dir / f"valid_interactions_{feature_count}_alpha{safe_name(alpha)}.csv", y_valid, valid_pred)
            signal_submission = make_submission(sample_submission, test_df, test_pred, prediction_col, id_col)
            signal_submission.to_csv(signal_submission_dir / f"submission_interactions_{feature_count}_alpha{safe_name(alpha)}.csv", index=False)
            signal_pearson = pearson_corr(y_valid, valid_pred)
            signal_rows.append(
                {
                    "feature_count": feature_count,
                    "alpha": alpha,
                    "pearson": signal_pearson,
                    "rmse": rmse(y_valid, valid_pred),
                    "holdout_delta_beta4": signal_pearson - pearson_corr(y_valid, beta4_valid),
                    "holdout_delta_current_best": signal_pearson - current_best_holdout,
                    "valid_corr_current_best": pearson_corr(current_best_valid, valid_pred),
                    "submission_corr_current_best": pearson_corr(current_best_test, test_pred),
                }
            )
            for signal_weight in [float(value) for value in blend_cfg["signal_weights"]]:
                add_candidate(
                    candidate_rows,
                    feature_count,
                    alpha,
                    signal_weight,
                    beta4_valid,
                    beta4_test,
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

    signal_metrics = pd.DataFrame(signal_rows).sort_values(["pearson", "submission_corr_current_best"], ascending=[False, True])
    signal_metrics.to_csv(run_dir / "signal_metrics.csv", index=False)
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

    selected = select_candidates(
        candidates,
        int(blend_cfg.get("max_selected_submissions", 3)),
        baseline_feature_count=120,
        baseline_alpha=200000.0,
        baseline_weight=float(current_cfg["interaction_weight"]),
    )
    selected.to_csv(run_dir / "selected_candidates.csv", index=False)
    if not selected.empty:
        shutil.copyfile(ROOT / str(selected.iloc[0]["path"]), submission_dir / "submission_best.csv")

    write_json(
        run_dir / "final_summary.json",
        {
            "beta4_base": beta4_cfg,
            "current_best": current_cfg,
            "beta4_holdout_pearson": pearson_corr(y_valid, beta4_valid),
            "current_best_holdout_pearson": current_best_holdout,
            "candidate_pool_count": len(candidate_scores),
            "max_pruned_features": len(selected_defs),
            "feature_counts": interaction_cfg["feature_counts"],
            "alphas": ridge_cfg["alphas"],
            "best_signal": signal_metrics.iloc[0].to_dict(),
            "selected_candidates": selected.to_dict(orient="records"),
        },
    )

    print("\n=== Signal Metrics ===")
    print(signal_metrics.head(20).to_string(index=False))
    print("\n=== Selected Candidates ===")
    if selected.empty:
        print("No eligible candidates selected for submission.")
    else:
        print(selected.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
