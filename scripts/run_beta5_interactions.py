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
    build_ordered_feature_pool,
    candidate_pool_from_scores,
    definitions_to_frame,
    generate_pairwise_interactions,
    prune_correlated_interactions,
    score_interactions,
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
    parser = argparse.ArgumentParser(description="Run Beta5-A interaction feature experiments.")
    parser.add_argument("--config", default="configs/01_main_beta5_interactions.yaml", help="Path to YAML config.")
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


def summarize_delta(pred: np.ndarray, reference: np.ndarray, prefix: str) -> dict[str, float]:
    delta = pred - reference
    return {
        f"{prefix}_corr_current_best": pearson_corr(reference, pred),
        f"{prefix}_delta_std_current_best": float(np.std(delta)),
        f"{prefix}_delta_mae_current_best": float(np.mean(np.abs(delta))),
    }


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


def read_feature_source(root: Path, source: dict[str, Any]) -> list[str]:
    features = read_lines(root / source["path"])
    top_n = source.get("top_n")
    if top_n is not None:
        features = features[: int(top_n)]
    return features


def add_blend_candidate(
    rows: list[dict[str, Any]],
    model_name: str,
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
    candidate = f"blend_current_w{safe_name(current_weight)}_{model_name}_w{safe_name(signal_weight)}"
    submission = make_submission(sample_submission, test_df, blend_test, prediction_col, id_col)
    path = submission_dir / f"submission_{candidate}.csv"
    submission.to_csv(path, index=False)
    holdout = pearson_corr(y_valid, blend_valid)
    rows.append(
        {
            "candidate": candidate,
            "model": model_name,
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

    xgb_pool = eligible[eligible["model"] == "xgb_core_plus_interactions"].sort_values(
        ["holdout_pearson", "submission_corr_current_best"],
        ascending=[False, True],
    )
    if not xgb_pool.empty:
        candidate = str(xgb_pool.iloc[0]["candidate"])
        selected_names.append(candidate)
        reasons[candidate] = "xgb_holdout_best"

    backup_pool = eligible[~eligible["candidate"].isin(selected_names)].sort_values(
        ["submission_corr_current_best", "holdout_pearson"],
        ascending=[True, False],
    )
    if len(selected_names) < int(max_selected) and not backup_pool.empty:
        candidate = str(backup_pool.iloc[0]["candidate"])
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

    source_features = [read_feature_source(ROOT, source) for source in pool_cfg["sources"]]
    base_features = build_ordered_feature_pool(source_features, int(pool_cfg["max_features"]))
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

    print("Reconstructing current-best validation and residual predictions.")
    beta3_components = current_cfg["beta3_components"]
    beta3_fit = weighted_ridge_prediction(train_part, y_train, train_part, beta3_components, prep_cfg, ROOT)
    beta3_valid = weighted_ridge_prediction(train_part, y_train, valid_part, beta3_components, prep_cfg, ROOT)

    shap_features = read_lines(ROOT / current_cfg["shap_stable_features_path"])
    shap_preprocessor = TabularPreprocessor(shap_features).fit(
        train_part,
        missing_fill=str(prep_cfg.get("missing_fill", "median")),
        standardize=bool(prep_cfg.get("standardize", True)),
    )
    x_shap_train = shap_preprocessor.transform(train_part)
    x_shap_valid = shap_preprocessor.transform(valid_part)
    shap_model, shap_valid_recomputed, _, _ = train_xgb_model(x_shap_train, y_train, x_shap_valid, y_valid, xgb_cfg, 2026)
    shap_fit = shap_model.predict(x_shap_train)

    loaded_shap_valid = load_prediction_csv(
        ROOT / current_cfg["shap_stable_valid_prediction_path"],
        prediction_col="prediction",
        expected_len=len(y_valid),
    )
    shap_valid = loaded_shap_valid if loaded_shap_valid is not None else shap_valid_recomputed
    current_best_fit = float(current_cfg["base_weight"]) * beta3_fit + float(current_cfg["signal_weight"]) * shap_fit
    current_best_valid = float(current_cfg["base_weight"]) * beta3_valid + float(current_cfg["signal_weight"]) * shap_valid
    current_best_test = load_prediction_csv(
        ROOT / current_cfg["test_submission_path"],
        prediction_col=prediction_col,
        expected_len=len(test_df),
    )
    if current_best_test is None:
        raise FileNotFoundError(f"Could not load current-best test predictions from {current_cfg['test_submission_path']}.")
    current_best_holdout = pearson_corr(y_valid, current_best_valid)
    residual = y_train - current_best_fit
    del x_shap_train, x_shap_valid, shap_model, beta3_fit, beta3_valid, shap_fit, shap_valid_recomputed
    gc.collect()

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
    print(f"Scoring {len(definitions)} interaction candidates from {len(base_features)} base features.")
    scores, fitted_definitions = score_interactions(
        x_base_train,
        definitions,
        label=y_train,
        residual=residual,
        eps=float(interaction_cfg.get("eps", 1e-6)),
        clip_quantiles=(
            float(interaction_cfg["clip_quantiles"][0]),
            float(interaction_cfg["clip_quantiles"][1]),
        ),
        block_size=int(interaction_cfg.get("score_block_size", 64)),
        spearman_sample_rows=int(interaction_cfg.get("spearman_sample_rows", 120000)),
    )
    scores.to_csv(run_dir / "interaction_candidate_scores.csv", index=False)
    definition_lookup = {definition.name: definition for definition in fitted_definitions}
    candidate_scores = candidate_pool_from_scores(scores, int(interaction_cfg["top_k_each_metric"]))
    candidate_scores.to_csv(run_dir / "interaction_candidate_pool.csv", index=False)
    candidate_definitions = [definition_lookup[str(name)] for name in candidate_scores["feature"]]

    selected_definitions, pruning_log = prune_correlated_interactions(
        x_base_train,
        candidate_definitions,
        candidate_scores,
        eps=float(interaction_cfg.get("eps", 1e-6)),
        corr_threshold=float(interaction_cfg["corr_prune_threshold"]),
        max_features=int(interaction_cfg["max_selected_features"]),
    )
    pruning_log.to_csv(run_dir / "interaction_correlation_pruning.csv", index=False)
    selected_frame = definitions_to_frame(selected_definitions)
    selected_frame.to_csv(run_dir / "selected_interaction_definitions.csv", index=False)
    write_selected_features(run_dir / "selected_interaction_features.txt", selected_frame["feature"].astype(str).tolist())

    print(f"Selected {len(selected_definitions)} interaction features after correlation pruning.")
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

    signals: dict[str, tuple[np.ndarray, np.ndarray, dict[str, Any]]] = {}
    metric_rows: list[dict[str, Any]] = []

    def record_signal(model_name: str, valid_pred: np.ndarray, test_pred: np.ndarray, extra: dict[str, Any]) -> None:
        save_valid_prediction(pred_dir / f"valid_{model_name}.csv", y_valid, valid_pred)
        signal_submission = make_submission(sample_submission, test_df, test_pred, prediction_col, id_col)
        signal_submission.to_csv(signal_submission_dir / f"submission_{model_name}.csv", index=False)
        signal_pearson = pearson_corr(y_valid, valid_pred)
        metric_rows.append(
            {
                "model": model_name,
                "base_feature_count": len(base_features),
                "interaction_feature_count": len(selected_definitions),
                "total_feature_count": extra.get("total_feature_count", len(selected_definitions)),
                "pearson": signal_pearson,
                "rmse": rmse(y_valid, valid_pred),
                "holdout_delta_current_best": signal_pearson - current_best_holdout,
                "valid_corr_current_best": pearson_corr(current_best_valid, valid_pred),
                "submission_corr_current_best": pearson_corr(current_best_test, test_pred),
                **extra,
            }
        )
        signals[model_name] = (valid_pred, test_pred, extra)

    print("Training Ridge on interactions only.")
    ridge_best, ridge_valid, ridge_grid = train_ridge_search(
        x_int_train,
        y_train,
        x_int_valid,
        y_valid,
        [float(alpha) for alpha in ridge_cfg["alphas"]],
    )
    ridge_grid.to_csv(run_dir / "ridge_alpha_search_interactions_only.csv", index=False)
    ridge_test = fit_ridge(float(ridge_best["alpha"]), x_int_full, y_full, x_int_test)
    record_signal(
        "ridge_interactions_only",
        ridge_valid,
        ridge_test,
        {"alpha": float(ridge_best["alpha"]), "total_feature_count": len(selected_definitions)},
    )

    print("Training Ridge on core plus interactions.")
    x_core_int_train = np.hstack([x_base_train, x_int_train]).astype(np.float32, copy=False)
    x_core_int_valid = np.hstack([x_base_valid, x_int_valid]).astype(np.float32, copy=False)
    x_core_int_full = np.hstack([x_base_full, x_int_full]).astype(np.float32, copy=False)
    x_core_int_test = np.hstack([x_base_test, x_int_test]).astype(np.float32, copy=False)
    ridge_core_best, ridge_core_valid, ridge_core_grid = train_ridge_search(
        x_core_int_train,
        y_train,
        x_core_int_valid,
        y_valid,
        [float(alpha) for alpha in ridge_cfg["alphas"]],
    )
    ridge_core_grid.to_csv(run_dir / "ridge_alpha_search_core_plus_interactions.csv", index=False)
    ridge_core_test = fit_ridge(float(ridge_core_best["alpha"]), x_core_int_full, y_full, x_core_int_test)
    record_signal(
        "ridge_core_plus_interactions",
        ridge_core_valid,
        ridge_core_test,
        {
            "alpha": float(ridge_core_best["alpha"]),
            "total_feature_count": x_core_int_train.shape[1],
        },
    )

    print("Training XGB on core plus interactions.")
    xgb_model, xgb_valid, xgb_final_params, xgb_row = train_xgb_model(
        x_core_int_train,
        y_train,
        x_core_int_valid,
        y_valid,
        xgb_cfg,
        random_state=2027,
    )
    xgb_test = fit_xgb_predict(xgb_final_params, x_core_int_full, y_full, x_core_int_test)
    record_signal(
        "xgb_core_plus_interactions",
        xgb_valid,
        xgb_test,
        {
            "best_iteration": xgb_row["best_iteration"],
            "params": json.dumps(xgb_final_params, sort_keys=True),
            "total_feature_count": x_core_int_train.shape[1],
        },
    )
    del xgb_model

    metrics = pd.DataFrame(metric_rows).sort_values("pearson", ascending=False)
    metrics.to_csv(run_dir / "signal_metrics.csv", index=False)

    candidate_rows: list[dict[str, Any]] = []
    for model_name, (valid_pred, test_pred, _) in signals.items():
        for signal_weight in [float(value) for value in blend_cfg["signal_weights"]]:
            add_blend_candidate(
                candidate_rows,
                model_name,
                signal_weight,
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
            "current_best_holdout_pearson": current_best_holdout,
            "base_feature_count": len(base_features),
            "candidate_interaction_count": len(definitions),
            "candidate_pool_count": len(candidate_scores),
            "selected_interaction_count": len(selected_definitions),
            "best_signal": metrics.iloc[0].to_dict(),
            "selected_candidates": selected.to_dict(orient="records"),
        },
    )

    print("\n=== Signal Metrics ===")
    print(metrics.to_string(index=False))
    print("\n=== Selected Candidates ===")
    if selected.empty:
        print("No eligible candidates selected for submission.")
    else:
        print(selected.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
