from __future__ import annotations

import argparse
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
from drw_crypto.pipeline import make_submission  # noqa: E402
from drw_crypto.preprocessing import (  # noqa: E402
    TabularPreprocessor,
    infer_feature_columns,
    infer_id_column,
    infer_target_column,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train top50/top100 Ridge alpha variants and blend candidates.")
    parser.add_argument("--config", default="configs/01_main_beta2_alpha_variants.yaml", help="Path to YAML config.")
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def split_time_ordered(df: pd.DataFrame, validation_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_at = int(len(df) * (1.0 - validation_fraction))
    if split_at <= 0 or split_at >= len(df):
        raise ValueError("Validation split produced an empty train or validation set.")
    return df.iloc[:split_at], df.iloc[split_at:]


def choose_prediction_column(sample_submission: pd.DataFrame | None, configured: str | None, target_col: str) -> str:
    if configured:
        return configured
    if sample_submission is None:
        return "prediction"
    if target_col in sample_submission.columns:
        return target_col
    if "prediction" in sample_submission.columns:
        return "prediction"
    return str(sample_submission.columns[-1])


def read_submission_prediction(path: Path, prediction_col: str) -> np.ndarray:
    frame = pd.read_csv(path)
    if prediction_col not in frame.columns:
        raise ValueError(f"Prediction column {prediction_col!r} is not in {path}")
    return frame[prediction_col].to_numpy(dtype=np.float64)


def fit_ridge(alpha: float, x_train: np.ndarray, y_train: np.ndarray, x_pred: np.ndarray) -> np.ndarray:
    start = time.perf_counter()
    pred = NumpyRidgeRegressor(alpha=alpha).fit(x_train, y_train).predict(x_pred)
    elapsed = time.perf_counter() - start
    return pred, elapsed


def safe_name(value: float) -> str:
    text = f"{value:g}".replace(".", "p").replace("-", "m")
    return text


def configured_alphas(component_cfg: dict[str, Any], ridge_cfg: dict[str, Any], component_name: str) -> list[float]:
    values = component_cfg.get("alphas", ridge_cfg.get("alphas"))
    if not values:
        raise ValueError(f"No Ridge alphas configured for {component_name}.")
    return [float(value) for value in values]


def summarize_delta(pred: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    delta = pred - reference
    return {
        "corr_current_best": pearson_corr(reference, pred),
        "delta_std_current_best": float(np.std(delta)),
        "delta_mae_current_best": float(np.mean(np.abs(delta))),
    }


def blend_name(top50_alpha: float, top100_alpha: float, top50_weight: float) -> str:
    top100_weight = 1.0 - top50_weight
    return (
        f"blend_t50a{safe_name(top50_alpha)}_t100a{safe_name(top100_alpha)}"
        f"_w50_{safe_name(top50_weight)}_w100_{safe_name(top100_weight)}"
    )


def main() -> int:
    args = parse_args()
    config_path = ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)
    config = load_config(config_path)

    data_cfg = config["data"]
    prep_cfg = config["preprocessing"]
    comp_cfg = config["components"]
    ridge_cfg = config["ridge"]
    blend_cfg = config["blend"]
    selection_cfg = config["selection"]
    output_cfg = config["output"]

    run_dir = ensure_dir(ROOT / output_cfg["run_dir"])
    pred_dir = ensure_dir(run_dir / "valid_predictions")
    submission_dir = ensure_dir(ROOT / output_cfg["submission_dir"])
    signal_dir = ensure_dir(submission_dir / "signals")
    prediction_col = str(output_cfg.get("prediction_col") or "prediction")

    raw_dir = ROOT / data_cfg["raw_dir"]
    train_path = find_first_existing(raw_dir, data_cfg.get("train_file"), TRAIN_CANDIDATES)
    test_path = find_first_existing(raw_dir, data_cfg.get("test_file"), TEST_CANDIDATES)
    sample_path = find_first_existing(raw_dir, data_cfg.get("sample_submission_file"), SAMPLE_SUBMISSION_CANDIDATES)
    if train_path is None or test_path is None:
        raise FileNotFoundError("Missing train/test files under data/raw.")

    train_df = read_table(train_path)
    test_df = read_table(test_path)
    sample_submission = read_table(sample_path) if sample_path is not None else None
    target_col = infer_target_column(train_df, test_df, data_cfg.get("target_col"))
    id_col = infer_id_column(train_df, test_df, sample_submission, target_col, data_cfg.get("id_col"))
    feature_cols = infer_feature_columns(
        train_df,
        target_col=target_col,
        id_col=id_col,
        drop_columns=data_cfg.get("drop_columns") or [],
        numeric_only=bool(prep_cfg.get("numeric_only", True)),
    )
    prediction_col = choose_prediction_column(sample_submission, prediction_col, target_col)

    train_part, valid_part = split_time_ordered(train_df, float(data_cfg["validation_fraction"]))
    y_train = train_part[target_col].to_numpy(dtype=np.float64)
    y_valid = valid_part[target_col].to_numpy(dtype=np.float64)
    y_full = train_df[target_col].to_numpy(dtype=np.float64)
    component_alphas = {
        component_name: configured_alphas(component_cfg, ridge_cfg, component_name)
        for component_name, component_cfg in comp_cfg.items()
    }

    valid_preds: dict[tuple[str, float], np.ndarray] = {}
    test_preds: dict[tuple[str, float], np.ndarray] = {}
    component_rows: list[dict[str, Any]] = []

    for component_name, component_cfg in comp_cfg.items():
        selected = read_lines(ROOT / component_cfg["selected_features_path"])
        missing = sorted(set(selected) - set(feature_cols))
        if missing:
            raise ValueError(f"{component_name} has selected features not present in training data: {missing[:5]}")
        write_lines(run_dir / f"selected_features_{component_name}.txt", selected)
        print(f"\n=== Training {component_name} alpha variants ({len(selected)} features) ===")

        split_preprocessor = TabularPreprocessor(selected).fit(
            train_part,
            missing_fill=str(prep_cfg.get("missing_fill", "median")),
            standardize=bool(prep_cfg.get("standardize", True)),
        )
        x_train = split_preprocessor.transform(train_part)
        x_valid = split_preprocessor.transform(valid_part)

        full_preprocessor = TabularPreprocessor(selected).fit(
            train_df,
            missing_fill=str(prep_cfg.get("missing_fill", "median")),
            standardize=bool(prep_cfg.get("standardize", True)),
        )
        x_full = full_preprocessor.transform(train_df)
        x_test = full_preprocessor.transform(test_df)

        for alpha in component_alphas[component_name]:
            valid_pred, valid_seconds = fit_ridge(alpha, x_train, y_train, x_valid)
            test_pred, final_seconds = fit_ridge(alpha, x_full, y_full, x_test)
            key = (component_name, alpha)
            valid_preds[key] = valid_pred
            test_preds[key] = test_pred

            alpha_name = safe_name(alpha)
            valid_path = pred_dir / f"valid_{component_name}_alpha{alpha_name}.csv"
            pd.DataFrame(
                {
                    "row_index": np.arange(len(y_valid), dtype=np.int64),
                    "y_true": y_valid,
                    "prediction": valid_pred,
                }
            ).to_csv(valid_path, index=False)

            submission = make_submission(sample_submission, test_df, test_pred, prediction_col, id_col)
            submission_path = signal_dir / f"submission_{component_name}_alpha{alpha_name}.csv"
            submission.to_csv(submission_path, index=False)

            component_rows.append(
                {
                    "component": component_name,
                    "alpha": alpha,
                    "feature_count": len(selected),
                    "holdout_pearson": pearson_corr(y_valid, valid_pred),
                    "holdout_rmse": rmse(y_valid, valid_pred),
                    "valid_prediction_std": float(np.std(valid_pred)),
                    "submission_prediction_std": float(np.std(test_pred)),
                    "valid_train_seconds": valid_seconds,
                    "final_train_seconds": final_seconds,
                    "valid_prediction_path": str(valid_path.relative_to(ROOT)),
                    "submission_path": str(submission_path.relative_to(ROOT)),
                }
            )

    component_metrics = pd.DataFrame(component_rows).sort_values(["component", "holdout_pearson"], ascending=[True, False])
    component_metrics.to_csv(run_dir / "component_metrics.csv", index=False)

    current_cfg = blend_cfg["current_best"]
    current_top50_weight = float(current_cfg["top50_weight"])
    current_top100_weight = 1.0 - current_top50_weight
    current_top50_alpha = float(current_cfg["top50_alpha"])
    current_top100_alpha = float(current_cfg["top100_alpha"])
    current_best_valid = (
        current_top50_weight * valid_preds[("top50", current_top50_alpha)]
        + current_top100_weight * valid_preds[("top100", current_top100_alpha)]
    )
    current_best_test = read_submission_prediction(ROOT / current_cfg["submission_path"], prediction_col)
    current_best_holdout = pearson_corr(y_valid, current_best_valid)
    current_best_test_std = float(np.std(current_best_test))

    candidate_rows: list[dict[str, Any]] = []
    candidate_predictions: dict[str, np.ndarray] = {}
    candidate_valid_predictions: dict[str, np.ndarray] = {}

    top50_alphas = component_alphas["top50"]
    top100_alphas = component_alphas["top100"]
    for top50_alpha in top50_alphas:
        for top100_alpha in top100_alphas:
            for top50_weight in [float(value) for value in blend_cfg["top50_weights"]]:
                top100_weight = 1.0 - top50_weight
                name = blend_name(top50_alpha, top100_alpha, top50_weight)
                valid_pred = (
                    top50_weight * valid_preds[("top50", top50_alpha)]
                    + top100_weight * valid_preds[("top100", top100_alpha)]
                )
                test_pred = (
                    top50_weight * test_preds[("top50", top50_alpha)]
                    + top100_weight * test_preds[("top100", top100_alpha)]
                )
                candidate_predictions[name] = test_pred
                candidate_valid_predictions[name] = valid_pred
                valid_delta = summarize_delta(valid_pred, current_best_valid)
                test_delta = summarize_delta(test_pred, current_best_test)
                candidate_rows.append(
                    {
                        "name": name,
                        "kind": "grid_blend",
                        "top50_alpha": top50_alpha,
                        "top100_alpha": top100_alpha,
                        "top50_weight": top50_weight,
                        "top100_weight": top100_weight,
                        "holdout_pearson": pearson_corr(y_valid, valid_pred),
                        "holdout_rmse": rmse(y_valid, valid_pred),
                        "valid_prediction_std": float(np.std(valid_pred)),
                        "submission_prediction_std": float(np.std(test_pred)),
                        "valid_corr_current_best": valid_delta["corr_current_best"],
                        "valid_delta_std_current_best": valid_delta["delta_std_current_best"],
                        "valid_delta_mae_current_best": valid_delta["delta_mae_current_best"],
                        "submission_corr_current_best": test_delta["corr_current_best"],
                        "submission_delta_std_current_best": test_delta["delta_std_current_best"],
                        "submission_delta_mae_current_best": test_delta["delta_mae_current_best"],
                        "is_current_best_equivalent": bool(
                            np.isclose(top50_alpha, current_top50_alpha)
                            and np.isclose(top100_alpha, current_top100_alpha)
                            and np.isclose(top50_weight, current_top50_weight)
                        ),
                    }
                )

    ensemble_cfg = blend_cfg["alpha_ensemble"]
    ensemble_top50_alphas = [
        float(value)
        for value in ensemble_cfg.get("top50_alphas", ensemble_cfg.get("alphas", []))
    ]
    ensemble_top100_alphas = [
        float(value)
        for value in ensemble_cfg.get("top100_alphas", ensemble_cfg.get("alphas", []))
    ]
    ensemble_top50_weight = float(ensemble_cfg["top50_weight"])
    ensemble_top100_weight = 1.0 - ensemble_top50_weight
    top50_valid_ensemble = np.mean([valid_preds[("top50", alpha)] for alpha in ensemble_top50_alphas], axis=0)
    top100_valid_ensemble = np.mean([valid_preds[("top100", alpha)] for alpha in ensemble_top100_alphas], axis=0)
    top50_test_ensemble = np.mean([test_preds[("top50", alpha)] for alpha in ensemble_top50_alphas], axis=0)
    top100_test_ensemble = np.mean([test_preds[("top100", alpha)] for alpha in ensemble_top100_alphas], axis=0)
    ensemble_name = f"alpha_ensemble_top50w{safe_name(ensemble_top50_weight)}_top100w{safe_name(ensemble_top100_weight)}"
    ensemble_valid_pred = ensemble_top50_weight * top50_valid_ensemble + ensemble_top100_weight * top100_valid_ensemble
    ensemble_test_pred = ensemble_top50_weight * top50_test_ensemble + ensemble_top100_weight * top100_test_ensemble
    candidate_predictions[ensemble_name] = ensemble_test_pred
    candidate_valid_predictions[ensemble_name] = ensemble_valid_pred
    valid_delta = summarize_delta(ensemble_valid_pred, current_best_valid)
    test_delta = summarize_delta(ensemble_test_pred, current_best_test)
    candidate_rows.append(
        {
            "name": ensemble_name,
            "kind": "alpha_ensemble",
            "top50_alpha": json.dumps(ensemble_top50_alphas),
            "top100_alpha": json.dumps(ensemble_top100_alphas),
            "top50_weight": ensemble_top50_weight,
            "top100_weight": ensemble_top100_weight,
            "holdout_pearson": pearson_corr(y_valid, ensemble_valid_pred),
            "holdout_rmse": rmse(y_valid, ensemble_valid_pred),
            "valid_prediction_std": float(np.std(ensemble_valid_pred)),
            "submission_prediction_std": float(np.std(ensemble_test_pred)),
            "valid_corr_current_best": valid_delta["corr_current_best"],
            "valid_delta_std_current_best": valid_delta["delta_std_current_best"],
            "valid_delta_mae_current_best": valid_delta["delta_mae_current_best"],
            "submission_corr_current_best": test_delta["corr_current_best"],
            "submission_delta_std_current_best": test_delta["delta_std_current_best"],
            "submission_delta_mae_current_best": test_delta["delta_mae_current_best"],
            "is_current_best_equivalent": False,
        }
    )

    candidates = pd.DataFrame(candidate_rows)
    candidates["std_ratio_current_best"] = candidates["submission_prediction_std"] / current_best_test_std
    candidates_sorted = candidates.sort_values("holdout_pearson", ascending=False).reset_index(drop=True)
    candidates_sorted.to_csv(run_dir / "candidate_metrics.csv", index=False)

    eligible = candidates_sorted[~candidates_sorted["is_current_best_equivalent"]].copy()
    stable = eligible[
        (eligible["valid_corr_current_best"] >= float(selection_cfg["stable_corr_min"]))
        & (eligible["std_ratio_current_best"] <= float(selection_cfg["stable_std_ratio_max"]))
    ]
    selected_names: list[str] = []
    selected_reasons: dict[str, str] = {}

    first_pool = stable if not stable.empty else eligible
    first = first_pool.sort_values("holdout_pearson", ascending=False).iloc[0]
    selected_names.append(str(first["name"]))
    selected_reasons[str(first["name"])] = "stable_holdout_best"

    best_holdout = float(eligible["holdout_pearson"].max())
    low_corr_pool = eligible[
        (eligible["holdout_pearson"] >= best_holdout - float(selection_cfg["low_corr_holdout_tolerance"]))
        & (~eligible["name"].isin(selected_names))
    ]
    if low_corr_pool.empty:
        low_corr_pool = eligible[~eligible["name"].isin(selected_names)]
    second = low_corr_pool.sort_values(["valid_corr_current_best", "holdout_pearson"], ascending=[True, False]).iloc[0]
    selected_names.append(str(second["name"]))
    selected_reasons[str(second["name"])] = "lower_corr_holdout_near_best"

    if ensemble_name not in selected_names:
        selected_names.append(ensemble_name)
        selected_reasons[ensemble_name] = "alpha_ensemble"

    selected_names = selected_names[:3]
    selected_rows = candidates[candidates["name"].isin(selected_names)].copy()
    selected_rows["selection_reason"] = selected_rows["name"].map(selected_reasons)
    selected_rows["path"] = ""

    for name in selected_names:
        test_pred = candidate_predictions[name]
        submission = make_submission(sample_submission, test_df, test_pred, prediction_col, id_col)
        output_path = submission_dir / f"submission_{name}.csv"
        submission.to_csv(output_path, index=False)
        selected_rows.loc[selected_rows["name"] == name, "path"] = str(output_path.relative_to(ROOT))

    selected_rows = selected_rows.set_index("name").loc[selected_names].reset_index()
    selected_rows.to_csv(run_dir / "selected_candidates.csv", index=False)
    default_name = selected_names[0]
    shutil.copyfile(submission_dir / f"submission_{default_name}.csv", submission_dir / "submission_best.csv")

    write_json(
        run_dir / "final_summary.json",
        {
            "current_best_private_score": current_cfg["private_score"],
            "current_best_holdout_pearson": current_best_holdout,
            "current_best_submission_path": current_cfg["submission_path"],
            "default_submission": default_name,
            "default_submission_path": str((submission_dir / "submission_best.csv").relative_to(ROOT)),
            "candidate_count": len(candidates),
            "selected_candidates": selected_names,
            "alphas": component_alphas,
            "top50_weights": blend_cfg["top50_weights"],
            "leakage_control": "features, preprocessing, and alpha variant validation use row-order 80/20 train split; final submissions fit on full train only after selection metrics are written.",
        },
    )

    print("\n=== Component metrics ===")
    print(component_metrics.to_string(index=False))
    print("\n=== Top candidate metrics ===")
    print(candidates_sorted.head(20).to_string(index=False))
    print("\n=== Selected candidates ===")
    print(selected_rows.to_string(index=False))
    print(f"\nDefault submission: {default_name} -> {submission_dir / 'submission_best.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
