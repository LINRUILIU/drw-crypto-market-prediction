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
    parser = argparse.ArgumentParser(description="Train intermediate Pearson top-k Ridge submissions.")
    parser.add_argument("--config", default="configs/01_main_beta2_intermediate_ridge.yaml", help="Path to YAML config.")
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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


def save_valid_prediction(path: Path, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "row_index": np.arange(len(y_true), dtype=np.int64),
            "y_true": y_true,
            "prediction": y_pred,
        }
    ).to_csv(path, index=False)


def fit_ridge(alpha: float, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    return NumpyRidgeRegressor(alpha=alpha).fit(x_train, y_train).predict(x_test)


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
    signal_dir = ensure_dir(submission_dir / "signals")

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
    prediction_col = choose_prediction_column(sample_submission, output_cfg.get("prediction_col"), target_col)

    ranking = pd.read_csv(ROOT / feature_cfg["ranking_path"])
    ranked_features = [feature for feature in ranking["feature"].astype(str).tolist() if feature in feature_cols]
    if not ranked_features:
        raise ValueError("No ranked features matched inferred feature columns.")

    train_part, valid_part = split_time_ordered(train_df, float(data_cfg["validation_fraction"]))
    y_train = train_part[target_col].to_numpy(dtype=np.float64)
    y_valid = valid_part[target_col].to_numpy(dtype=np.float64)
    y_full = train_df[target_col].to_numpy(dtype=np.float64)
    alphas = [float(alpha) for alpha in model_cfg["ridge"].get("alphas", [1000.0])]

    rows: list[dict[str, Any]] = []
    output_paths: dict[str, Path] = {}
    for top_k in [int(value) for value in feature_cfg["top_k"]]:
        scheme = f"top{top_k}"
        selected = ranked_features[: min(top_k, len(ranked_features))]
        write_lines(run_dir / f"selected_features_{scheme}.txt", selected)
        print(f"\n=== Running {scheme} Ridge ({len(selected)} features) ===")

        preprocessor = TabularPreprocessor(selected).fit(
            train_part,
            missing_fill=str(prep_cfg.get("missing_fill", "median")),
            standardize=bool(prep_cfg.get("standardize", True)),
        )
        x_train = preprocessor.transform(train_part)
        x_valid = preprocessor.transform(valid_part)
        best, valid_pred, alpha_grid = train_ridge_search(x_train, y_train, x_valid, y_valid, alphas)
        alpha_grid.insert(0, "scheme", scheme)
        alpha_grid.to_csv(run_dir / f"ridge_alpha_search_{scheme}.csv", index=False)
        save_valid_prediction(pred_dir / f"valid_{scheme}_ridge.csv", y_valid, valid_pred)

        final_preprocessor = TabularPreprocessor(selected).fit(
            train_df,
            missing_fill=str(prep_cfg.get("missing_fill", "median")),
            standardize=bool(prep_cfg.get("standardize", True)),
        )
        x_full = final_preprocessor.transform(train_df)
        x_test = final_preprocessor.transform(test_df)
        test_pred = fit_ridge(float(best["alpha"]), x_full, y_full, x_test)
        submission = make_submission(sample_submission, test_df, test_pred, prediction_col, id_col)
        signal_path = signal_dir / f"submission_{scheme}_ridge.csv"
        candidate_path = submission_dir / f"submission_{scheme}_ridge.csv"
        submission.to_csv(signal_path, index=False)
        submission.to_csv(candidate_path, index=False)
        output_paths[scheme] = candidate_path

        rows.append(
            {
                "scheme": scheme,
                "feature_count": len(selected),
                "alpha": best["alpha"],
                "holdout_pearson": best["pearson"],
                "holdout_rmse": best["rmse"],
                "valid_prediction_std": float(np.std(valid_pred)),
                "submission_prediction_std": float(np.std(test_pred)),
                "path": str(candidate_path.relative_to(ROOT)),
            }
        )

    metrics = pd.DataFrame(rows).sort_values("holdout_pearson", ascending=False)
    metrics.to_csv(run_dir / "intermediate_ridge_metrics.csv", index=False)
    best_row = metrics.iloc[0].to_dict()
    best_scheme = str(best_row["scheme"])
    shutil.copyfile(output_paths[best_scheme], submission_dir / "submission_best.csv")
    write_json(
        run_dir / "final_summary.json",
        {
            "best_validation_row": best_row,
            "default_submission": best_scheme,
            "default_submission_path": str((submission_dir / "submission_best.csv").relative_to(ROOT)),
            "top_k": feature_cfg["top_k"],
            "alphas": alphas,
        },
    )

    print("\n=== Intermediate Ridge metrics ===")
    print(metrics.to_string(index=False))
    print(f"\nDefault submission: {best_scheme} -> {submission_dir / 'submission_best.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
