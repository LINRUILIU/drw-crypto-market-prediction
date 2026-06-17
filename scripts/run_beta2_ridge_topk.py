from __future__ import annotations

import argparse
import json
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
from drw_crypto.models import NumpyRidgeRegressor  # noqa: E402
from drw_crypto.preprocessing import TabularPreprocessor, infer_id_column, infer_target_column  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and blend Ridge-only Pearson top-k submissions.")
    parser.add_argument("--config", default="configs/01_main_beta2_ridge_topk.yaml", help="Path to YAML config.")
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def best_alpha(path: Path) -> float:
    grid = pd.read_csv(path)
    return float(grid.sort_values("pearson", ascending=False).iloc[0]["alpha"])


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


def make_submission(
    sample_submission: pd.DataFrame | None,
    test_df: pd.DataFrame,
    pred: np.ndarray,
    prediction_col: str,
    id_col: str | None,
) -> pd.DataFrame:
    if sample_submission is not None:
        submission = sample_submission.copy()
        if prediction_col not in submission.columns:
            raise ValueError(f"Prediction column {prediction_col!r} is not in sample submission.")
        submission[prediction_col] = pred
        return submission
    if id_col and id_col in test_df.columns:
        return pd.DataFrame({id_col: test_df[id_col].to_numpy(), prediction_col: pred})
    return pd.DataFrame({"row_id": np.arange(len(pred), dtype=np.int64), prediction_col: pred})


def ensure_ridge_submission(
    signal_name: str,
    signal_cfg: dict[str, Any],
    config: dict[str, Any],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    id_col: str | None,
    sample_submission: pd.DataFrame | None,
    prediction_col: str,
) -> Path:
    output_path = ROOT / signal_cfg["output_submission_path"]
    if output_path.exists():
        return output_path

    selected_features = read_lines(ROOT / signal_cfg["selected_features_path"])
    alpha = best_alpha(ROOT / signal_cfg["ridge_alpha_search_path"])
    prep_cfg = config["preprocessing"]
    preprocessor = TabularPreprocessor(selected_features).fit(
        train_df,
        missing_fill=str(prep_cfg.get("missing_fill", "median")),
        standardize=bool(prep_cfg.get("standardize", True)),
    )
    x_train = preprocessor.transform(train_df)
    y_train = train_df[target_col].to_numpy(dtype=np.float64)
    x_test = preprocessor.transform(test_df)
    pred = NumpyRidgeRegressor(alpha=alpha).fit(x_train, y_train).predict(x_test)
    submission = make_submission(sample_submission, test_df, pred, prediction_col, id_col)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"Generated {signal_name}: {output_path.relative_to(ROOT)} alpha={alpha}")
    return output_path


def read_valid_prediction(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(ROOT / path)
    return frame["y_true"].to_numpy(dtype=np.float64), frame["prediction"].to_numpy(dtype=np.float64)


def read_submission(path: Path, prediction_col: str) -> tuple[pd.DataFrame, np.ndarray]:
    frame = pd.read_csv(path)
    return frame, frame[prediction_col].to_numpy(dtype=np.float64)


def assert_submission_ids_match(left: pd.DataFrame, right: pd.DataFrame, prediction_col: str) -> None:
    for col in left.columns:
        if col == prediction_col:
            continue
        if col in right.columns and not left[col].equals(right[col]):
            raise ValueError(f"Submission id column mismatch: {col}")


def normalize_weights(weights: dict[str, Any]) -> dict[str, float]:
    clean = {str(name): float(value) for name, value in weights.items() if float(value) != 0.0}
    total = sum(clean.values())
    if not np.isfinite(total) or total == 0.0:
        raise ValueError(f"Invalid zero-sum weights: {weights}")
    return {name: value / total for name, value in clean.items()}


def blend_valid_predictions(
    weights: dict[str, float],
    valid_predictions: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    y_ref: np.ndarray | None = None
    pred: np.ndarray | None = None
    for signal_name, weight in weights.items():
        y_true, signal_pred = valid_predictions[signal_name]
        if y_ref is None:
            y_ref = y_true
            pred = np.zeros_like(signal_pred, dtype=np.float64)
        elif not np.allclose(y_ref, y_true, rtol=0.0, atol=1e-10):
            raise ValueError(f"Validation target mismatch for {signal_name}")
        pred += weight * signal_pred
    if y_ref is None or pred is None:
        raise ValueError("No valid predictions were blended.")
    return y_ref, pred


def blend_submissions(
    weights: dict[str, float],
    submission_paths: dict[str, Path],
    prediction_col: str,
) -> pd.DataFrame:
    output: pd.DataFrame | None = None
    pred: np.ndarray | None = None
    for signal_name, weight in weights.items():
        frame, signal_pred = read_submission(submission_paths[signal_name], prediction_col)
        if output is None:
            output = frame.copy()
            pred = np.zeros_like(signal_pred, dtype=np.float64)
        else:
            assert_submission_ids_match(output, frame, prediction_col)
        pred += weight * signal_pred
    if output is None or pred is None:
        raise ValueError("No submissions were blended.")
    output[prediction_col] = pred
    return output


def main() -> int:
    args = parse_args()
    config_path = ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)
    config = load_config(config_path)
    output_cfg = config["output"]
    run_dir = ensure_dir(ROOT / output_cfg["run_dir"])
    submission_dir = ensure_dir(ROOT / output_cfg["submission_dir"])
    prediction_col = str(output_cfg.get("prediction_col") or "prediction")

    data_cfg = config["data"]
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
    prediction_col = choose_prediction_column(sample_submission, prediction_col, target_col)

    signal_paths: dict[str, Path] = {}
    valid_predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    signal_rows: list[dict[str, Any]] = []
    for signal_name, signal_cfg in config["ridge_signals"].items():
        signal_paths[signal_name] = ensure_ridge_submission(
            signal_name,
            signal_cfg,
            config,
            train_df,
            test_df,
            target_col,
            id_col,
            sample_submission,
            prediction_col,
        )
        y_true, pred = read_valid_prediction(signal_cfg["valid_prediction_path"])
        valid_predictions[signal_name] = (y_true, pred)
        signal_rows.append(
            {
                "signal": signal_name,
                "alpha": best_alpha(ROOT / signal_cfg["ridge_alpha_search_path"]),
                "feature_count": len(read_lines(ROOT / signal_cfg["selected_features_path"])),
                "holdout_pearson": pearson_corr(y_true, pred),
                "holdout_rmse": rmse(y_true, pred),
                "valid_prediction_std": float(np.std(pred)),
                "submission_path": str(signal_paths[signal_name].relative_to(ROOT)),
            }
        )

    pd.DataFrame(signal_rows).sort_values("holdout_pearson", ascending=False).to_csv(
        run_dir / "ridge_signal_metrics.csv",
        index=False,
    )

    signal_names = list(config["ridge_signals"].keys())
    corr = pd.DataFrame(index=signal_names, columns=signal_names, dtype=float)
    for left in signal_names:
        for right in signal_names:
            corr.loc[left, right] = pearson_corr(valid_predictions[left][1], valid_predictions[right][1])
    corr.to_csv(run_dir / "ridge_signal_correlation.csv")

    candidate_rows: list[dict[str, Any]] = []
    written: dict[str, Path] = {}
    for candidate in config["candidates"]["fixed"]:
        name = str(candidate["name"])
        weights = normalize_weights(candidate["weights"])
        y_true, valid_pred = blend_valid_predictions(weights, valid_predictions)
        submission = blend_submissions(weights, signal_paths, prediction_col)
        output_path = submission_dir / f"submission_{name}.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        submission.to_csv(output_path, index=False)
        written[name] = output_path
        candidate_rows.append(
            {
                "name": name,
                "weights": json.dumps(weights, sort_keys=True),
                "holdout_pearson": pearson_corr(y_true, valid_pred),
                "holdout_rmse": rmse(y_true, valid_pred),
                "valid_prediction_std": float(np.std(valid_pred)),
                "submission_prediction_std": float(np.std(submission[prediction_col].to_numpy(dtype=np.float64))),
                "path": str(output_path.relative_to(ROOT)),
            }
        )

    candidates = pd.DataFrame(candidate_rows).sort_values("holdout_pearson", ascending=False)
    candidates.to_csv(run_dir / "candidate_submissions.csv", index=False)

    default_name = str(config["candidates"]["default"])
    if default_name not in written:
        raise ValueError(f"Default candidate {default_name!r} was not written.")
    shutil.copyfile(written[default_name], submission_dir / "submission_best.csv")
    write_json(
        run_dir / "final_summary.json",
        {
            "default_submission": default_name,
            "default_submission_path": str((submission_dir / "submission_best.csv").relative_to(ROOT)),
            "candidate_count": len(candidates),
            "signal_count": len(signal_paths),
        },
    )

    print("\n=== Ridge signal metrics ===")
    print(pd.DataFrame(signal_rows).sort_values("holdout_pearson", ascending=False).to_string(index=False))
    print("\n=== Ridge candidate submissions ===")
    print(candidates[["name", "holdout_pearson", "holdout_rmse", "path"]].to_string(index=False))
    print(f"\nDefault submission: {default_name} -> {submission_dir / 'submission_best.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
