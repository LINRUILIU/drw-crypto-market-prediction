from __future__ import annotations

import argparse
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
from drw_crypto.pipeline import make_submission  # noqa: E402
from drw_crypto.preprocessing import TabularPreprocessor, infer_feature_columns, infer_id_column, infer_target_column  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blend current best submission with new candidate signals.")
    parser.add_argument("--config", default="configs/01_main_beta1_signal_blend.yaml", help="Path to YAML config.")
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


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


def best_alpha(path: Path) -> float:
    grid = pd.read_csv(path)
    return float(grid.sort_values("pearson", ascending=False).iloc[0]["alpha"])


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
    print(f"Generated Ridge signal {signal_name}: {output_path.relative_to(ROOT)} alpha={alpha}")
    return output_path


def ensure_signal_submissions(
    config: dict[str, Any],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    id_col: str | None,
    sample_submission: pd.DataFrame | None,
    prediction_col: str,
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for signal_name, signal_cfg in config["signals"].items():
        signal_type = str(signal_cfg["type"])
        if signal_type == "existing_submission":
            paths[signal_name] = ROOT / signal_cfg["submission_path"]
        elif signal_type == "ridge":
            paths[signal_name] = ensure_ridge_submission(
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
        else:
            raise ValueError(f"Unsupported signal type for {signal_name}: {signal_type}")
    return paths


def read_valid_prediction(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(ROOT / path)
    return frame["y_true"].to_numpy(dtype=np.float64), frame["prediction"].to_numpy(dtype=np.float64)


def base_valid_prediction(config: dict[str, Any], split: str) -> tuple[np.ndarray, np.ndarray]:
    base_cfg = config["base"]
    top_y, top_pred = read_valid_prediction(base_cfg["valid_predictions"]["top200"][split])
    stable_y, stable_pred = read_valid_prediction(base_cfg["valid_predictions"]["stability"][split])
    if not np.allclose(top_y, stable_y, rtol=0.0, atol=1e-10):
        raise ValueError(f"Base validation target mismatch for split {split}")
    weights = base_cfg["components"]
    pred = float(weights["top200_weight"]) * top_pred + float(weights["stability_weight"]) * stable_pred
    return top_y, pred


def signal_valid_prediction(config: dict[str, Any], signal_name: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    return read_valid_prediction(config["signals"][signal_name]["valid_predictions"][split])


def evaluate_signal_grid(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    blend_cfg = config["blend"]
    splits = [str(blend_cfg["holdout_split"]), *[str(split) for split in blend_cfg["rolling_splits"]]]
    split_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    objective_cfg = blend_cfg["objective"]

    for signal_name in config["signals"]:
        for weight in [float(value) for value in blend_cfg["candidate_weights"]]:
            split_scores: dict[str, float] = {}
            split_rmses: dict[str, float] = {}
            split_corrs: dict[str, float] = {}
            for split in splits:
                y_true, base_pred = base_valid_prediction(config, split)
                signal_y, signal_pred = signal_valid_prediction(config, signal_name, split)
                if not np.allclose(y_true, signal_y, rtol=0.0, atol=1e-10):
                    raise ValueError(f"Target mismatch for signal {signal_name}, split {split}")
                pred = (1.0 - weight) * base_pred + weight * signal_pred
                split_scores[split] = pearson_corr(y_true, pred)
                split_rmses[split] = rmse(y_true, pred)
                split_corrs[split] = pearson_corr(base_pred, signal_pred)
                split_rows.append(
                    {
                        "signal": signal_name,
                        "signal_weight": weight,
                        "split": split,
                        "pearson": split_scores[split],
                        "rmse": split_rmses[split],
                        "base_signal_corr": split_corrs[split],
                        "prediction_std": float(np.std(pred)),
                    }
                )

            holdout = str(blend_cfg["holdout_split"])
            rolling_scores = np.asarray([split_scores[str(split)] for split in blend_cfg["rolling_splits"]])
            objective = (
                float(objective_cfg["holdout_weight"]) * split_scores[holdout]
                + float(objective_cfg["rolling_min_weight"]) * float(np.min(rolling_scores))
                - float(objective_cfg["rolling_std_penalty"]) * float(np.std(rolling_scores))
            )
            summary_rows.append(
                {
                    "signal": signal_name,
                    "signal_weight": weight,
                    "holdout_pearson": split_scores[holdout],
                    "holdout_rmse": split_rmses[holdout],
                    "rolling_mean": float(np.mean(rolling_scores)),
                    "rolling_std": float(np.std(rolling_scores)),
                    "rolling_min": float(np.min(rolling_scores)),
                    "rolling_max": float(np.max(rolling_scores)),
                    "base_signal_corr_mean": float(np.mean(list(split_corrs.values()))),
                    "objective": float(objective),
                }
            )
    return pd.DataFrame(split_rows), pd.DataFrame(summary_rows)


def read_submission(path: Path, prediction_col: str) -> tuple[pd.DataFrame, np.ndarray]:
    frame = pd.read_csv(path)
    return frame, frame[prediction_col].to_numpy(dtype=np.float64)


def assert_submission_ids_match(left: pd.DataFrame, right: pd.DataFrame, prediction_col: str) -> None:
    for col in left.columns:
        if col == prediction_col:
            continue
        if col in right.columns and not left[col].equals(right[col]):
            raise ValueError(f"Submission id column mismatch: {col}")


def write_blend_submission(
    base_submission: pd.DataFrame,
    base_pred: np.ndarray,
    signal_submission: pd.DataFrame,
    signal_pred: np.ndarray,
    prediction_col: str,
    output_path: Path,
    signal_weight: float,
) -> dict[str, Any]:
    assert_submission_ids_match(base_submission, signal_submission, prediction_col)
    pred = (1.0 - signal_weight) * base_pred + signal_weight * signal_pred
    output = base_submission.copy()
    output[prediction_col] = pred
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    return {
        "path": str(output_path.relative_to(ROOT)),
        "rows": len(output),
        "columns": list(output.columns),
        "prediction_mean": float(np.mean(pred)),
        "prediction_std": float(np.std(pred)),
    }


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
    infer_feature_columns(
        train_df,
        target_col=target_col,
        id_col=id_col,
        drop_columns=data_cfg.get("drop_columns") or [],
        numeric_only=bool(config["preprocessing"].get("numeric_only", True)),
    )

    signal_submission_paths = ensure_signal_submissions(
        config,
        train_df,
        test_df,
        target_col,
        id_col,
        sample_submission,
        prediction_col,
    )

    split_metrics, summary = evaluate_signal_grid(config)
    split_metrics.to_csv(run_dir / "signal_split_metrics.csv", index=False)
    summary.sort_values(["objective", "holdout_pearson"], ascending=[False, False]).to_csv(
        run_dir / "signal_weight_summary_ranked.csv",
        index=False,
    )
    summary.sort_values(["signal", "signal_weight"]).to_csv(run_dir / "signal_weight_summary.csv", index=False)

    base_submission, base_pred = read_submission(ROOT / config["base"]["submission_path"], prediction_col)
    candidate_rows: list[dict[str, Any]] = []
    written: dict[str, dict[str, Any]] = {}
    for candidate in config["candidates"]["fixed"]:
        name = str(candidate["name"])
        signal = str(candidate["signal"])
        weight = float(candidate["signal_weight"])
        signal_submission, signal_pred = read_submission(signal_submission_paths[signal], prediction_col)
        output_path = submission_dir / f"submission_{name}.csv"
        file_info = write_blend_submission(
            base_submission,
            base_pred,
            signal_submission,
            signal_pred,
            prediction_col,
            output_path,
            weight,
        )
        match = summary[(summary["signal"] == signal) & np.isclose(summary["signal_weight"], weight)]
        metrics = match.iloc[0].to_dict() if not match.empty else {}
        row = {"name": name, "signal": signal, "signal_weight": weight, **metrics, **file_info}
        candidate_rows.append(row)
        written[name] = row

    default_name = str(config["candidates"]["default"])
    if default_name not in written:
        raise ValueError(f"Default candidate {default_name!r} was not written.")
    default = written[default_name]
    default_signal_submission, default_signal_pred = read_submission(
        signal_submission_paths[str(default["signal"])],
        prediction_col,
    )
    write_blend_submission(
        base_submission,
        base_pred,
        default_signal_submission,
        default_signal_pred,
        prediction_col,
        submission_dir / "submission_best.csv",
        float(default["signal_weight"]),
    )

    candidates_df = pd.DataFrame(candidate_rows)
    candidates_df.to_csv(run_dir / "candidate_submissions.csv", index=False)
    write_json(
        run_dir / "final_summary.json",
        {
            "base_submission": config["base"]["submission_path"],
            "default_submission": default_name,
            "default_submission_path": str((submission_dir / "submission_best.csv").relative_to(ROOT)),
            "candidate_count": len(candidates_df),
            "signal_submission_paths": {name: str(path.relative_to(ROOT)) for name, path in signal_submission_paths.items()},
        },
    )

    print("\n=== Signal blend candidates ===")
    print(
        candidates_df[
            [
                "name",
                "signal",
                "signal_weight",
                "holdout_pearson",
                "rolling_mean",
                "rolling_std",
                "rolling_min",
                "base_signal_corr_mean",
                "path",
            ]
        ].to_string(index=False)
    )
    print(f"\nDefault submission: {default_name} -> {submission_dir / 'submission_best.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
