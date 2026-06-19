from __future__ import annotations

import argparse
import json
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
from drw_crypto.pipeline import (  # noqa: E402
    choose_prediction_column,
    make_submission,
    split_time_ordered,
    weighted_ridge_prediction,
)
from drw_crypto.preprocessing import infer_id_column, infer_target_column  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Beta7.1 MLP blend-weight refinement.")
    parser.add_argument("--config", default="configs/01_main_beta7_1_mlp_weight_refine.yaml", help="Path to YAML config.")
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_prediction_csv(path: Path, prediction_col: str, expected_len: int | None = None) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if prediction_col in frame.columns:
        pred = frame[prediction_col].to_numpy(dtype=np.float64)
    elif "prediction" in frame.columns:
        pred = frame["prediction"].to_numpy(dtype=np.float64)
    else:
        pred = frame.iloc[:, -1].to_numpy(dtype=np.float64)
    if expected_len is not None and len(pred) != expected_len:
        raise ValueError(f"Prediction length mismatch for {path}: got {len(pred)}, expected {expected_len}")
    return pred


def summarize_delta(pred: np.ndarray, reference: np.ndarray, prefix: str) -> dict[str, float]:
    delta = pred - reference
    return {
        f"{prefix}_corr_current_best": pearson_corr(reference, pred),
        f"{prefix}_delta_std_current_best": float(np.std(delta)),
        f"{prefix}_delta_mae_current_best": float(np.mean(np.abs(delta))),
    }


def reconstruct_beta6_2_valid(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    y_train: np.ndarray,
    y_valid: np.ndarray,
    current_cfg: dict[str, Any],
    prep_cfg: dict[str, Any],
) -> np.ndarray:
    beta4_cfg = current_cfg["beta4_base"]
    beta5_cfg = current_cfg["beta5_interaction"]
    beta6_cfg = current_cfg["beta6_2_supervised_ae"]

    beta3_valid = weighted_ridge_prediction(train_df, y_train, valid_df, beta4_cfg["beta3_components"], prep_cfg, ROOT)
    shap_valid = load_prediction_csv(ROOT / beta4_cfg["shap_stable_valid_prediction_path"], "prediction", len(y_valid))
    beta4_valid = float(beta4_cfg["base_weight"]) * beta3_valid + float(beta4_cfg["signal_weight"]) * shap_valid

    beta5_signal = load_prediction_csv(ROOT / beta5_cfg["valid_prediction_path"], "prediction", len(y_valid))
    beta5_valid = (1.0 - float(beta5_cfg["weight"])) * beta4_valid + float(beta5_cfg["weight"]) * beta5_signal

    beta6_signal = load_prediction_csv(ROOT / beta6_cfg["valid_prediction_path"], "prediction", len(y_valid))
    return (1.0 - float(beta6_cfg["weight"])) * beta5_valid + float(beta6_cfg["weight"]) * beta6_signal


def main() -> int:
    args = parse_args()
    config_path = ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)
    config = load_config(config_path)
    data_cfg = config["data"]
    prep_cfg = config["preprocessing"]
    current_cfg = config["current_best"]
    signal_cfg = config["signal"]
    output_cfg = config["output"]

    run_dir = ensure_dir(ROOT / output_cfg["run_dir"])
    submission_dir = ensure_dir(ROOT / output_cfg["submission_dir"])

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
    prediction_col = choose_prediction_column(sample_submission, output_cfg.get("prediction_col"), target_col)

    train_part, valid_part = split_time_ordered(train_df, float(data_cfg["validation_fraction"]))
    y_train = train_part[target_col].to_numpy(dtype=np.float64)
    y_valid = valid_part[target_col].to_numpy(dtype=np.float64)

    current_valid = reconstruct_beta6_2_valid(train_part, valid_part, y_train, y_valid, current_cfg, prep_cfg)
    current_test = load_prediction_csv(ROOT / current_cfg["test_submission_path"], prediction_col, len(test_df))
    signal_valid = load_prediction_csv(ROOT / signal_cfg["valid_prediction_path"], "prediction", len(y_valid))
    signal_test = load_prediction_csv(ROOT / signal_cfg["test_submission_path"], prediction_col, len(test_df))

    rows: list[dict[str, Any]] = []
    submit_weights = {float(value) for value in config["weights"].get("submit", [])}
    for signal_weight in [float(value) for value in config["weights"]["grid"]]:
        current_weight = 1.0 - signal_weight
        valid_pred = current_weight * current_valid + signal_weight * signal_valid
        test_pred = current_weight * current_test + signal_weight * signal_test
        candidate = f"blend_beta6p2_w{str(current_weight).replace('.', 'p')}_{signal_cfg['name']}_w{str(signal_weight).replace('.', 'p')}"
        path = submission_dir / f"submission_{candidate}.csv"
        make_submission(sample_submission, test_df, test_pred, prediction_col, id_col).to_csv(path, index=False)
        holdout = pearson_corr(y_valid, valid_pred)
        rows.append(
            {
                "candidate": candidate,
                "signal": signal_cfg["name"],
                "current_best_weight": current_weight,
                "signal_weight": signal_weight,
                "holdout_pearson": holdout,
                "holdout_rmse": rmse(y_valid, valid_pred),
                "holdout_delta_current_best": holdout - pearson_corr(y_valid, current_valid),
                "valid_prediction_std": float(np.std(valid_pred)),
                "submission_prediction_std": float(np.std(test_pred)),
                "std_ratio_current_best": float(np.std(test_pred) / np.std(current_test)),
                **summarize_delta(valid_pred, current_valid, "valid"),
                **summarize_delta(test_pred, current_test, "submission"),
                "selected_for_submit": signal_weight in submit_weights,
                "path": str(path.relative_to(ROOT)),
            }
        )

    metrics = pd.DataFrame(rows).sort_values(["holdout_pearson", "signal_weight"], ascending=[False, True])
    metrics.to_csv(run_dir / "weight_candidate_metrics.csv", index=False)
    selected = metrics[metrics["selected_for_submit"]].copy()
    selected.to_csv(run_dir / "selected_candidates.csv", index=False)
    best = metrics.iloc[0]
    best_path = ROOT / str(best["path"])
    if best_path.exists():
        (submission_dir / "submission_best_local.csv").write_bytes(best_path.read_bytes())

    write_json(
        run_dir / "final_summary.json",
        {
            "current_best": current_cfg,
            "current_best_holdout_pearson": pearson_corr(y_valid, current_valid),
            "signal": signal_cfg,
            "weights": config["weights"],
            "best_local_candidate": best.to_dict(),
            "selected_candidates": selected.to_dict(orient="records"),
        },
    )
    print(metrics.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
