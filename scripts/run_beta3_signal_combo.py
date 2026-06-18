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

from drw_crypto.io import ensure_dir, write_json  # noqa: E402
from drw_crypto.metrics import pearson_corr, rmse  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blend Beta3 residual and Spearman signals with the current best.")
    parser.add_argument("--config", default="configs/01_main_beta3_signal_combo.yaml", help="Path to YAML config.")
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def safe_name(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def read_valid_prediction(path: Path) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path)
    if "y_true" not in frame.columns or "prediction" not in frame.columns:
        raise ValueError(f"Expected y_true,prediction columns in {path}")
    return frame["y_true"].to_numpy(dtype=np.float64), frame["prediction"].to_numpy(dtype=np.float64)


def read_submission(path: Path, prediction_col: str) -> tuple[pd.DataFrame, np.ndarray]:
    frame = pd.read_csv(path)
    if prediction_col not in frame.columns:
        raise ValueError(f"Prediction column {prediction_col!r} is not in {path}")
    return frame, frame[prediction_col].to_numpy(dtype=np.float64)


def bin_pearson_summary(y_true: np.ndarray, pred: np.ndarray, bins: int) -> dict[str, float]:
    values: list[float] = []
    for indices in np.array_split(np.arange(len(y_true)), int(bins)):
        if len(indices) < 2:
            continue
        values.append(pearson_corr(y_true[indices], pred[indices]))
    if not values:
        values = [0.0]
    arr = np.asarray(values, dtype=np.float64)
    return {
        "valid_bin_mean_pearson": float(np.mean(arr)),
        "valid_bin_min_pearson": float(np.min(arr)),
        "valid_bin_std_pearson": float(np.std(arr, ddof=0)),
    }


def prediction_delta(pred: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    delta = pred - reference
    return {
        "corr_reference": pearson_corr(reference, pred),
        "delta_std_reference": float(np.std(delta)),
        "delta_mae_reference": float(np.mean(np.abs(delta))),
    }


def weighted_sum(weights: dict[str, float], predictions: dict[str, np.ndarray]) -> np.ndarray:
    output: np.ndarray | None = None
    for name, weight in weights.items():
        if name not in predictions:
            raise KeyError(f"Missing prediction source {name!r}")
        part = float(weight) * predictions[name]
        output = part if output is None else output + part
    if output is None:
        raise ValueError("No weights were provided.")
    return output


def main() -> int:
    args = parse_args()
    config_path = ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)
    config = load_config(config_path)

    output_cfg = config["output"]
    prediction_col = str(output_cfg.get("prediction_col") or "prediction")
    run_dir = ensure_dir(ROOT / output_cfg["run_dir"])
    submission_dir = ensure_dir(ROOT / output_cfg["submission_dir"])

    source_cfg = config["sources"]
    valid_predictions: dict[str, np.ndarray] = {}
    test_predictions: dict[str, np.ndarray] = {}
    y_true: np.ndarray | None = None
    submission_template: pd.DataFrame | None = None

    for name, item in source_cfg.items():
        valid_path = item.get("valid_prediction_path")
        if valid_path:
            y_candidate, pred = read_valid_prediction(ROOT / valid_path)
            if y_true is None:
                y_true = y_candidate
            elif len(y_true) != len(y_candidate) or not np.allclose(y_true, y_candidate):
                raise ValueError(f"Validation target mismatch for {name}")
            valid_predictions[name] = pred

        submission_path = item.get("signal_submission_path") or item.get("submission_path")
        if submission_path:
            frame, pred = read_submission(ROOT / submission_path, prediction_col)
            if submission_template is None:
                submission_template = frame.copy()
            elif len(submission_template) != len(frame):
                raise ValueError(f"Submission length mismatch for {name}")
            test_predictions[name] = pred

    if y_true is None or submission_template is None:
        raise ValueError("Need at least one valid prediction and one submission template.")

    for name, item in source_cfg.items():
        if item.get("kind") != "formula":
            continue
        valid_predictions[name] = weighted_sum(item["valid_formula"], valid_predictions)
        if "submission_path" in item:
            _, test_predictions[name] = read_submission(ROOT / item["submission_path"], prediction_col)

    current_cfg = config["current_best"]
    current_best_valid = weighted_sum(current_cfg["valid_formula"], valid_predictions)
    _, current_best_test = read_submission(ROOT / current_cfg["submission_path"], prediction_col)
    beta2_base_valid = valid_predictions["beta2_base"]
    beta2_base_test = test_predictions["beta2_base"]
    residual_valid = valid_predictions["residual_pearson_top100"]
    residual_test = test_predictions["residual_pearson_top100"]
    spearman_valid = valid_predictions["spearman_top50"]
    spearman_test = test_predictions["spearman_top50"]

    valid_bins = int(config["selection"]["valid_bins"])
    current_best_holdout = pearson_corr(y_true, current_best_valid)
    beta2_base_holdout = pearson_corr(y_true, beta2_base_valid)
    current_best_summary = {
        "current_best_holdout_pearson": current_best_holdout,
        "current_best_rmse": rmse(y_true, current_best_valid),
        "current_best_valid_prediction_std": float(np.std(current_best_valid)),
        "current_best_submission_prediction_std": float(np.std(current_best_test)),
        **bin_pearson_summary(y_true, current_best_valid, valid_bins),
    }

    candidate_rows: list[dict[str, Any]] = []
    candidate_predictions: dict[str, np.ndarray] = {}
    residual_weights = [float(value) for value in config["grid"]["residual_weights"]]
    spearman_weights = [float(value) for value in config["grid"]["spearman_weights"]]
    min_total_signal = float(config["grid"]["min_total_signal_weight"])
    max_total_signal = float(config["grid"]["max_total_signal_weight"])
    require_residual = bool(config["grid"].get("require_positive_residual", False))
    require_spearman = bool(config["grid"].get("require_positive_spearman", False))

    for residual_weight in residual_weights:
        for spearman_weight in spearman_weights:
            if require_residual and residual_weight <= 0:
                continue
            if require_spearman and spearman_weight <= 0:
                continue
            total_signal_weight = residual_weight + spearman_weight
            if total_signal_weight < min_total_signal or total_signal_weight > max_total_signal:
                continue
            base_weight = 1.0 - total_signal_weight
            if base_weight < 0:
                continue

            name = (
                f"blend_base_w{safe_name(base_weight)}"
                f"_resid_w{safe_name(residual_weight)}"
                f"_spear_w{safe_name(spearman_weight)}"
            )
            valid_pred = base_weight * beta2_base_valid + residual_weight * residual_valid + spearman_weight * spearman_valid
            test_pred = base_weight * beta2_base_test + residual_weight * residual_test + spearman_weight * spearman_test
            candidate_predictions[name] = test_pred

            valid_current_delta = prediction_delta(valid_pred, current_best_valid)
            test_current_delta = prediction_delta(test_pred, current_best_test)
            valid_base_delta = prediction_delta(valid_pred, beta2_base_valid)
            test_base_delta = prediction_delta(test_pred, beta2_base_test)
            bin_summary = bin_pearson_summary(y_true, valid_pred, valid_bins)
            holdout = pearson_corr(y_true, valid_pred)
            candidate_rows.append(
                {
                    "candidate": name,
                    "base_weight": base_weight,
                    "residual_weight": residual_weight,
                    "spearman_weight": spearman_weight,
                    "total_signal_weight": total_signal_weight,
                    "holdout_pearson": holdout,
                    "holdout_rmse": rmse(y_true, valid_pred),
                    "holdout_delta_current_best": holdout - current_best_holdout,
                    "holdout_delta_beta2_base": holdout - beta2_base_holdout,
                    "valid_prediction_std": float(np.std(valid_pred)),
                    "submission_prediction_std": float(np.std(test_pred)),
                    "std_ratio_current_best": float(np.std(test_pred) / np.std(current_best_test)),
                    "valid_corr_current_best": valid_current_delta["corr_reference"],
                    "submission_corr_current_best": test_current_delta["corr_reference"],
                    "submission_delta_std_current_best": test_current_delta["delta_std_reference"],
                    "submission_delta_mae_current_best": test_current_delta["delta_mae_reference"],
                    "valid_corr_beta2_base": valid_base_delta["corr_reference"],
                    "submission_corr_beta2_base": test_base_delta["corr_reference"],
                    "submission_delta_std_beta2_base": test_base_delta["delta_std_reference"],
                    "submission_delta_mae_beta2_base": test_base_delta["delta_mae_reference"],
                    **bin_summary,
                }
            )

    candidates = pd.DataFrame(candidate_rows)
    candidates["eligible"] = (
        (candidates["holdout_delta_current_best"] >= -float(config["selection"]["holdout_tolerance"]))
        & (candidates["std_ratio_current_best"] <= float(config["selection"]["max_std_ratio_current_best"]))
    )
    candidates_sorted = candidates.sort_values(
        ["holdout_pearson", "valid_bin_min_pearson", "submission_corr_current_best"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    candidates_sorted.to_csv(run_dir / "candidate_metrics.csv", index=False)

    selected_names: list[str] = []
    selected_reasons: dict[str, str] = {}
    max_selected = int(config["selection"]["max_selected_submissions"])
    eligible = candidates_sorted[candidates_sorted["eligible"]].copy()
    selection_pools = [
        ("holdout_best", eligible),
        (
            "balanced_residual_spearman",
            eligible.assign(weight_gap=(eligible["residual_weight"] - eligible["spearman_weight"]).abs()).sort_values(
                ["weight_gap", "holdout_pearson"],
                ascending=[True, False],
            ),
        ),
        (
            "lower_corr_near_best",
            eligible[
                eligible["holdout_pearson"] >= eligible["holdout_pearson"].max() - float(config["selection"]["holdout_tolerance"])
            ].sort_values(["submission_corr_current_best", "holdout_pearson"], ascending=[True, False]),
        ),
    ]
    for reason, pool in selection_pools:
        if len(selected_names) >= max_selected or pool.empty:
            continue
        for _, row in pool.iterrows():
            candidate = str(row["candidate"])
            if candidate in selected_names:
                continue
            selected_names.append(candidate)
            selected_reasons[candidate] = reason
            break

    if len(selected_names) < max_selected:
        for _, row in eligible.iterrows():
            candidate = str(row["candidate"])
            if candidate in selected_names:
                continue
            selected_names.append(candidate)
            selected_reasons[candidate] = "overall_best"
            if len(selected_names) >= max_selected:
                break

    selected = candidates[candidates["candidate"].isin(selected_names)].copy()
    selected["selection_reason"] = selected["candidate"].map(selected_reasons)
    selected["path"] = ""
    for candidate in selected_names:
        pred = candidate_predictions[candidate]
        output = submission_template.copy()
        output[prediction_col] = pred
        path = submission_dir / f"submission_{candidate}.csv"
        output.to_csv(path, index=False)
        selected.loc[selected["candidate"] == candidate, "path"] = str(path.relative_to(ROOT))

    if selected_names:
        selected = selected.set_index("candidate").loc[selected_names].reset_index()
        shutil.copyfile(submission_dir / f"submission_{selected_names[0]}.csv", submission_dir / "submission_best.csv")
    selected.to_csv(run_dir / "selected_candidates.csv", index=False)

    write_json(
        run_dir / "final_summary.json",
        {
            "current_best": current_cfg,
            "current_best_summary": current_best_summary,
            "beta2_base_holdout_pearson": beta2_base_holdout,
            "candidate_count": len(candidates),
            "eligible_candidate_count": int(candidates["eligible"].sum()),
            "selected_candidates": selected_names,
            "leakage_control": "This script only blends precomputed train-split validation predictions and test submissions; no validation target is used outside metric evaluation.",
        },
    )

    print("\n=== Current best summary ===")
    print(json.dumps(current_best_summary, indent=2))
    print("\n=== Top candidates ===")
    print(candidates_sorted.head(30).to_string(index=False))
    print("\n=== Selected candidates ===")
    print(selected.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
