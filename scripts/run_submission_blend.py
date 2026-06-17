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

from drw_crypto.io import ensure_dir, write_json  # noqa: E402
from drw_crypto.metrics import pearson_corr, rmse  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blend existing submissions and validation predictions.")
    parser.add_argument("--config", default="configs/01_main_submission_blend.yaml", help="Path to YAML config.")
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    std = float(np.std(values))
    if not np.isfinite(std) or std == 0.0:
        return values - float(np.mean(values))
    return (values - float(np.mean(values))) / std


def blend_predictions(top200: np.ndarray, stability: np.ndarray, top200_weight: float, mode: str) -> np.ndarray:
    top200 = np.asarray(top200, dtype=np.float64)
    stability = np.asarray(stability, dtype=np.float64)
    if mode == "raw":
        return top200_weight * top200 + (1.0 - top200_weight) * stability
    if mode == "zscore":
        blended = top200_weight * zscore(top200) + (1.0 - top200_weight) * zscore(stability)
        return float(np.mean(top200)) + float(np.std(top200)) * blended
    raise ValueError(f"Unsupported blend mode: {mode}")


def read_valid_pair(config: dict[str, Any], split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    top_path = ROOT / config["sources"]["top200"]["valid_predictions"][split]
    stable_path = ROOT / config["sources"]["stability"]["valid_predictions"][split]
    top = pd.read_csv(top_path)
    stable = pd.read_csv(stable_path)
    if len(top) != len(stable):
        raise ValueError(f"Validation length mismatch for {split}: {len(top)} vs {len(stable)}")
    y_top = top["y_true"].to_numpy(dtype=np.float64)
    y_stable = stable["y_true"].to_numpy(dtype=np.float64)
    if not np.allclose(y_top, y_stable, rtol=0.0, atol=1e-10):
        raise ValueError(f"Validation target mismatch for {split}")
    return (
        y_top,
        top["prediction"].to_numpy(dtype=np.float64),
        stable["prediction"].to_numpy(dtype=np.float64),
    )


def evaluate_grid(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    blend_cfg = config["blend"]
    holdout_split = str(blend_cfg["holdout_split"])
    rolling_splits = [str(split) for split in blend_cfg["rolling_splits"]]
    all_splits = [holdout_split, *rolling_splits]

    split_cache = {split: read_valid_pair(config, split) for split in all_splits}
    split_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for mode in blend_cfg["modes"]:
        for weight in [float(value) for value in blend_cfg["top200_weight_grid"]]:
            split_scores: dict[str, float] = {}
            split_rmses: dict[str, float] = {}
            for split, (y_true, top_pred, stable_pred) in split_cache.items():
                pred = blend_predictions(top_pred, stable_pred, weight, str(mode))
                score = pearson_corr(y_true, pred)
                error = rmse(y_true, pred)
                source_corr = pearson_corr(top_pred, stable_pred)
                split_scores[split] = score
                split_rmses[split] = error
                split_rows.append(
                    {
                        "mode": mode,
                        "top200_weight": weight,
                        "split": split,
                        "pearson": score,
                        "rmse": error,
                        "source_prediction_corr": source_corr,
                        "prediction_mean": float(np.mean(pred)),
                        "prediction_std": float(np.std(pred)),
                    }
                )

            rolling_scores = np.asarray([split_scores[split] for split in rolling_splits], dtype=np.float64)
            public_proxy = (
                weight * float(config["sources"]["top200"]["public_score"])
                + (1.0 - weight) * float(config["sources"]["stability"]["public_score"])
            )
            private_proxy = (
                weight * float(config["sources"]["top200"]["private_score"])
                + (1.0 - weight) * float(config["sources"]["stability"]["private_score"])
            )
            objective_cfg = blend_cfg["balanced_objective"]
            balanced_objective = (
                float(objective_cfg["holdout_weight"]) * split_scores[holdout_split]
                + float(objective_cfg["rolling_min_weight"]) * float(np.min(rolling_scores))
                - float(objective_cfg["rolling_std_penalty"]) * float(np.std(rolling_scores))
            )
            summary_rows.append(
                {
                    "mode": mode,
                    "top200_weight": weight,
                    "holdout_pearson": split_scores[holdout_split],
                    "holdout_rmse": split_rmses[holdout_split],
                    "rolling_mean": float(np.mean(rolling_scores)),
                    "rolling_std": float(np.std(rolling_scores)),
                    "rolling_min": float(np.min(rolling_scores)),
                    "rolling_max": float(np.max(rolling_scores)),
                    "balanced_objective": float(balanced_objective),
                    "public_linear_proxy": float(public_proxy),
                    "private_linear_proxy": float(private_proxy),
                }
            )

    return pd.DataFrame(split_rows), pd.DataFrame(summary_rows)


def select_private_safe(summary: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    blend_cfg = config["blend"]
    mode = str(blend_cfg.get("private_safe_mode", "raw"))
    top200_holdout = float(
        summary[(summary["mode"] == mode) & (summary["top200_weight"] == 1.0)]["holdout_pearson"].iloc[0]
    )
    drop_limit = float(blend_cfg.get("private_safe_holdout_drop_limit", 0.001))
    eligible = summary[
        (summary["mode"] == mode)
        & (summary["holdout_pearson"] >= top200_holdout - drop_limit)
        & (summary["top200_weight"] < 1.0)
    ].copy()
    if eligible.empty:
        eligible = summary[(summary["mode"] == mode) & (summary["top200_weight"] == 1.0)].copy()
    return eligible.sort_values(["rolling_min", "rolling_std"], ascending=[False, True]).iloc[0].to_dict()


def select_balanced(summary: pd.DataFrame) -> dict[str, Any]:
    return summary.sort_values(["balanced_objective", "holdout_pearson"], ascending=[False, False]).iloc[0].to_dict()


def read_submission_pair(config: dict[str, Any]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    top_path = ROOT / config["sources"]["top200"]["submission_path"]
    stable_path = ROOT / config["sources"]["stability"]["submission_path"]
    top = pd.read_csv(top_path)
    stable = pd.read_csv(stable_path)
    if len(top) != len(stable):
        raise ValueError(f"Submission length mismatch: {len(top)} vs {len(stable)}")
    prediction_col = str(config["output"].get("prediction_col") or "prediction")
    non_prediction_cols = [col for col in top.columns if col != prediction_col]
    for col in non_prediction_cols:
        if col in stable.columns and not top[col].equals(stable[col]):
            raise ValueError(f"Submission id column mismatch: {col}")
    return (
        top.copy(),
        top[prediction_col].to_numpy(dtype=np.float64),
        stable[prediction_col].to_numpy(dtype=np.float64),
    )


def write_candidate_submission(
    base_submission: pd.DataFrame,
    top_pred: np.ndarray,
    stable_pred: np.ndarray,
    prediction_col: str,
    output_path: Path,
    mode: str,
    top200_weight: float,
) -> dict[str, Any]:
    pred = blend_predictions(top_pred, stable_pred, top200_weight, mode)
    submission = base_submission.copy()
    submission[prediction_col] = pred
    submission.to_csv(output_path, index=False)
    return {
        "path": str(output_path.relative_to(ROOT)),
        "rows": int(len(submission)),
        "columns": list(submission.columns),
        "prediction_mean": float(np.mean(pred)),
        "prediction_std": float(np.std(pred)),
        "prediction_min": float(np.min(pred)),
        "prediction_max": float(np.max(pred)),
    }


def main() -> int:
    args = parse_args()
    config_path = ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)
    config = load_config(config_path)
    run_dir = ensure_dir(ROOT / config["output"]["run_dir"])
    submission_dir = ensure_dir(ROOT / config["output"]["submission_dir"])

    split_metrics, summary = evaluate_grid(config)
    split_metrics.to_csv(run_dir / "blend_split_metrics.csv", index=False)
    summary.sort_values(["mode", "top200_weight"]).to_csv(run_dir / "blend_weight_summary.csv", index=False)
    summary.sort_values(["balanced_objective", "holdout_pearson"], ascending=[False, False]).to_csv(
        run_dir / "blend_weight_summary_ranked.csv",
        index=False,
    )

    selected_private_safe = select_private_safe(summary, config)
    selected_balanced = select_balanced(summary)
    fixed_candidates = list(config.get("candidates", {}).get("fixed", []))

    candidate_rows: list[dict[str, Any]] = []
    candidate_specs: list[dict[str, Any]] = [
        {"name": "auto_private_safe", **selected_private_safe},
        {"name": "auto_balanced", **selected_balanced},
        *fixed_candidates,
    ]

    base_submission, top_submission_pred, stable_submission_pred = read_submission_pair(config)
    prediction_col = str(config["output"].get("prediction_col") or "prediction")
    seen: set[str] = set()
    written: dict[str, Any] = {}
    for candidate in candidate_specs:
        name = str(candidate["name"])
        if name in seen:
            continue
        seen.add(name)
        mode = str(candidate["mode"])
        top200_weight = float(candidate["top200_weight"])
        match = summary[(summary["mode"] == mode) & np.isclose(summary["top200_weight"], top200_weight)]
        metrics = match.iloc[0].to_dict() if not match.empty else {}
        output_path = submission_dir / f"submission_{name}.csv"
        file_info = write_candidate_submission(
            base_submission,
            top_submission_pred,
            stable_submission_pred,
            prediction_col,
            output_path,
            mode,
            top200_weight,
        )
        row = {"name": name, "mode": mode, "top200_weight": top200_weight, **metrics, **file_info}
        candidate_rows.append(row)
        written[name] = row

    default_name = str(config.get("candidates", {}).get("default", "private_safe"))
    if default_name not in written:
        default_name = "auto_private_safe"
    default_candidate = written[default_name]
    write_candidate_submission(
        base_submission,
        top_submission_pred,
        stable_submission_pred,
        prediction_col,
        submission_dir / "submission_best.csv",
        str(default_candidate["mode"]),
        float(default_candidate["top200_weight"]),
    )

    candidates_df = pd.DataFrame(candidate_rows)
    candidates_df.to_csv(run_dir / "candidate_submissions.csv", index=False)

    source_submission_corr = pearson_corr(top_submission_pred, stable_submission_pred)
    final_summary = {
        "source_scores": {
            "top200": {
                "public": float(config["sources"]["top200"]["public_score"]),
                "private": float(config["sources"]["top200"]["private_score"]),
            },
            "stability": {
                "public": float(config["sources"]["stability"]["public_score"]),
                "private": float(config["sources"]["stability"]["private_score"]),
            },
        },
        "source_submission_corr": source_submission_corr,
        "auto_private_safe": selected_private_safe,
        "auto_balanced": selected_balanced,
        "default_submission": default_name,
        "default_submission_path": str((submission_dir / "submission_best.csv").relative_to(ROOT)),
        "candidate_count": int(len(candidates_df)),
    }
    write_json(run_dir / "final_summary.json", final_summary)
    with (run_dir / "final_summary_compact.json").open("w", encoding="utf-8") as f:
        json.dump(final_summary, f, indent=2, ensure_ascii=True, default=str)

    print("\n=== Candidate submissions ===")
    print(candidates_df[["name", "mode", "top200_weight", "holdout_pearson", "rolling_mean", "rolling_std", "rolling_min", "public_linear_proxy", "private_linear_proxy", "path"]].to_string(index=False))
    print(f"\nDefault submission: {default_name} -> {submission_dir / 'submission_best.csv'}")
    print(f"Source submission correlation: {source_submission_corr:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
