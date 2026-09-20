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

from drw_crypto.feature_selection import (  # noqa: E402
    spearman_feature_ranking,
    stable_pearson_feature_ranking,
    target_feature_ranking,
)
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
    parser = argparse.ArgumentParser(description="Run Beta3 new feature-selection Ridge signals.")
    parser.add_argument("--config", default="configs/01_main_beta3_new_feature_signals.yaml", help="Path to YAML config.")
    parser.add_argument("--max-train-rows", type=int, default=None, help="Optional tail sample for smoke runs.")
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


def save_valid_prediction(path: Path, y_true: np.ndarray, pred: np.ndarray) -> None:
    pd.DataFrame(
        {
            "row_index": np.arange(len(y_true), dtype=np.int64),
            "y_true": y_true,
            "prediction": pred,
        }
    ).to_csv(path, index=False)


def safe_name(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def fit_ridge(alpha: float, x_train: np.ndarray, y_train: np.ndarray, x_pred: np.ndarray) -> np.ndarray:
    return NumpyRidgeRegressor(alpha=alpha).fit(x_train, y_train).predict(x_pred)


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
        pred = fit_ridge(alpha, x_train, y_train, x_valid)
        elapsed = time.perf_counter() - start
        row = {
            "alpha": float(alpha),
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


def weighted_ridge_prediction(
    fit_df: pd.DataFrame,
    y_fit: np.ndarray,
    pred_df: pd.DataFrame,
    components: list[dict[str, Any]],
    prep_cfg: dict[str, Any],
) -> np.ndarray:
    output: np.ndarray | None = None
    for component in components:
        selected = read_lines(ROOT / component["selected_features_path"])
        preprocessor = TabularPreprocessor(selected).fit(
            fit_df,
            missing_fill=str(prep_cfg.get("missing_fill", "median")),
            standardize=bool(prep_cfg.get("standardize", True)),
        )
        x_fit = preprocessor.transform(fit_df)
        x_pred = preprocessor.transform(pred_df)
        pred = fit_ridge(float(component["alpha"]), x_fit, y_fit, x_pred)
        weighted = float(component["weight"]) * pred
        output = weighted if output is None else output + weighted
        del preprocessor, x_fit, x_pred, pred, weighted
        gc.collect()
    if output is None:
        raise ValueError("No current-best components configured.")
    return output


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


def summarize_delta(pred: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    delta = pred - reference
    return {
        "corr_current_best": pearson_corr(reference, pred),
        "delta_std_current_best": float(np.std(delta)),
        "delta_mae_current_best": float(np.mean(np.abs(delta))),
    }


def build_ranking(
    family_cfg: dict[str, Any],
    train_part: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    residual: pd.Series,
    stable_cfg: dict[str, Any],
) -> pd.DataFrame:
    method = str(family_cfg["method"])
    if method == "spearman":
        return spearman_feature_ranking(train_part, feature_cols, target_col)
    if method == "stable_pearson":
        return stable_pearson_feature_ranking(
            train_part,
            feature_cols,
            target_col,
            stable_cfg["windows"],
            float(stable_cfg["stability_penalty"]),
            float(stable_cfg["min_weight"]),
        )
    if method == "residual_pearson":
        return target_feature_ranking(train_part, feature_cols, residual, "pearson", "residual_pearson")
    if method == "residual_spearman":
        return target_feature_ranking(train_part, feature_cols, residual, "spearman", "residual_spearman")
    raise ValueError(f"Unsupported ranking method: {method}")


def main() -> int:
    args = parse_args()
    config_path = ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)
    config = load_config(config_path)

    data_cfg = config["data"]
    prep_cfg = config["preprocessing"]
    current_cfg = config["current_best"]
    stable_cfg = config["stable_ranking"]
    ridge_cfg = config["ridge"]
    blend_cfg = config["blend"]
    output_cfg = config["output"]

    run_dir = ensure_dir(ROOT / output_cfg["run_dir"])
    ranking_dir = ensure_dir(run_dir / "feature_rankings")
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
    test_df = read_table(test_path)
    sample_submission = read_table(sample_path) if sample_path is not None else None

    max_train_rows = args.max_train_rows if args.max_train_rows is not None else data_cfg.get("max_train_rows")
    if max_train_rows and max_train_rows < len(train_df):
        train_df = train_df.tail(int(max_train_rows)).reset_index(drop=True)

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

    train_part, valid_part = split_time_ordered(train_df, float(data_cfg["validation_fraction"]))
    y_train = train_part[target_col].to_numpy(dtype=np.float64)
    y_valid = valid_part[target_col].to_numpy(dtype=np.float64)
    y_full = train_df[target_col].to_numpy(dtype=np.float64)

    print("=== Computing current-best train/valid predictions ===")
    current_components = list(current_cfg["components"])
    current_best_train = weighted_ridge_prediction(train_part, y_train, train_part, current_components, prep_cfg)
    current_best_valid = weighted_ridge_prediction(train_part, y_train, valid_part, current_components, prep_cfg)
    current_best_test = read_submission_prediction(ROOT / current_cfg["submission_path"], prediction_col)
    current_best_holdout = pearson_corr(y_valid, current_best_valid)
    current_best_bins = bin_pearson_summary(y_valid, current_best_valid, int(blend_cfg["valid_bins"]))
    residual = pd.Series(y_train - current_best_train, index=train_part.index, name="current_best_residual")
    residual_summary = {
        "residual_std": float(np.std(residual.to_numpy(dtype=np.float64))),
        "residual_corr_current_best_train": pearson_corr(y_train, current_best_train),
        "current_best_holdout_pearson": current_best_holdout,
        **current_best_bins,
    }
    write_json(run_dir / "current_best_summary.json", residual_summary)

    current_feature_union: set[str] = set()
    for component in current_components:
        current_feature_union.update(read_lines(ROOT / component["selected_features_path"]))

    ridge_alphas = [float(alpha) for alpha in ridge_cfg["alphas"]]
    signal_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    signal_valid_predictions: dict[str, np.ndarray] = {}
    signal_test_predictions: dict[str, np.ndarray] = {}
    candidate_predictions: dict[str, np.ndarray] = {}

    for family_cfg in config["ranking_families"]:
        family_name = str(family_cfg["name"])
        family_group = str(family_cfg["family"])
        print(f"\n=== Ranking family: {family_name} ===")
        ranking = build_ranking(family_cfg, train_part, feature_cols, target_col, residual, stable_cfg)
        ranking.to_csv(ranking_dir / f"{family_name}_feature_ranking.csv", index=False)

        for top_k in [int(value) for value in family_cfg["top_k"]]:
            selected = ranking["feature"].head(min(top_k, len(feature_cols))).tolist()
            signal_name = f"{family_name}_top{top_k}"
            write_lines(run_dir / f"selected_features_{signal_name}.txt", selected)
            overlap = len(set(selected) & current_feature_union) / max(1, len(selected))
            print(f"Training {signal_name}: {len(selected)} features, overlap with current union={overlap:.3f}")

            split_preprocessor = TabularPreprocessor(selected).fit(
                train_part,
                missing_fill=str(prep_cfg.get("missing_fill", "median")),
                standardize=bool(prep_cfg.get("standardize", True)),
            )
            x_train = split_preprocessor.transform(train_part)
            x_valid = split_preprocessor.transform(valid_part)
            best, valid_pred, alpha_grid = train_ridge_search(x_train, y_train, x_valid, y_valid, ridge_alphas)
            alpha_grid.to_csv(run_dir / f"ridge_alpha_search_{signal_name}.csv", index=False)
            valid_path = pred_dir / f"valid_{signal_name}_ridge.csv"
            save_valid_prediction(valid_path, y_valid, valid_pred)

            full_preprocessor = TabularPreprocessor(selected).fit(
                train_df,
                missing_fill=str(prep_cfg.get("missing_fill", "median")),
                standardize=bool(prep_cfg.get("standardize", True)),
            )
            x_full = full_preprocessor.transform(train_df)
            x_test = full_preprocessor.transform(test_df)
            test_pred = fit_ridge(float(best["alpha"]), x_full, y_full, x_test)
            signal_valid_predictions[signal_name] = valid_pred
            signal_test_predictions[signal_name] = test_pred
            signal_path = signal_submission_dir / f"submission_{signal_name}_ridge.csv"
            make_submission(sample_submission, test_df, test_pred, prediction_col, id_col).to_csv(signal_path, index=False)

            valid_signal_delta = summarize_delta(valid_pred, current_best_valid)
            test_signal_delta = summarize_delta(test_pred, current_best_test)
            signal_bins = bin_pearson_summary(y_valid, valid_pred, int(blend_cfg["valid_bins"]))
            signal_rows.append(
                {
                    "signal": signal_name,
                    "family": family_group,
                    "ranking": family_name,
                    "top_k": top_k,
                    "feature_count": len(selected),
                    "feature_overlap_current_union": overlap,
                    "best_alpha": float(best["alpha"]),
                    "holdout_pearson": pearson_corr(y_valid, valid_pred),
                    "holdout_rmse": rmse(y_valid, valid_pred),
                    "valid_prediction_std": float(np.std(valid_pred)),
                    "submission_prediction_std": float(np.std(test_pred)),
                    "valid_corr_current_best": valid_signal_delta["corr_current_best"],
                    "submission_corr_current_best": test_signal_delta["corr_current_best"],
                    "valid_bin_mean_pearson": signal_bins["valid_bin_mean_pearson"],
                    "valid_bin_min_pearson": signal_bins["valid_bin_min_pearson"],
                    "valid_bin_std_pearson": signal_bins["valid_bin_std_pearson"],
                    "valid_prediction_path": str(valid_path.relative_to(ROOT)),
                    "signal_submission_path": str(signal_path.relative_to(ROOT)),
                }
            )

            for signal_weight in [float(value) for value in blend_cfg["signal_weights"]]:
                current_weight = 1.0 - signal_weight
                candidate_name = f"blend_{signal_name}_w{safe_name(signal_weight)}"
                blend_valid = current_weight * current_best_valid + signal_weight * valid_pred
                blend_test = current_weight * current_best_test + signal_weight * test_pred
                candidate_predictions[candidate_name] = blend_test

                valid_blend_delta = summarize_delta(blend_valid, current_best_valid)
                test_blend_delta = summarize_delta(blend_test, current_best_test)
                blend_bins = bin_pearson_summary(y_valid, blend_valid, int(blend_cfg["valid_bins"]))
                signal_too_homogeneous = test_signal_delta["corr_current_best"] > float(
                    blend_cfg["max_signal_corr_current_best"]
                )
                holdout_drop = current_best_holdout - pearson_corr(y_valid, blend_valid)
                bin_min_drop = current_best_bins["valid_bin_min_pearson"] - blend_bins["valid_bin_min_pearson"]
                candidate_rows.append(
                    {
                        "candidate": candidate_name,
                        "signal": signal_name,
                        "family": family_group,
                        "ranking": family_name,
                        "top_k": top_k,
                        "best_alpha": float(best["alpha"]),
                        "signal_weight": signal_weight,
                        "current_best_weight": current_weight,
                        "holdout_pearson": pearson_corr(y_valid, blend_valid),
                        "holdout_rmse": rmse(y_valid, blend_valid),
                        "valid_prediction_std": float(np.std(blend_valid)),
                        "submission_prediction_std": float(np.std(blend_test)),
                        "signal_valid_corr_current_best": valid_signal_delta["corr_current_best"],
                        "signal_submission_corr_current_best": test_signal_delta["corr_current_best"],
                        "blend_valid_corr_current_best": valid_blend_delta["corr_current_best"],
                        "blend_submission_corr_current_best": test_blend_delta["corr_current_best"],
                        "blend_delta_std_current_best": test_blend_delta["delta_std_current_best"],
                        "blend_delta_mae_current_best": test_blend_delta["delta_mae_current_best"],
                        "valid_bin_mean_pearson": blend_bins["valid_bin_mean_pearson"],
                        "valid_bin_min_pearson": blend_bins["valid_bin_min_pearson"],
                        "valid_bin_std_pearson": blend_bins["valid_bin_std_pearson"],
                        "feature_overlap_current_union": overlap,
                        "signal_too_homogeneous": signal_too_homogeneous,
                        "holdout_drop_current_best": holdout_drop,
                        "valid_bin_min_drop_current_best": bin_min_drop,
                    }
                )

            del split_preprocessor, full_preprocessor, x_train, x_valid, x_full, x_test, valid_pred, test_pred
            gc.collect()

    signal_metrics = pd.DataFrame(signal_rows).sort_values("holdout_pearson", ascending=False).reset_index(drop=True)
    signal_metrics.to_csv(run_dir / "signal_metrics.csv", index=False)

    candidates = pd.DataFrame(candidate_rows)
    candidates["eligible"] = (
        (~candidates["signal_too_homogeneous"])
        & (candidates["holdout_drop_current_best"] <= float(blend_cfg["holdout_tolerance"]))
        & (candidates["valid_bin_min_drop_current_best"] <= float(blend_cfg["valid_bin_min_tolerance"]))
    )
    candidates_sorted = candidates.sort_values(
        ["holdout_pearson", "valid_bin_min_pearson", "signal_submission_corr_current_best"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    candidates_sorted.to_csv(run_dir / "candidate_metrics.csv", index=False)

    selected_rows: list[pd.Series] = []
    selected_names: set[str] = set()
    preferred_family_order = ["residual", "spearman", "stable"]
    eligible = candidates_sorted[candidates_sorted["eligible"]].copy()
    for family in preferred_family_order:
        pool = eligible[(eligible["family"] == family) & (~eligible["candidate"].isin(selected_names))]
        if pool.empty:
            continue
        selected = pool.iloc[0].copy()
        selected["selection_reason"] = f"{family}_best_eligible"
        selected_rows.append(selected)
        selected_names.add(str(selected["candidate"]))
        if len(selected_rows) >= int(blend_cfg["max_selected_submissions"]):
            break

    if len(selected_rows) < int(blend_cfg["max_selected_submissions"]):
        pool = eligible[~eligible["candidate"].isin(selected_names)]
        for _, selected in pool.iterrows():
            selected = selected.copy()
            selected["selection_reason"] = "overall_best_eligible"
            selected_rows.append(selected)
            selected_names.add(str(selected["candidate"]))
            if len(selected_rows) >= int(blend_cfg["max_selected_submissions"]):
                break

    selected_df = pd.DataFrame(selected_rows)
    if selected_df.empty:
        selected_df = pd.DataFrame(columns=[*candidates.columns, "selection_reason", "path"])
    else:
        selected_df["path"] = ""
        for idx, row in selected_df.iterrows():
            name = str(row["candidate"])
            submission = make_submission(sample_submission, test_df, candidate_predictions[name], prediction_col, id_col)
            path = submission_dir / f"submission_{name}.csv"
            submission.to_csv(path, index=False)
            selected_df.at[idx, "path"] = str(path.relative_to(ROOT))
        default_path = submission_dir / "submission_best.csv"
        shutil.copyfile(ROOT / str(selected_df.iloc[0]["path"]), default_path)

    selected_df.to_csv(run_dir / "selected_candidates.csv", index=False)

    write_json(
        run_dir / "final_summary.json",
        {
            "current_best_name": current_cfg["name"],
            "current_best_private_score": current_cfg["private_score"],
            "current_best_holdout_pearson": current_best_holdout,
            "current_best_submission_path": current_cfg["submission_path"],
            "current_best_valid_bin_summary": current_best_bins,
            "signal_count": len(signal_metrics),
            "candidate_count": len(candidates),
            "eligible_candidate_count": int(candidates["eligible"].sum()),
            "selected_candidates": selected_df["candidate"].tolist() if "candidate" in selected_df else [],
            "ranking_families": config["ranking_families"],
            "ridge_alphas": ridge_alphas,
            "leakage_control": "rankings and preprocessing are fitted on the 80% train split only; validation target is only used for model/weight evaluation.",
        },
    )

    print("\n=== Current best ===")
    print(json.dumps({"holdout": current_best_holdout, **current_best_bins}, indent=2))
    print("\n=== Signal metrics ===")
    print(signal_metrics.to_string(index=False))
    print("\n=== Top candidate metrics ===")
    print(candidates_sorted.head(30).to_string(index=False))
    print("\n=== Selected candidates ===")
    print(selected_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
