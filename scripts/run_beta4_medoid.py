from __future__ import annotations

import argparse
import gc
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

from drw_crypto.feature_selection import medoid_feature_clusters  # noqa: E402
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
    parser = argparse.ArgumentParser(description="Run Beta4-A correlation-cluster medoid Ridge experiments.")
    parser.add_argument("--config", default="configs/01_main_beta4_medoid.yaml", help="Path to YAML config.")
    parser.add_argument("--max-train-rows", type=int, default=None, help="Optional tail sample for smoke runs.")
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def summarize_delta(pred: np.ndarray, reference: np.ndarray, prefix: str) -> dict[str, float]:
    delta = pred - reference
    return {
        f"{prefix}_corr_current_best": pearson_corr(reference, pred),
        f"{prefix}_delta_std_current_best": float(np.std(delta)),
        f"{prefix}_delta_mae_current_best": float(np.mean(np.abs(delta))),
    }


def main() -> int:
    args = parse_args()
    config_path = ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)
    config = load_config(config_path)

    data_cfg = config["data"]
    prep_cfg = config["preprocessing"]
    current_cfg = config["current_best"]
    medoid_cfg = config["medoid"]
    ridge_cfg = config["ridge"]
    blend_cfg = config["blend"]
    output_cfg = config["output"]

    run_dir = ensure_dir(ROOT / output_cfg["run_dir"])
    cluster_dir = ensure_dir(run_dir / "clusters")
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
    if target_col in feature_cols or (id_col and id_col in feature_cols):
        raise ValueError("Target or ID column leaked into feature columns.")
    prediction_col = choose_prediction_column(sample_submission, output_cfg.get("prediction_col"), target_col)

    train_part, valid_part = split_time_ordered(train_df, float(data_cfg["validation_fraction"]))
    y_train = train_part[target_col].to_numpy(dtype=np.float64)
    y_valid = valid_part[target_col].to_numpy(dtype=np.float64)
    y_full = train_df[target_col].to_numpy(dtype=np.float64)

    current_best_valid = weighted_ridge_prediction(
        train_part,
        y_train,
        valid_part,
        current_cfg["components"],
        prep_cfg,
        ROOT,
    )
    current_best_test = weighted_ridge_prediction(
        train_df,
        y_full,
        test_df,
        current_cfg["components"],
        prep_cfg,
        ROOT,
    )
    current_best_holdout = pearson_corr(y_valid, current_best_valid)

    write_json(
        run_dir / "data_summary.json",
        {
            "train_path": str(train_path.relative_to(ROOT)),
            "test_path": str(test_path.relative_to(ROOT)),
            "sample_submission_path": str(sample_path.relative_to(ROOT)) if sample_path else None,
            "train_shape": list(train_df.shape),
            "test_shape": list(test_df.shape),
            "target_col": target_col,
            "id_col": id_col,
            "prediction_col": prediction_col,
            "feature_count": len(feature_cols),
            "validation_fraction": data_cfg["validation_fraction"],
            "max_train_rows": args.max_train_rows,
            "current_best_holdout_pearson": current_best_holdout,
            "current_best_private_score": current_cfg.get("private_score"),
        },
    )
    write_json(run_dir / "resolved_config.json", config)

    metrics_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    candidate_paths: dict[str, Path] = {}

    for threshold in [float(value) for value in medoid_cfg["thresholds"]]:
        threshold_name = f"t{safe_name(threshold)}"
        print(f"\n=== Beta4-A medoid threshold {threshold:g} ===")
        clusters, selected_features = medoid_feature_clusters(
            train_part,
            feature_cols,
            target_col,
            threshold=threshold,
            low_target_corr_threshold=float(medoid_cfg.get("low_target_corr_threshold", 1e-4)),
            linkage_method=str(medoid_cfg.get("linkage_method", "average")),
        )
        clusters.to_csv(cluster_dir / f"feature_clusters_{threshold_name}.csv", index=False)
        write_selected_features(run_dir / f"selected_features_medoid_{threshold_name}.txt", selected_features)

        preprocessor = TabularPreprocessor(selected_features).fit(
            train_part,
            missing_fill=str(prep_cfg.get("missing_fill", "median")),
            standardize=bool(prep_cfg.get("standardize", True)),
        )
        x_train = preprocessor.transform(train_part)
        x_valid = preprocessor.transform(valid_part)
        ridge_best, valid_pred, ridge_grid = train_ridge_search(
            x_train,
            y_train,
            x_valid,
            y_valid,
            [float(alpha) for alpha in ridge_cfg["alphas"]],
        )
        ridge_grid.insert(0, "threshold", threshold)
        ridge_grid.insert(1, "feature_count", len(selected_features))
        ridge_grid.to_csv(run_dir / f"ridge_alpha_search_medoid_{threshold_name}.csv", index=False)
        save_valid_prediction(pred_dir / f"valid_medoid_{threshold_name}_ridge.csv", y_valid, valid_pred)

        signal_holdout = pearson_corr(y_valid, valid_pred)
        metrics_rows.append(
            {
                "threshold": threshold,
                "scheme": f"medoid_{threshold_name}",
                "feature_count": len(selected_features),
                "model": "ridge",
                "alpha": float(ridge_best["alpha"]),
                "pearson": signal_holdout,
                "rmse": rmse(y_valid, valid_pred),
                "current_best_holdout_pearson": current_best_holdout,
                "holdout_delta_current_best": signal_holdout - current_best_holdout,
            }
        )

        final_preprocessor = TabularPreprocessor(selected_features).fit(
            train_df,
            missing_fill=str(prep_cfg.get("missing_fill", "median")),
            standardize=bool(prep_cfg.get("standardize", True)),
        )
        x_full = final_preprocessor.transform(train_df)
        x_test = final_preprocessor.transform(test_df)
        test_pred = fit_ridge(float(ridge_best["alpha"]), x_full, y_full, x_test)
        signal_submission = make_submission(sample_submission, test_df, test_pred, prediction_col, id_col)
        signal_path = signal_submission_dir / f"submission_medoid_{threshold_name}_ridge.csv"
        signal_submission.to_csv(signal_path, index=False)

        for signal_weight in [float(value) for value in blend_cfg["signal_weights"]]:
            base_weight = 1.0 - signal_weight
            blend_valid = base_weight * current_best_valid + signal_weight * valid_pred
            blend_test = base_weight * current_best_test + signal_weight * test_pred
            candidate = f"blend_current_w{safe_name(base_weight)}_medoid_{threshold_name}_w{safe_name(signal_weight)}"
            submission = make_submission(sample_submission, test_df, blend_test, prediction_col, id_col)
            path = submission_dir / f"submission_{candidate}.csv"
            submission.to_csv(path, index=False)
            candidate_paths[candidate] = path
            candidate_rows.append(
                {
                    "candidate": candidate,
                    "threshold": threshold,
                    "feature_count": len(selected_features),
                    "model": "ridge",
                    "ridge_alpha": float(ridge_best["alpha"]),
                    "current_best_weight": base_weight,
                    "signal_weight": signal_weight,
                    "holdout_pearson": pearson_corr(y_valid, blend_valid),
                    "holdout_rmse": rmse(y_valid, blend_valid),
                    "holdout_delta_current_best": pearson_corr(y_valid, blend_valid) - current_best_holdout,
                    "valid_prediction_std": float(np.std(blend_valid)),
                    "submission_prediction_std": float(np.std(blend_test)),
                    "std_ratio_current_best": float(np.std(blend_test) / np.std(current_best_test)),
                    **summarize_delta(blend_valid, current_best_valid, "valid"),
                    **summarize_delta(blend_test, current_best_test, "submission"),
                    "path": str(path.relative_to(ROOT)),
                }
            )

        pd.DataFrame(metrics_rows).sort_values("pearson", ascending=False).to_csv(
            run_dir / "metrics_medoid_ridge.csv",
            index=False,
        )
        del preprocessor, final_preprocessor, x_train, x_valid, x_full, x_test, valid_pred, test_pred
        gc.collect()

    candidates = pd.DataFrame(candidate_rows)
    candidates["eligible"] = (
        (candidates["holdout_delta_current_best"] >= -float(blend_cfg.get("holdout_tolerance", 0.003)))
        & (candidates["std_ratio_current_best"] <= float(blend_cfg.get("max_std_ratio_current_best", 1.15)))
    )
    candidates = candidates.sort_values(
        ["eligible", "holdout_pearson", "submission_corr_current_best"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    candidates.to_csv(run_dir / "candidate_metrics.csv", index=False)

    max_selected = int(blend_cfg.get("max_selected_submissions", 3))
    selected = candidates[candidates["eligible"]].head(max_selected).copy()
    if selected.empty:
        selected = candidates.head(max_selected).copy()
    selected.to_csv(run_dir / "selected_candidates.csv", index=False)
    if not selected.empty:
        best_path = ROOT / str(selected.iloc[0]["path"])
        shutil.copyfile(best_path, submission_dir / "submission_best.csv")

    summary = {
        "current_best": current_cfg,
        "current_best_holdout_pearson": current_best_holdout,
        "best_metric_row": pd.DataFrame(metrics_rows).sort_values("pearson", ascending=False).iloc[0].to_dict(),
        "best_candidate": selected.iloc[0].to_dict() if not selected.empty else None,
        "selected_candidates": selected["candidate"].tolist(),
    }
    write_json(run_dir / "final_summary.json", summary)

    print("\n=== Medoid Ridge Metrics ===")
    print(pd.DataFrame(metrics_rows).sort_values("pearson", ascending=False).to_string(index=False))
    print("\n=== Selected Candidates ===")
    print(selected.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
