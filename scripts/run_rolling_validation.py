from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drw_crypto.feature_selection import pearson_feature_ranking  # noqa: E402
from drw_crypto.io import TRAIN_CANDIDATES, ensure_dir, find_first_existing, read_table, write_json, write_lines  # noqa: E402
from drw_crypto.metrics import pearson_corr, rmse  # noqa: E402
from drw_crypto.models import NumpyRidgeRegressor  # noqa: E402
from drw_crypto.preprocessing import TabularPreprocessor, infer_feature_columns, infer_id_column, infer_target_column  # noqa: E402
from drw_crypto.validation import rolling_time_window_indices  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run rolling validation for temporal stability analysis.")
    parser.add_argument("--config", default="configs/03_temporal_rolling_validation.yaml", help="Path to YAML config.")
    parser.add_argument("--max-train-rows", type=int, default=None, help="Optional tail sample for smoke runs.")
    parser.add_argument("--save-all-predictions", action="store_true", help="Save all model validation predictions.")
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def describe_array(values: np.ndarray, prefix: str) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_q05": float(np.quantile(values, 0.05)),
        f"{prefix}_q25": float(np.quantile(values, 0.25)),
        f"{prefix}_q50": float(np.quantile(values, 0.50)),
        f"{prefix}_q75": float(np.quantile(values, 0.75)),
        f"{prefix}_q95": float(np.quantile(values, 0.95)),
    }


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


def train_lightgbm(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    feature_cols: list[str],
    params: dict[str, Any],
) -> tuple[dict[str, Any], np.ndarray, dict[str, Any], pd.DataFrame]:
    import lightgbm as lgb

    def lgb_pearson_eval(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[str, float, bool]:
        return "pearson", pearson_corr(y_true, y_pred), True

    model_params = {
        "n_estimators": int(params.get("n_estimators", 1200)),
        "learning_rate": float(params.get("learning_rate", 0.03)),
        "num_leaves": int(params.get("num_leaves", 63)),
        "subsample": float(params.get("subsample", 0.85)),
        "colsample_bytree": float(params.get("colsample_bytree", 0.85)),
        "reg_alpha": float(params.get("reg_alpha", 0.0)),
        "reg_lambda": float(params.get("reg_lambda", 1.0)),
        "objective": "regression",
        "metric": "None",
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": -1,
    }
    callbacks = [
        lgb.early_stopping(int(params.get("early_stopping_rounds", 100)), verbose=False),
        lgb.log_evaluation(period=100),
    ]

    start = time.perf_counter()
    model = lgb.LGBMRegressor(**model_params)
    model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], eval_metric=lgb_pearson_eval, callbacks=callbacks)
    pred = model.predict(x_valid)
    elapsed = time.perf_counter() - start
    best_iter = getattr(model, "best_iteration_", None)
    final_params = dict(model_params)
    if best_iter:
        final_params["n_estimators"] = int(best_iter)

    importance = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": model.feature_importances_.astype(float),
        }
    ).sort_values(["importance", "feature"], ascending=[False, True])

    row = {
        "pearson": pearson_corr(y_valid, pred),
        "rmse": rmse(y_valid, pred),
        "train_seconds": elapsed,
        "best_iteration": best_iter,
    }
    return row, pred, final_params, importance


def save_prediction(path: Path, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "row_index": np.arange(len(y_true), dtype=np.int64),
            "y_true": y_true,
            "prediction": y_pred,
        }
    ).to_csv(path, index=False)


def add_metric_row(
    rows: list[dict[str, Any]],
    fold: dict[str, Any],
    scheme: str,
    feature_count: int,
    model: str,
    y_valid: np.ndarray,
    pred: np.ndarray,
    train_seconds: float,
    best_iteration: Any,
    notes: str,
) -> None:
    row = {
        "fold": fold["name"],
        "train_start": fold["train_start"],
        "train_end": fold["train_end"],
        "valid_start": fold["valid_start"],
        "valid_end": fold["valid_end"],
        "scheme": scheme,
        "feature_count": feature_count,
        "model": model,
        "pearson": pearson_corr(y_valid, pred),
        "rmse": rmse(y_valid, pred),
        "train_seconds": train_seconds,
        "best_iteration": best_iteration if best_iteration is not None else "",
        "prediction_std": float(np.std(pred)),
        "notes": notes,
    }
    row.update(describe_array(y_valid, "target"))
    row.update(describe_array(pred, "prediction"))
    rows.append(row)


def plot_rolling_pearson(metrics: pd.DataFrame, figure_dir: Path) -> None:
    ensure_dir(figure_dir)
    ensemble = metrics[metrics["model"] == "ensemble_ridge_lgbm"].copy()
    if ensemble.empty:
        return
    pivot = ensemble.pivot(index="fold", columns="scheme", values="pearson")
    ordered_folds = metrics["fold"].drop_duplicates().tolist()
    pivot = pivot.reindex(ordered_folds)

    plt.figure(figsize=(10, 5))
    for scheme in pivot.columns:
        plt.plot(pivot.index, pivot[scheme], marker="o", label=scheme)
    plt.axhline(0.06610, color="gray", linestyle="--", linewidth=1, label="private score 0.06610")
    plt.title("Rolling Validation Pearson by Feature Scheme")
    plt.xlabel("Fold")
    plt.ylabel("Pearson")
    plt.xticks(rotation=20)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(figure_dir / "rolling_pearson_by_scheme.png", dpi=160)
    plt.close()


def plot_target_distribution(fold_stats: pd.DataFrame, figure_dir: Path) -> None:
    ensure_dir(figure_dir)
    plt.figure(figsize=(8, 4))
    plt.errorbar(
        fold_stats["fold"],
        fold_stats["target_mean"],
        yerr=fold_stats["target_std"],
        marker="o",
        capsize=4,
    )
    plt.title("Validation Target Mean and Std by Fold")
    plt.xlabel("Fold")
    plt.ylabel("Target")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(figure_dir / "target_distribution_by_fold.png", dpi=160)
    plt.close()


def summarize_scheme_stability(metrics: pd.DataFrame) -> pd.DataFrame:
    ensemble = metrics[metrics["model"] == "ensemble_ridge_lgbm"]
    rows: list[dict[str, Any]] = []
    for scheme, group in ensemble.groupby("scheme"):
        rows.append(
            {
                "scheme": scheme,
                "fold_count": int(group["fold"].nunique()),
                "pearson_mean": float(group["pearson"].mean()),
                "pearson_std": float(group["pearson"].std(ddof=0)),
                "pearson_min": float(group["pearson"].min()),
                "pearson_max": float(group["pearson"].max()),
                "rmse_mean": float(group["rmse"].mean()),
                "private_gap_abs": float(abs(group["pearson"].mean() - 0.06610)),
            }
        )
    return pd.DataFrame(rows).sort_values(["private_gap_abs", "pearson_mean"], ascending=[True, False])


def summarize_lightgbm_importance(importance_dir: Path, fold_count_by_scheme: dict[str, int]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(importance_dir.glob("lightgbm_importance_*.csv")):
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        frame["rank"] = frame.groupby(["fold", "scheme"])["importance"].rank(method="first", ascending=False)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()

    importance = pd.concat(frames, ignore_index=True)
    grouped = importance.groupby(["scheme", "feature"], as_index=False).agg(
        fold_count=("fold", "nunique"),
        nonzero_fold_count=("importance", lambda values: int((values > 0).sum())),
        importance_mean=("importance", "mean"),
        importance_std=("importance", lambda values: float(values.std(ddof=0))),
        importance_max=("importance", "max"),
        mean_rank=("rank", "mean"),
        best_rank=("rank", "min"),
    )
    grouped["fold_coverage"] = grouped.apply(
        lambda row: row["fold_count"] / max(1, fold_count_by_scheme.get(str(row["scheme"]), 1)),
        axis=1,
    )
    return grouped.sort_values(
        ["scheme", "fold_coverage", "nonzero_fold_count", "importance_mean", "feature"],
        ascending=[True, False, False, False, True],
    )


def main() -> int:
    args = parse_args()
    config_path = ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)
    config = load_config(config_path)

    data_cfg = config["data"]
    prep_cfg = config["preprocessing"]
    feature_cfg = config["feature_selection"]
    model_cfg = config["models"]
    fold_cfg = config["folds"]
    output_cfg = config["output"]

    run_dir = ensure_dir(ROOT / output_cfg["run_dir"])
    figure_dir = ensure_dir(ROOT / output_cfg["figure_dir"])
    ranking_dir = ensure_dir(run_dir / "feature_rankings")
    selected_dir = ensure_dir(run_dir / "selected_features")
    pred_dir = ensure_dir(run_dir / "valid_predictions")
    importance_dir = ensure_dir(run_dir / "lightgbm_importance")

    raw_dir = ROOT / data_cfg["raw_dir"]
    train_path = find_first_existing(raw_dir, data_cfg.get("train_file"), TRAIN_CANDIDATES)
    if train_path is None:
        raise FileNotFoundError("Missing train file under data/raw.")

    train_df = read_table(train_path)
    max_train_rows = args.max_train_rows
    if max_train_rows and max_train_rows < len(train_df):
        train_df = train_df.tail(int(max_train_rows)).reset_index(drop=True)

    target_col = infer_target_column(train_df, None, data_cfg.get("target_col"))
    id_col = infer_id_column(train_df, None, None, target_col, data_cfg.get("id_col"))
    feature_cols = infer_feature_columns(
        train_df,
        target_col=target_col,
        id_col=id_col,
        drop_columns=data_cfg.get("drop_columns") or [],
        numeric_only=bool(prep_cfg.get("numeric_only", True)),
    )
    top_k_values = [int(k) for k in feature_cfg.get("top_k", [])]

    write_json(
        run_dir / "data_summary.json",
        {
            "train_path": str(train_path.relative_to(ROOT)),
            "train_shape": list(train_df.shape),
            "target_col": target_col,
            "feature_count": len(feature_cols),
            "top_k": top_k_values,
            "include_full": bool(feature_cfg.get("include_full", True)),
            "fold_count": len(fold_cfg["definitions"]),
            "embargo_rows": int(fold_cfg.get("embargo_rows", 0)),
            "max_train_rows": max_train_rows,
        },
    )
    write_json(run_dir / "resolved_config.json", config)

    metrics_rows: list[dict[str, Any]] = []
    fold_stats_rows: list[dict[str, Any]] = []
    selected_feature_sets: dict[tuple[str, str], set[str]] = {}

    for fold in fold_cfg["definitions"]:
        print(f"\n=== Rolling fold: {fold['name']} ===")
        train_indices, valid_indices = rolling_time_window_indices(
            len(train_df),
            train_start=float(fold["train_start"]),
            train_end=float(fold["train_end"]),
            valid_start=float(fold["valid_start"]),
            valid_end=float(fold["valid_end"]),
            embargo_rows=fold_cfg.get("embargo_rows", 0),
        )
        train_part = train_df.iloc[train_indices]
        valid_part = train_df.iloc[valid_indices]

        y_train = train_part[target_col].to_numpy(dtype=np.float64)
        y_valid = valid_part[target_col].to_numpy(dtype=np.float64)
        fold_stats = {
            "fold": fold["name"],
            "train_rows": len(train_part),
            "valid_rows": len(valid_part),
        }
        fold_stats.update(describe_array(y_train, "train_target"))
        fold_stats.update(describe_array(y_valid, "target"))
        fold_stats_rows.append(fold_stats)

        ranking = pearson_feature_ranking(train_part, feature_cols, target_col)
        ranking.to_csv(ranking_dir / f"pearson_feature_ranking_{fold['name']}.csv", index=False)

        schemes: list[tuple[str, list[str]]] = []
        for k in top_k_values:
            selected = ranking["feature"].head(min(k, len(feature_cols))).tolist()
            scheme = f"top{k}"
            write_lines(selected_dir / f"selected_features_{fold['name']}_{scheme}.txt", selected)
            selected_feature_sets[(fold["name"], scheme)] = set(selected)
            schemes.append((scheme, selected))
        if bool(feature_cfg.get("include_full", True)):
            write_lines(selected_dir / f"selected_features_{fold['name']}_full.txt", feature_cols)
            selected_feature_sets[(fold["name"], "full")] = set(feature_cols)
            schemes.append(("full", feature_cols))

        for scheme, selected_features in schemes:
            print(f"--- Scheme {scheme}: {len(selected_features)} features")
            preprocessor = TabularPreprocessor(selected_features).fit(
                train_part,
                missing_fill=str(prep_cfg.get("missing_fill", "median")),
                standardize=bool(prep_cfg.get("standardize", True)),
            )
            x_train = preprocessor.transform(train_part)
            x_valid = preprocessor.transform(valid_part)

            ridge_best, ridge_pred, ridge_grid = train_ridge_search(
                x_train,
                y_train,
                x_valid,
                y_valid,
                [float(alpha) for alpha in model_cfg["ridge"].get("alphas", [1.0])],
            )
            ridge_grid.insert(0, "scheme", scheme)
            ridge_grid.insert(0, "fold", fold["name"])
            ridge_grid.to_csv(run_dir / f"ridge_alpha_search_{fold['name']}_{scheme}.csv", index=False)
            add_metric_row(
                metrics_rows,
                fold,
                scheme,
                len(selected_features),
                "ridge",
                y_valid,
                ridge_pred,
                ridge_best["train_seconds"],
                "",
                f"alpha={ridge_best['alpha']}",
            )

            lgbm_row, lgbm_pred, lgbm_params, importance = train_lightgbm(
                x_train,
                y_train,
                x_valid,
                y_valid,
                selected_features,
                model_cfg["lightgbm"],
            )
            importance.insert(0, "scheme", scheme)
            importance.insert(0, "fold", fold["name"])
            importance.to_csv(importance_dir / f"lightgbm_importance_{fold['name']}_{scheme}.csv", index=False)
            add_metric_row(
                metrics_rows,
                fold,
                scheme,
                len(selected_features),
                "lightgbm",
                y_valid,
                lgbm_pred,
                lgbm_row["train_seconds"],
                lgbm_row["best_iteration"],
                json.dumps(lgbm_params, sort_keys=True),
            )

            step = float(config["ensemble"].get("ridge_lgbm_grid_step", 0.05))
            best_weight = 0.0
            best_score = -np.inf
            best_pred: np.ndarray | None = None
            grid_rows: list[dict[str, Any]] = []
            for weight in np.arange(0.0, 1.0 + step / 2.0, step):
                pred = weight * ridge_pred + (1.0 - weight) * lgbm_pred
                score = pearson_corr(y_valid, pred)
                grid_rows.append(
                    {
                        "fold": fold["name"],
                        "scheme": scheme,
                        "ridge_weight": float(weight),
                        "pearson": score,
                        "rmse": rmse(y_valid, pred),
                    }
                )
                if score > best_score:
                    best_score = score
                    best_weight = float(weight)
                    best_pred = pred
            assert best_pred is not None
            pd.DataFrame(grid_rows).to_csv(run_dir / f"ensemble_ridge_lgbm_grid_{fold['name']}_{scheme}.csv", index=False)
            add_metric_row(
                metrics_rows,
                fold,
                scheme,
                len(selected_features),
                "ensemble_ridge_lgbm",
                y_valid,
                best_pred,
                0.0,
                "",
                f"ridge_weight={best_weight:.4f}",
            )

            if args.save_all_predictions or scheme in {"full", "top100", "top200"}:
                save_prediction(pred_dir / f"valid_{fold['name']}_{scheme}_ridge.csv", y_valid, ridge_pred)
                save_prediction(pred_dir / f"valid_{fold['name']}_{scheme}_lightgbm.csv", y_valid, lgbm_pred)
                save_prediction(pred_dir / f"valid_{fold['name']}_{scheme}_ensemble_ridge_lgbm.csv", y_valid, best_pred)

            metrics_df = pd.DataFrame(metrics_rows)
            metrics_df.to_csv(run_dir / "metrics_rolling_validation.csv", index=False)
            del preprocessor, x_train, x_valid, ridge_pred, lgbm_pred, best_pred
            gc.collect()

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(run_dir / "metrics_rolling_validation.csv", index=False)
    fold_stats_df = pd.DataFrame(fold_stats_rows)
    fold_stats_df.to_csv(run_dir / "target_distribution_by_fold.csv", index=False)

    stability = summarize_scheme_stability(metrics_df)
    stability.to_csv(run_dir / "scheme_stability_summary.csv", index=False)

    overlap_rows: list[dict[str, Any]] = []
    fold_names = [fold["name"] for fold in fold_cfg["definitions"]]
    schemes = sorted({scheme for _, scheme in selected_feature_sets if scheme != "full"})
    for scheme in schemes:
        for left, right in combinations(fold_names, 2):
            left_set = selected_feature_sets[(left, scheme)]
            right_set = selected_feature_sets[(right, scheme)]
            overlap_rows.append(
                {
                    "scheme": scheme,
                    "fold_left": left,
                    "fold_right": right,
                    "intersection": len(left_set & right_set),
                    "union": len(left_set | right_set),
                    "jaccard": len(left_set & right_set) / max(1, len(left_set | right_set)),
                    "overlap_ratio_of_k": len(left_set & right_set) / max(1, len(left_set)),
                }
            )
    pd.DataFrame(overlap_rows).to_csv(run_dir / "feature_overlap.csv", index=False)

    frequency_rows: list[dict[str, Any]] = []
    for scheme in schemes:
        counts: dict[str, int] = {}
        for fold_name in fold_names:
            for feature in selected_feature_sets[(fold_name, scheme)]:
                counts[feature] = counts.get(feature, 0) + 1
        for feature, count in counts.items():
            frequency_rows.append({"scheme": scheme, "feature": feature, "fold_count": count})
    pd.DataFrame(frequency_rows).sort_values(["scheme", "fold_count", "feature"], ascending=[True, False, True]).to_csv(
        run_dir / "feature_selection_frequency.csv",
        index=False,
    )

    fold_count_by_scheme = {
        scheme: len({fold_name for fold_name, selected_scheme in selected_feature_sets if selected_scheme == scheme})
        for scheme in {scheme for _, scheme in selected_feature_sets}
    }
    importance_stability = summarize_lightgbm_importance(importance_dir, fold_count_by_scheme)
    if not importance_stability.empty:
        importance_stability.to_csv(run_dir / "lightgbm_importance_stability.csv", index=False)

    plot_rolling_pearson(metrics_df, figure_dir)
    plot_target_distribution(fold_stats_df, figure_dir)

    write_json(
        run_dir / "final_summary.json",
        {
            "best_mean_stability_row": stability.iloc[0].to_dict() if not stability.empty else None,
            "best_mean_pearson_row": stability.sort_values("pearson_mean", ascending=False).iloc[0].to_dict()
            if not stability.empty
            else None,
            "private_score_reference": 0.06610,
            "public_score_reference": 0.02826,
            "figure_dir": str(figure_dir.relative_to(ROOT)),
            "lightgbm_importance_stability": str((run_dir / "lightgbm_importance_stability.csv").relative_to(ROOT))
            if not importance_stability.empty
            else None,
        },
    )

    print("\n=== Scheme stability summary ===")
    print(stability.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
