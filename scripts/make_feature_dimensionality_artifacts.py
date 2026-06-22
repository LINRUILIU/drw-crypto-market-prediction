from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drw_crypto.feature_selection import (  # noqa: E402
    absolute_feature_correlation,
    pearson_feature_ranking,
    spearman_feature_ranking,
)
from drw_crypto.io import TRAIN_CANDIDATES, ensure_dir, find_first_existing, read_table, write_json  # noqa: E402
from drw_crypto.metrics import pearson_corr, rmse  # noqa: E402
from drw_crypto.pipeline import safe_name, split_time_ordered, train_ridge_search  # noqa: E402
from drw_crypto.preprocessing import TabularPreprocessor, infer_feature_columns, infer_id_column, infer_target_column  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Extension 1 feature dimensionality artifacts.")
    parser.add_argument("--config", default="configs/02_feature_dimensionality.yaml", help="Path to YAML config.")
    parser.add_argument("--max-train-rows", type=int, default=None, help="Optional tail sample for smoke runs.")
    parser.add_argument("--run-dir", default=None, help="Override run output directory.")
    parser.add_argument("--figure-dir", default=None, help="Override figure output directory.")
    parser.add_argument("--skip-shap", action="store_true", help="Skip SHAP ranking and SHAP top-k evaluation.")
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = ROOT / path if not Path(path).is_absolute() else Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def has_module(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def savefig(figure_dir: Path, name: str) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(figure_dir / name, dpi=180, bbox_inches="tight")
    plt.close()


def feature_matrix(train_part: pd.DataFrame, valid_part: pd.DataFrame, features: list[str], prep_cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    preprocessor = TabularPreprocessor(features).fit(
        train_part,
        missing_fill=str(prep_cfg.get("missing_fill", "median")),
        standardize=bool(prep_cfg.get("standardize", True)),
    )
    return preprocessor.transform(train_part), preprocessor.transform(valid_part)


def evaluate_ridge_scheme(
    *,
    method: str,
    scheme: str,
    feature_count: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    alphas: list[float],
    run_dir: Path,
    extra: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    best, _pred, grid = train_ridge_search(x_train, y_train, x_valid, y_valid, alphas)
    grid.insert(0, "method", method)
    grid.insert(1, "scheme", scheme)
    grid.insert(2, "feature_count", int(feature_count))
    grid.to_csv(run_dir / f"ridge_alpha_search_{method}_{scheme}.csv", index=False)
    row: dict[str, Any] = {
        "method": method,
        "scheme": scheme,
        "model": "ridge",
        "feature_count": int(feature_count),
        "best_alpha": float(best["alpha"]),
        "pearson": float(best["pearson"]),
        "rmse": float(best["rmse"]),
        "fit_seconds": float(best["train_seconds"]),
        "fit_scope": "outer_train_only",
        "selection_scope": "outer_train_only",
    }
    if extra:
        row.update(extra)
    return row, grid


def upper_triangle_values(corr: pd.DataFrame) -> np.ndarray:
    values = corr.to_numpy(dtype=np.float64, copy=False)
    mask = np.triu(np.ones(values.shape, dtype=bool), k=1)
    return values[mask]


def build_raw_feature_summary(
    train_part: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    run_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = train_part.loc[:, feature_cols].replace([np.inf, -np.inf], np.nan)
    summary = pd.DataFrame(
        {
            "feature": feature_cols,
            "missing_rate": frame.isna().mean(axis=0).reindex(feature_cols).to_numpy(dtype=np.float64),
            "mean": frame.mean(axis=0).reindex(feature_cols).to_numpy(dtype=np.float64),
            "std": frame.std(axis=0, ddof=0).reindex(feature_cols).to_numpy(dtype=np.float64),
            "variance": frame.var(axis=0, ddof=0).reindex(feature_cols).to_numpy(dtype=np.float64),
            "unique_count": [int(frame[col].nunique(dropna=True)) for col in feature_cols],
        }
    )
    pearson = pearson_feature_ranking(train_part, feature_cols, target_col)
    spearman = spearman_feature_ranking(train_part, feature_cols, target_col)
    target_corr = pearson.merge(spearman, on="feature", how="outer")
    target_corr["pearson_rank"] = target_corr["abs_pearson"].rank(method="first", ascending=False).astype(int)
    target_corr["spearman_rank"] = target_corr["abs_spearman"].rank(method="first", ascending=False).astype(int)
    target_corr = target_corr.sort_values(["pearson_rank", "feature"]).reset_index(drop=True)
    summary = summary.merge(target_corr, on="feature", how="left")
    summary.to_csv(run_dir / "raw_feature_summary.csv", index=False)
    target_corr.to_csv(run_dir / "feature_target_correlation.csv", index=False)
    return summary, pearson, spearman


def build_correlation_artifacts(
    train_part: pd.DataFrame,
    feature_cols: list[str],
    high_corr_thresholds: list[float],
    run_dir: Path,
    figure_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    corr = absolute_feature_correlation(train_part, feature_cols)
    upper = upper_triangle_values(corr)
    rows: list[dict[str, Any]] = []
    rows.append(
        {
            "metric": "pair_count",
            "value": int(len(upper)),
        }
    )
    for name, value in {
        "mean_abs_corr": float(np.mean(upper)),
        "median_abs_corr": float(np.median(upper)),
        "p90_abs_corr": float(np.quantile(upper, 0.90)),
        "p95_abs_corr": float(np.quantile(upper, 0.95)),
        "p99_abs_corr": float(np.quantile(upper, 0.99)),
        "max_abs_corr": float(np.max(upper)),
    }.items():
        rows.append({"metric": name, "value": value})
    for threshold in high_corr_thresholds:
        rows.append({"metric": f"pairs_abs_corr_ge_{threshold:g}", "value": int(np.sum(upper >= float(threshold)))})
    summary = pd.DataFrame(rows)
    summary.to_csv(run_dir / "feature_correlation_summary.csv", index=False)

    pair_rows = []
    for idx_i, feature_i in enumerate(feature_cols):
        values = corr.iloc[idx_i, idx_i + 1 :].to_numpy(dtype=np.float64, copy=False)
        for feature_j, value in zip(feature_cols[idx_i + 1 :], values):
            pair_rows.append({"feature_i": feature_i, "feature_j": feature_j, "abs_corr": float(value)})
    pair_corr = pd.DataFrame(pair_rows)
    pair_corr.to_csv(run_dir / "feature_pair_abs_correlation.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.hist(upper, bins=50, color="#2563eb", alpha=0.85)
    ax.set_xlabel("Absolute feature-feature Pearson correlation")
    ax.set_ylabel("Pair count")
    ax.set_title("Anonymous Feature Redundancy Distribution")
    ax.grid(axis="y", alpha=0.25)
    savefig(figure_dir, "correlation_distribution.png")
    return corr, summary, upper


def cluster_mean_matrices(
    train_part: pd.DataFrame,
    valid_part: pd.DataFrame,
    feature_cols: list[str],
    clusters: pd.DataFrame,
    prep_cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    ordered = clusters.loc[:, ["feature", "cluster_id"]].drop_duplicates().copy()
    cluster_ids = sorted(int(value) for value in ordered["cluster_id"].unique())
    x_train_all, x_valid_all = feature_matrix(train_part, valid_part, feature_cols, prep_cfg)
    feature_to_pos = {feature: idx for idx, feature in enumerate(feature_cols)}
    x_train_parts: list[np.ndarray] = []
    x_valid_parts: list[np.ndarray] = []
    manifest_rows: list[dict[str, Any]] = []
    for cluster_id in cluster_ids:
        members = ordered.loc[ordered["cluster_id"] == cluster_id, "feature"].astype(str).tolist()
        positions = [feature_to_pos[feature] for feature in members]
        x_train_parts.append(x_train_all[:, positions].mean(axis=1, keepdims=True))
        x_valid_parts.append(x_valid_all[:, positions].mean(axis=1, keepdims=True))
        manifest_rows.append({"cluster_id": cluster_id, "feature_count": len(members), "features": ";".join(members)})
    return np.hstack(x_train_parts), np.hstack(x_valid_parts), pd.DataFrame(manifest_rows)


def medoid_feature_clusters_from_corr(
    corr: pd.DataFrame,
    pearson_ranking_df: pd.DataFrame,
    threshold: float,
    low_target_corr_threshold: float,
    linkage_method: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Cluster features from a precomputed absolute correlation matrix."""
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be between 0 and 1.")

    feature_cols = corr.index.astype(str).tolist()
    if len(feature_cols) == 1:
        return pd.DataFrame(
            [
                {
                    "cluster_id": 1,
                    "threshold": threshold,
                    "feature": feature_cols[0],
                    "cluster_size": 1,
                    "medoid_feature": feature_cols[0],
                    "is_medoid": True,
                    "medoid_score": 0.0,
                    "target_pearson": 0.0,
                    "abs_target_pearson": 0.0,
                    "selected_after_target_filter": False,
                    "cluster_medoid_score": 0.0,
                    "cluster_medoid_abs_target_pearson": 0.0,
                }
            ]
        ), []

    distance = 1.0 - corr.to_numpy(dtype=np.float64, copy=True)
    np.fill_diagonal(distance, 0.0)
    tree = linkage(squareform(distance, checks=False), method=linkage_method)
    labels = fcluster(tree, t=1.0 - float(threshold), criterion="distance")

    target_corr = pearson_ranking_df.set_index("feature")[["pearson", "abs_pearson"]]
    rows: list[dict[str, float | int | str | bool]] = []
    selected: list[tuple[str, float, int]] = []

    for cluster_id in sorted(set(int(label) for label in labels)):
        members = [feature for feature, label in zip(feature_cols, labels) if int(label) == cluster_id]
        sub_corr = corr.loc[members, members]
        medoid_scores = sub_corr.sum(axis=1) - 1.0
        candidates = pd.DataFrame(
            {
                "feature": members,
                "medoid_score": [float(medoid_scores.loc[feature]) for feature in members],
                "abs_target_pearson": [float(target_corr.loc[feature, "abs_pearson"]) for feature in members],
            }
        ).sort_values(["medoid_score", "abs_target_pearson", "feature"], ascending=[False, False, True])
        medoid = str(candidates.iloc[0]["feature"])
        medoid_score = float(candidates.iloc[0]["medoid_score"])
        medoid_abs_target = float(target_corr.loc[medoid, "abs_pearson"])
        keep_medoid = medoid_abs_target > float(low_target_corr_threshold)
        if keep_medoid:
            selected.append((medoid, medoid_abs_target, len(members)))

        for feature in members:
            abs_target = float(target_corr.loc[feature, "abs_pearson"])
            rows.append(
                {
                    "cluster_id": cluster_id,
                    "threshold": float(threshold),
                    "feature": feature,
                    "cluster_size": len(members),
                    "medoid_feature": medoid,
                    "is_medoid": feature == medoid,
                    "medoid_score": float(medoid_scores.loc[feature]),
                    "target_pearson": float(target_corr.loc[feature, "pearson"]),
                    "abs_target_pearson": abs_target,
                    "selected_after_target_filter": feature == medoid and keep_medoid,
                    "cluster_medoid_score": medoid_score,
                    "cluster_medoid_abs_target_pearson": medoid_abs_target,
                }
            )

    clusters = pd.DataFrame(rows).sort_values(
        ["cluster_id", "is_medoid", "abs_target_pearson", "feature"],
        ascending=[True, False, False, True],
    )
    selected_features = [
        feature
        for feature, _, _ in sorted(selected, key=lambda item: (-item[1], -item[2], item[0]))
    ]
    if not selected_features:
        fallback = clusters[clusters["is_medoid"]].sort_values(
            ["abs_target_pearson", "cluster_size", "feature"],
            ascending=[False, False, True],
        )
        selected_features = fallback["feature"].astype(str).tolist()
    return clusters.reset_index(drop=True), selected_features


def train_lightgbm_ranker(
    train_part: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    prep_cfg: dict[str, Any],
    lgb_cfg: dict[str, Any],
    inner_validation_fraction: float,
    run_dir: Path,
) -> tuple[Any | None, pd.DataFrame | None, np.ndarray | None, pd.DataFrame | None, dict[str, Any]]:
    if not has_module("lightgbm"):
        return None, None, None, None, {"skipped": "lightgbm is not installed"}

    import lightgbm as lgb

    inner_train, inner_valid = split_time_ordered(train_part, inner_validation_fraction)
    y_inner_train = inner_train[target_col].to_numpy(dtype=np.float64)
    y_inner_valid = inner_valid[target_col].to_numpy(dtype=np.float64)
    preprocessor = TabularPreprocessor(feature_cols).fit(
        inner_train,
        missing_fill=str(prep_cfg.get("missing_fill", "median")),
        standardize=bool(prep_cfg.get("standardize", True)),
    )
    x_inner_train = preprocessor.transform(inner_train)
    x_inner_valid = preprocessor.transform(inner_valid)

    def lgb_pearson_eval(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[str, float, bool]:
        return "pearson", pearson_corr(y_true, y_pred), True

    model_params = {
        "n_estimators": int(lgb_cfg.get("n_estimators", 1200)),
        "learning_rate": float(lgb_cfg.get("learning_rate", 0.03)),
        "num_leaves": int(lgb_cfg.get("num_leaves", 63)),
        "subsample": float(lgb_cfg.get("subsample", 0.85)),
        "colsample_bytree": float(lgb_cfg.get("colsample_bytree", 0.85)),
        "reg_alpha": float(lgb_cfg.get("reg_alpha", 0.0)),
        "reg_lambda": float(lgb_cfg.get("reg_lambda", 1.0)),
        "objective": "regression",
        "metric": "None",
        "random_state": int(lgb_cfg.get("random_state", 42)),
        "n_jobs": -1,
        "verbosity": -1,
    }
    callbacks = [
        lgb.early_stopping(int(lgb_cfg.get("early_stopping_rounds", 100)), verbose=False),
        lgb.log_evaluation(period=0),
    ]
    start = time.perf_counter()
    model = lgb.LGBMRegressor(**model_params)
    model.fit(x_inner_train, y_inner_train, eval_set=[(x_inner_valid, y_inner_valid)], eval_metric=lgb_pearson_eval, callbacks=callbacks)
    elapsed = time.perf_counter() - start
    pred = model.predict(x_inner_valid)
    best_iter = getattr(model, "best_iteration_", None)
    ranking = pd.DataFrame(
        {
            "feature": feature_cols,
            "gain_importance": model.booster_.feature_importance(importance_type="gain"),
            "split_importance": model.booster_.feature_importance(importance_type="split"),
        }
    ).sort_values(["gain_importance", "split_importance", "feature"], ascending=[False, False, True])
    ranking["rank"] = np.arange(1, len(ranking) + 1, dtype=np.int64)
    ranking["fit_scope"] = "outer_train_inner_split_only"
    ranking.to_csv(run_dir / "lightgbm_importance_ranking.csv", index=False)
    summary = {
        "inner_train_rows": len(inner_train),
        "inner_valid_rows": len(inner_valid),
        "inner_valid_pearson": pearson_corr(y_inner_valid, pred),
        "inner_valid_rmse": rmse(y_inner_valid, pred),
        "fit_seconds": elapsed,
        "best_iteration": int(best_iter) if best_iter else None,
    }
    return model, ranking, x_inner_valid, inner_valid, summary


def build_shap_ranking(
    model: Any,
    x_inner_valid: np.ndarray,
    inner_valid: pd.DataFrame,
    feature_cols: list[str],
    sample_rows: int,
    run_dir: Path,
) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    if not has_module("shap"):
        return None, {"skipped": "shap is not installed"}
    import shap

    n_rows = x_inner_valid.shape[0]
    take = min(int(sample_rows), n_rows)
    rng = np.random.default_rng(42)
    if take < n_rows:
        sample_idx = np.sort(rng.choice(n_rows, size=take, replace=False))
    else:
        sample_idx = np.arange(n_rows, dtype=np.int64)
    x_sample = x_inner_valid[sample_idx]
    start = time.perf_counter()
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(x_sample, check_additivity=False)
    elapsed = time.perf_counter() - start
    if isinstance(values, list):
        values = values[0]
    values = np.asarray(values, dtype=np.float64)
    ranking = pd.DataFrame(
        {
            "feature": feature_cols,
            "mean_abs_shap": np.mean(np.abs(values), axis=0),
        }
    ).sort_values(["mean_abs_shap", "feature"], ascending=[False, True])
    ranking["rank"] = np.arange(1, len(ranking) + 1, dtype=np.int64)
    ranking["fit_scope"] = "outer_train_inner_valid_shap_sample_only"
    ranking["sample_rows"] = int(take)
    ranking.to_csv(run_dir / "shap_feature_ranking.csv", index=False)
    return ranking, {"sample_rows": int(take), "fit_seconds": elapsed, "inner_valid_rows": int(len(inner_valid))}


def evaluate_feature_list_schemes(
    method: str,
    ranking: pd.DataFrame,
    score_col: str,
    top_k_values: list[int],
    train_part: pd.DataFrame,
    valid_part: pd.DataFrame,
    y_train: np.ndarray,
    y_valid: np.ndarray,
    prep_cfg: dict[str, Any],
    alphas: list[float],
    run_dir: Path,
    extra: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ordered = ranking.sort_values([score_col, "feature"], ascending=[False, True])
    for top_k in top_k_values:
        selected = ordered["feature"].head(min(int(top_k), len(ordered))).astype(str).tolist()
        x_train, x_valid = feature_matrix(train_part, valid_part, selected, prep_cfg)
        row, _grid = evaluate_ridge_scheme(
            method=method,
            scheme=f"top{top_k}",
            feature_count=len(selected),
            x_train=x_train,
            y_train=y_train,
            x_valid=x_valid,
            y_valid=y_valid,
            alphas=alphas,
            run_dir=run_dir,
            extra=extra,
        )
        rows.append(row)
    return rows


def plot_dimension_vs_pearson(metrics: pd.DataFrame, figure_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for method, part in metrics.groupby("method"):
        if method not in {"pearson", "spearman", "pca", "lightgbm_importance", "shap"}:
            continue
        part = part.sort_values("feature_count")
        ax.plot(part["feature_count"], part["pearson"], marker="o", label=method)
    ax.set_xlabel("Feature count / component count")
    ax.set_ylabel("Holdout Pearson")
    ax.set_title("Dimensionality vs Predictive Preservation")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    savefig(figure_dir, "dimension_vs_pearson.png")


def plot_method_comparison(metrics: pd.DataFrame, figure_dir: Path) -> None:
    best = metrics.sort_values("pearson", ascending=False).groupby("method", as_index=False).head(1)
    best = best.sort_values("pearson", ascending=True)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.barh(best["method"], best["pearson"], color="#2563eb")
    ax.set_xlabel("Best holdout Pearson")
    ax.set_title("Best Scheme by Method")
    ax.grid(axis="x", alpha=0.25)
    savefig(figure_dir, "method_comparison_bar.png")


def plot_tradeoff(metrics: pd.DataFrame, figure_dir: Path, full_feature_count: int) -> None:
    plot_df = metrics.copy()
    plot_df["compression_ratio"] = 1.0 - plot_df["feature_count"].astype(float) / float(full_feature_count)
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    for method, part in plot_df.groupby("method"):
        ax.scatter(part["compression_ratio"], part["pearson"], label=method, s=42, alpha=0.85)
    ax.set_xlabel("Compression ratio vs full feature count")
    ax.set_ylabel("Holdout Pearson")
    ax.set_title("Compression-Prediction Tradeoff")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    savefig(figure_dir, "compression_tradeoff_frontier.png")


def plot_model_driven(metrics: pd.DataFrame, figure_dir: Path) -> None:
    part = metrics[metrics["method"].isin(["pearson", "lightgbm_importance", "shap"])].copy()
    if part.empty:
        return
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for method, method_df in part.groupby("method"):
        method_df = method_df.sort_values("feature_count")
        ax.plot(method_df["feature_count"], method_df["pearson"], marker="o", label=method)
    ax.set_xlabel("Top-k feature count")
    ax.set_ylabel("Holdout Pearson")
    ax.set_title("Correlation vs Model-Driven Feature Ranking")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    savefig(figure_dir, "model_driven_topk_comparison.png")


def plot_cluster_sizes(all_clusters: list[pd.DataFrame], figure_dir: Path) -> None:
    if not all_clusters:
        return
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for clusters in all_clusters:
        threshold = clusters["threshold"].iloc[0]
        sizes = clusters.loc[clusters["is_medoid"], "cluster_size"].astype(int)
        ax.hist(sizes, bins=30, alpha=0.55, label=f"threshold={threshold:g}")
    ax.set_xlabel("Cluster size")
    ax.set_ylabel("Cluster count")
    ax.set_title("Correlation Cluster Size Distribution")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    savefig(figure_dir, "cluster_size_distribution.png")


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    data_cfg = config["data"]
    prep_cfg = config["preprocessing"]
    feature_cfg = config["feature_selection"]
    pca_cfg = config["pca"]
    cluster_cfg = config["cluster"]
    model_cfg = config["models"]
    model_driven_cfg = config["model_driven"]
    output_cfg = config["output"]

    run_dir = ensure_dir(ROOT / (args.run_dir or output_cfg["run_dir"]))
    figure_dir = ensure_dir(ROOT / (args.figure_dir or output_cfg["figure_dir"]))

    raw_dir = ROOT / data_cfg["raw_dir"]
    train_path = find_first_existing(raw_dir, data_cfg.get("train_file"), TRAIN_CANDIDATES)
    if train_path is None:
        raise FileNotFoundError("Missing train file under data/raw.")

    train_df = read_table(train_path)
    max_train_rows = args.max_train_rows if args.max_train_rows is not None else data_cfg.get("max_train_rows")
    if max_train_rows and int(max_train_rows) < len(train_df):
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
    train_part, valid_part = split_time_ordered(train_df, float(data_cfg["validation_fraction"]))
    y_train = train_part[target_col].to_numpy(dtype=np.float64)
    y_valid = valid_part[target_col].to_numpy(dtype=np.float64)
    alphas = [float(value) for value in model_cfg["ridge"].get("alphas", [1.0])]
    top_k_values = [int(value) for value in feature_cfg.get("top_k", [])]

    write_json(
        run_dir / "data_summary.json",
        {
            "train_path": str(train_path.relative_to(ROOT)),
            "train_shape": list(train_df.shape),
            "target_col": target_col,
            "id_col": id_col,
            "feature_count": len(feature_cols),
            "validation_fraction": data_cfg["validation_fraction"],
            "outer_train_rows": len(train_part),
            "outer_holdout_rows": len(valid_part),
            "max_train_rows": max_train_rows,
            "no_submission_generated": True,
        },
    )
    write_json(run_dir / "resolved_config.json", config)

    raw_summary, pearson_ranking, spearman_ranking = build_raw_feature_summary(train_part, feature_cols, target_col, run_dir)
    _corr, corr_summary, _upper = build_correlation_artifacts(
        train_part,
        feature_cols,
        [float(value) for value in feature_cfg.get("high_corr_thresholds", [])],
        run_dir,
        figure_dir,
    )

    all_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    pca_rows: list[dict[str, Any]] = []
    cluster_rows: list[dict[str, Any]] = []

    print("Running full-feature Ridge baseline")
    x_train_full, x_valid_full = feature_matrix(train_part, valid_part, feature_cols, prep_cfg)
    full_row, _ = evaluate_ridge_scheme(
        method="full",
        scheme="all_features",
        feature_count=len(feature_cols),
        x_train=x_train_full,
        y_train=y_train,
        x_valid=x_valid_full,
        y_valid=y_valid,
        alphas=alphas,
        run_dir=run_dir,
    )
    all_rows.append(full_row)
    selection_rows.append(full_row)

    print("Running Pearson and Spearman top-k Ridge schemes")
    pearson_rows = evaluate_feature_list_schemes(
        "pearson",
        pearson_ranking.rename(columns={"abs_pearson": "score"}),
        "score",
        top_k_values,
        train_part,
        valid_part,
        y_train,
        y_valid,
        prep_cfg,
        alphas,
        run_dir,
    )
    spearman_rows = evaluate_feature_list_schemes(
        "spearman",
        spearman_ranking.rename(columns={"abs_spearman": "score"}),
        "score",
        top_k_values,
        train_part,
        valid_part,
        y_train,
        y_valid,
        prep_cfg,
        alphas,
        run_dir,
    )
    all_rows.extend(pearson_rows + spearman_rows)
    selection_rows.extend(pearson_rows + spearman_rows)

    print("Running PCA Ridge schemes")
    from sklearn.decomposition import PCA

    for dim in [int(value) for value in pca_cfg.get("dims", [])]:
        dim = min(dim, x_train_full.shape[1])
        start = time.perf_counter()
        pca = PCA(n_components=dim, svd_solver=str(pca_cfg.get("svd_solver", "randomized")), random_state=int(pca_cfg.get("random_state", 42)))
        x_train_pca = pca.fit_transform(x_train_full)
        x_valid_pca = pca.transform(x_valid_full)
        pca_fit_seconds = time.perf_counter() - start
        row, _ = evaluate_ridge_scheme(
            method="pca",
            scheme=f"pca{dim}",
            feature_count=dim,
            x_train=x_train_pca,
            y_train=y_train,
            x_valid=x_valid_pca,
            y_valid=y_valid,
            alphas=alphas,
            run_dir=run_dir,
            extra={
                "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
                "transform_fit_seconds": pca_fit_seconds,
            },
        )
        all_rows.append(row)
        pca_rows.append(row)

    print("Running correlation cluster representative and mean schemes")
    all_cluster_frames: list[pd.DataFrame] = []
    for threshold in [float(value) for value in cluster_cfg.get("thresholds", [])]:
        threshold_name = f"t{safe_name(threshold)}"
        clusters, selected_features = medoid_feature_clusters_from_corr(
            _corr,
            pearson_ranking,
            threshold=threshold,
            low_target_corr_threshold=float(cluster_cfg.get("low_target_corr_threshold", 1e-4)),
            linkage_method=str(cluster_cfg.get("linkage_method", "average")),
        )
        clusters.to_csv(run_dir / f"cluster_members_{threshold_name}.csv", index=False)
        all_cluster_frames.append(clusters)
        x_train_rep, x_valid_rep = feature_matrix(train_part, valid_part, selected_features, prep_cfg)
        rep_row, _ = evaluate_ridge_scheme(
            method="cluster_representative",
            scheme=threshold_name,
            feature_count=len(selected_features),
            x_train=x_train_rep,
            y_train=y_train,
            x_valid=x_valid_rep,
            y_valid=y_valid,
            alphas=alphas,
            run_dir=run_dir,
            extra={"cluster_threshold": threshold},
        )
        all_rows.append(rep_row)
        cluster_rows.append(rep_row)

        x_train_mean, x_valid_mean, mean_manifest = cluster_mean_matrices(train_part, valid_part, feature_cols, clusters, prep_cfg)
        mean_manifest.to_csv(run_dir / f"cluster_mean_manifest_{threshold_name}.csv", index=False)
        mean_row, _ = evaluate_ridge_scheme(
            method="cluster_mean",
            scheme=threshold_name,
            feature_count=x_train_mean.shape[1],
            x_train=x_train_mean,
            y_train=y_train,
            x_valid=x_valid_mean,
            y_valid=y_valid,
            alphas=alphas,
            run_dir=run_dir,
            extra={"cluster_threshold": threshold},
        )
        all_rows.append(mean_row)
        cluster_rows.append(mean_row)
    plot_cluster_sizes(all_cluster_frames, figure_dir)

    print("Running ElasticNet full-feature baseline")
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import ElasticNet

    elastic_cfg = model_cfg["elasticnet"]
    start = time.perf_counter()
    elastic = ElasticNet(
        alpha=float(elastic_cfg.get("alpha", 0.0005)),
        l1_ratio=float(elastic_cfg.get("l1_ratio", 0.2)),
        max_iter=int(elastic_cfg.get("max_iter", 3000)),
        tol=float(elastic_cfg.get("tol", 0.0001)),
        random_state=int(elastic_cfg.get("random_state", 42)),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        elastic.fit(x_train_full, y_train)
    elastic_pred = elastic.predict(x_valid_full)
    elastic_row = {
        "method": "elasticnet",
        "scheme": "full_features",
        "model": "elasticnet",
        "feature_count": len(feature_cols),
        "best_alpha": float(elastic_cfg.get("alpha", 0.0005)),
        "pearson": pearson_corr(y_valid, elastic_pred),
        "rmse": rmse(y_valid, elastic_pred),
        "fit_seconds": time.perf_counter() - start,
        "fit_scope": "outer_train_only",
        "selection_scope": "outer_train_only",
        "nonzero_coef_count": int(np.sum(np.abs(elastic.coef_) > 1e-12)),
        "l1_ratio": float(elastic_cfg.get("l1_ratio", 0.2)),
        "warnings": ";".join(str(item.message) for item in caught),
    }
    all_rows.append(elastic_row)
    pd.DataFrame([elastic_row]).to_csv(run_dir / "elasticnet_metrics.csv", index=False)

    print("Running LightGBM importance ranking")
    lgb_model, lgb_ranking, x_inner_valid, inner_valid, lgb_summary = train_lightgbm_ranker(
        train_part,
        feature_cols,
        target_col,
        prep_cfg,
        model_cfg["lightgbm"],
        float(model_driven_cfg.get("inner_validation_fraction", 0.2)),
        run_dir,
    )
    model_driven_rows: list[dict[str, Any]] = []
    if lgb_ranking is not None:
        lgb_rows = evaluate_feature_list_schemes(
            "lightgbm_importance",
            lgb_ranking.rename(columns={"gain_importance": "score"}),
            "score",
            [int(value) for value in model_driven_cfg.get("top_k", top_k_values)],
            train_part,
            valid_part,
            y_train,
            y_valid,
            prep_cfg,
            alphas,
            run_dir,
            extra={"selection_scope": "outer_train_inner_split_only"},
        )
        all_rows.extend(lgb_rows)
        model_driven_rows.extend(lgb_rows)

    shap_summary: dict[str, Any] = {"skipped": bool(args.skip_shap)}
    if not args.skip_shap and lgb_model is not None and x_inner_valid is not None and inner_valid is not None:
        print("Running SHAP ranking")
        shap_ranking, shap_summary = build_shap_ranking(
            lgb_model,
            x_inner_valid,
            inner_valid,
            feature_cols,
            int(model_driven_cfg.get("shap_sample_rows", 20000)),
            run_dir,
        )
        if shap_ranking is not None:
            shap_rows = evaluate_feature_list_schemes(
                "shap",
                shap_ranking.rename(columns={"mean_abs_shap": "score"}),
                "score",
                [int(value) for value in model_driven_cfg.get("top_k", top_k_values)],
                train_part,
                valid_part,
                y_train,
                y_valid,
                prep_cfg,
                alphas,
                run_dir,
                extra={"selection_scope": "outer_train_inner_valid_shap_sample_only"},
            )
            all_rows.extend(shap_rows)
            model_driven_rows.extend(shap_rows)
    elif args.skip_shap:
        pd.DataFrame(columns=["feature", "mean_abs_shap", "rank", "fit_scope", "sample_rows"]).to_csv(run_dir / "shap_feature_ranking.csv", index=False)

    metrics = pd.DataFrame(all_rows)
    metrics = metrics.sort_values(["pearson", "method", "scheme"], ascending=[False, True, True]).reset_index(drop=True)
    metrics.to_csv(run_dir / "tradeoff_summary.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(run_dir / "feature_selection_metrics.csv", index=False)
    pd.DataFrame(pca_rows).to_csv(run_dir / "pca_metrics.csv", index=False)
    pd.DataFrame(cluster_rows).to_csv(run_dir / "cluster_reduction_metrics.csv", index=False)
    if model_driven_rows:
        pd.DataFrame(model_driven_rows).to_csv(run_dir / "model_driven_metrics.csv", index=False)

    plot_dimension_vs_pearson(metrics, figure_dir)
    plot_method_comparison(metrics, figure_dir)
    plot_tradeoff(metrics, figure_dir, len(feature_cols))
    plot_model_driven(metrics, figure_dir)

    best_by_method = metrics.sort_values("pearson", ascending=False).groupby("method", as_index=False).head(1)
    best_by_method.to_csv(run_dir / "best_by_method.csv", index=False)
    write_json(
        run_dir / "feature_dimensionality_summary.json",
        {
            "data_summary": {
                "train_shape": list(train_df.shape),
                "feature_count": len(feature_cols),
                "outer_train_rows": len(train_part),
                "outer_holdout_rows": len(valid_part),
            },
            "best_overall": metrics.iloc[0].to_dict(),
            "best_by_method": best_by_method.to_dict(orient="records"),
            "correlation_summary": corr_summary.to_dict(orient="records"),
            "lightgbm_summary": lgb_summary,
            "shap_summary": shap_summary,
            "no_submission_generated": True,
        },
    )

    print("\n=== Extension 1 tradeoff summary ===")
    print(metrics.loc[:, ["method", "scheme", "feature_count", "model", "pearson", "rmse", "best_alpha"]].head(30).to_string(index=False))
    print(f"\nRun directory: {run_dir.relative_to(ROOT)}")
    print(f"Figure directory: {figure_dir.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
