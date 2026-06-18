from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


def target_feature_ranking(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    target: pd.Series,
    method: str,
    score_name: str | None = None,
) -> pd.DataFrame:
    """Rank features by train-only absolute correlation with an arbitrary target."""
    if method not in {"pearson", "spearman"}:
        raise ValueError(f"Unsupported ranking method: {method}")

    target = target.astype("float64").replace([np.inf, -np.inf], np.nan)
    metric = score_name or method
    rows: list[dict[str, float | str]] = []

    for feature in feature_cols:
        x = train_df[feature].astype("float64").replace([np.inf, -np.inf], np.nan)
        valid = x.notna() & target.notna()
        if valid.sum() < 2:
            corr = 0.0
        else:
            x_valid = x[valid]
            y_valid = target[valid]
            if x_valid.nunique(dropna=True) <= 1 or y_valid.nunique(dropna=True) <= 1:
                corr = 0.0
            else:
                corr = float(x_valid.corr(y_valid, method=method))
                if not np.isfinite(corr):
                    corr = 0.0
        rows.append({"feature": feature, metric: corr, f"abs_{metric}": abs(corr)})

    ranking = pd.DataFrame(rows)
    return ranking.sort_values([f"abs_{metric}", "feature"], ascending=[False, True]).reset_index(drop=True)


def pearson_feature_ranking(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> pd.DataFrame:
    """Rank features by train-only absolute Pearson correlation with target."""
    return target_feature_ranking(train_df, feature_cols, train_df[target_col], "pearson")


def spearman_feature_ranking(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> pd.DataFrame:
    """Rank features by train-only absolute Spearman correlation with target."""
    return target_feature_ranking(train_df, feature_cols, train_df[target_col], "spearman")


def stable_pearson_feature_ranking(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    windows: list[dict[str, float | str]],
    stability_penalty: float,
    min_weight: float,
) -> pd.DataFrame:
    """Rank features by rolling-window Pearson strength and stability."""
    n_rows = len(train_df)
    if n_rows < 2:
        raise ValueError("Need at least two rows for stable Pearson ranking.")

    per_window: list[pd.DataFrame] = []
    for window in windows:
        name = str(window["name"])
        start = int(n_rows * float(window["start"]))
        end = int(n_rows * float(window["end"]))
        if start < 0 or end > n_rows or start >= end:
            raise ValueError(f"Invalid stable ranking window: {window}")
        part = train_df.iloc[start:end]
        ranking = pearson_feature_ranking(part, feature_cols, target_col)
        ranking = ranking[["feature", "abs_pearson"]].rename(columns={"abs_pearson": f"abs_corr_{name}"})
        per_window.append(ranking)

    merged = per_window[0]
    for frame in per_window[1:]:
        merged = merged.merge(frame, on="feature", how="outer")
    corr_cols = [col for col in merged.columns if col.startswith("abs_corr_")]
    merged[corr_cols] = merged[corr_cols].fillna(0.0)
    merged["mean_abs_corr"] = merged[corr_cols].mean(axis=1)
    merged["std_abs_corr"] = merged[corr_cols].std(axis=1, ddof=0)
    merged["min_abs_corr"] = merged[corr_cols].min(axis=1)
    merged["stable_score"] = (
        merged["mean_abs_corr"]
        - float(stability_penalty) * merged["std_abs_corr"]
        + float(min_weight) * merged["min_abs_corr"]
    )
    return merged.sort_values(["stable_score", "feature"], ascending=[False, True]).reset_index(drop=True)


def absolute_feature_correlation(train_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Compute a train-only absolute feature-feature Pearson correlation matrix."""
    if not feature_cols:
        raise ValueError("feature_cols must not be empty.")
    frame = train_df.loc[:, feature_cols].replace([np.inf, -np.inf], np.nan)
    corr = frame.corr(method="pearson").abs()
    corr = corr.reindex(index=feature_cols, columns=feature_cols).fillna(0.0)
    values = corr.to_numpy(dtype=np.float32, copy=True)
    np.fill_diagonal(values, 1.0)
    return pd.DataFrame(values, index=feature_cols, columns=feature_cols)


def medoid_feature_clusters(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    threshold: float,
    low_target_corr_threshold: float = 1e-4,
    linkage_method: str = "average",
) -> tuple[pd.DataFrame, list[str]]:
    """Cluster correlated features and select one medoid feature per cluster.

    Distance is ``1 - abs(feature_feature_corr)``. The medoid is the feature with the
    highest sum of absolute correlations to other features in the same cluster.
    """
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be between 0 and 1.")
    if len(feature_cols) == 1:
        return (
            pd.DataFrame(
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
                    }
                ]
            ),
            [],
        )

    corr = absolute_feature_correlation(train_df, feature_cols)
    distance = 1.0 - corr.to_numpy(dtype=np.float64, copy=True)
    np.fill_diagonal(distance, 0.0)
    condensed = squareform(distance, checks=False)
    tree = linkage(condensed, method=linkage_method)
    labels = fcluster(tree, t=1.0 - float(threshold), criterion="distance")

    target_corr = pearson_feature_ranking(train_df, feature_cols, target_col)
    target_corr = target_corr.set_index("feature")[["pearson", "abs_pearson"]]

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


def aggregate_shap_stable_features(
    fold_top_features: pd.DataFrame,
    min_fold_appearances: int,
    fill_to_n: int,
) -> pd.DataFrame:
    """Aggregate per-fold SHAP top features into a stable selected feature table."""
    required = {"fold", "feature", "rank", "mean_abs_shap"}
    missing = required - set(fold_top_features.columns)
    if missing:
        raise ValueError(f"fold_top_features missing columns: {sorted(missing)}")

    grouped = (
        fold_top_features.groupby("feature", as_index=False)
        .agg(
            fold_appearances=("fold", "nunique"),
            mean_rank=("rank", "mean"),
            best_rank=("rank", "min"),
            mean_abs_shap=("mean_abs_shap", "mean"),
        )
        .sort_values(
            ["fold_appearances", "mean_rank", "mean_abs_shap", "feature"],
            ascending=[False, True, False, True],
        )
        .reset_index(drop=True)
    )
    grouped["selected"] = grouped["fold_appearances"] >= int(min_fold_appearances)
    grouped["selection_reason"] = np.where(grouped["selected"], "stable", "")

    if int(fill_to_n) > 0 and int(grouped["selected"].sum()) < int(fill_to_n):
        need = int(fill_to_n) - int(grouped["selected"].sum())
        fill_index = grouped.index[~grouped["selected"]][:need]
        grouped.loc[fill_index, "selected"] = True
        grouped.loc[fill_index, "selection_reason"] = "rank_fill"

    return grouped
