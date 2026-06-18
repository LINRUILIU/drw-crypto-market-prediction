from __future__ import annotations

import numpy as np
import pandas as pd


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
