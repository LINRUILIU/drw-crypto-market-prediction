from __future__ import annotations

import numpy as np
import pandas as pd


def pearson_feature_ranking(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> pd.DataFrame:
    """Rank features by train-only absolute Pearson correlation with target."""
    y = train_df[target_col].astype("float64")
    rows: list[dict[str, float | str]] = []

    for feature in feature_cols:
        x = train_df[feature].astype("float64").replace([np.inf, -np.inf], np.nan)
        valid = x.notna() & y.notna()
        if valid.sum() < 2:
            corr = 0.0
        else:
            x_valid = x[valid]
            y_valid = y[valid]
            if x_valid.nunique(dropna=True) <= 1:
                corr = 0.0
            else:
                corr = float(x_valid.corr(y_valid))
                if not np.isfinite(corr):
                    corr = 0.0
        rows.append({"feature": feature, "pearson": corr, "abs_pearson": abs(corr)})

    ranking = pd.DataFrame(rows)
    return ranking.sort_values(["abs_pearson", "feature"], ascending=[False, True]).reset_index(drop=True)

