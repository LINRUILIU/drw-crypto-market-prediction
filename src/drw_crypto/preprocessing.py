from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


TARGET_CANDIDATES = ("target", "label", "y", "Target", "Label")
ID_CANDIDATES = ("id", "ID", "Id", "row_id", "rowid", "sample_id")


def infer_target_column(train_df: pd.DataFrame, test_df: pd.DataFrame | None, configured: str | None) -> str:
    if configured and configured in train_df.columns:
        return configured

    for candidate in TARGET_CANDIDATES:
        if candidate in train_df.columns:
            return candidate

    if test_df is not None:
        train_only = [col for col in train_df.columns if col not in test_df.columns]
        if len(train_only) == 1:
            return train_only[0]

    raise ValueError(
        "Could not infer target column. Set data.target_col in configs/main_baseline.yaml."
    )


def infer_id_column(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame | None,
    sample_submission: pd.DataFrame | None,
    target_col: str,
    configured: str | None,
) -> str | None:
    if configured and configured in train_df.columns:
        return configured

    for candidate in ID_CANDIDATES:
        if candidate in train_df.columns and candidate != target_col:
            return candidate

    if sample_submission is not None and test_df is not None and len(sample_submission.columns) >= 2:
        first_col = str(sample_submission.columns[0])
        if first_col in train_df.columns and first_col in test_df.columns and first_col != target_col:
            return first_col

    return None


def infer_feature_columns(
    train_df: pd.DataFrame,
    target_col: str,
    id_col: str | None,
    drop_columns: Sequence[str],
    numeric_only: bool,
) -> list[str]:
    excluded = {target_col, *drop_columns}
    if id_col:
        excluded.add(id_col)

    feature_cols: list[str] = []
    for col in train_df.columns:
        if col in excluded:
            continue
        if numeric_only and not pd.api.types.is_numeric_dtype(train_df[col]):
            continue
        feature_cols.append(col)

    if not feature_cols:
        raise ValueError("No feature columns were inferred.")
    return feature_cols


@dataclass
class TabularPreprocessor:
    feature_cols: list[str]
    fill_values: pd.Series | None = None
    means: pd.Series | None = None
    scales: pd.Series | None = None
    standardize: bool = True

    def fit(self, df: pd.DataFrame, missing_fill: str = "median", standardize: bool = True) -> "TabularPreprocessor":
        self.standardize = standardize
        frame = self._numeric_frame(df)

        if missing_fill == "median":
            fill_values = frame.median(axis=0, skipna=True)
        elif missing_fill == "mean":
            fill_values = frame.mean(axis=0, skipna=True)
        elif missing_fill == "zero":
            fill_values = pd.Series(0.0, index=self.feature_cols)
        else:
            raise ValueError(f"Unsupported missing_fill: {missing_fill}")

        self.fill_values = fill_values.fillna(0.0).astype("float32")
        filled = frame.fillna(self.fill_values)

        self.means = filled.mean(axis=0).astype("float32")
        scales = filled.std(axis=0, ddof=0).replace(0.0, 1.0).fillna(1.0)
        self.scales = scales.astype("float32")
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.fill_values is None or self.means is None or self.scales is None:
            raise RuntimeError("TabularPreprocessor must be fitted before transform().")

        frame = self._numeric_frame(df).fillna(self.fill_values)
        x = frame.to_numpy(dtype=np.float32, copy=True)
        if self.standardize:
            x -= self.means.to_numpy(dtype=np.float32)
            x /= self.scales.to_numpy(dtype=np.float32)
        return x

    def _numeric_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.loc[:, self.feature_cols].replace([np.inf, -np.inf], np.nan)

