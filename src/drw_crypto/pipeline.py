from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from drw_crypto.io import write_lines
from drw_crypto.metrics import pearson_corr, rmse
from drw_crypto.models import NumpyRidgeRegressor
from drw_crypto.preprocessing import ID_CANDIDATES, TabularPreprocessor


def split_time_ordered(df: pd.DataFrame, validation_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")
    split_at = int(len(df) * (1.0 - validation_fraction))
    if split_at <= 0 or split_at >= len(df):
        raise ValueError("Validation split produced an empty train or validation set.")
    return df.iloc[:split_at], df.iloc[split_at:]


def choose_prediction_column(
    sample_submission: pd.DataFrame | None,
    configured: str | None,
    target_col: str,
) -> str:
    if configured:
        return configured
    if sample_submission is None:
        return "prediction"
    if target_col in sample_submission.columns:
        return target_col
    if "prediction" in sample_submission.columns:
        return "prediction"
    return str(sample_submission.columns[-1])


def make_submission(
    sample_submission: pd.DataFrame | None,
    test_df: pd.DataFrame,
    pred: np.ndarray,
    prediction_col: str,
    id_col: str | None,
) -> pd.DataFrame:
    """Validate predictions and preserve the row order used to generate them.

    When both inputs contain IDs, their order must match; this function never
    silently reorders predictions. ID-free test data uses positional alignment.
    """
    values = np.asarray(pred)
    if values.ndim != 1 or len(values) != len(test_df):
        raise ValueError("Predictions must be one-dimensional and match the test row count.")
    if (
        not np.issubdtype(values.dtype, np.number)
        or np.iscomplexobj(values)
        or not np.isfinite(values).all()
    ):
        raise ValueError("Predictions must contain only finite real numbers.")
    if not test_df.columns.is_unique:
        raise ValueError("Test data has duplicate column names.")
    if prediction_col == id_col or (
        prediction_col in ID_CANDIDATES
        and (
            prediction_col in test_df.columns
            or (sample_submission is not None and prediction_col in sample_submission.columns)
        )
    ):
        raise ValueError("The prediction column must not overwrite the ID column.")

    if sample_submission is not None:
        if not sample_submission.columns.is_unique:
            raise ValueError("Sample submission has duplicate column names.")
        if len(sample_submission) != len(values):
            raise ValueError("Prediction length does not match sample submission length.")
        if prediction_col not in sample_submission.columns:
            raise ValueError(f"Prediction column {prediction_col!r} is not in sample submission.")

    if id_col is None:
        candidates = [
            col for col in ID_CANDIDATES
            if col != prediction_col and (
                col in test_df.columns
                or (sample_submission is not None and col in sample_submission.columns)
            )
        ]
        if len(candidates) > 1:
            raise ValueError("Multiple ID columns found; specify id_col explicitly.")
        if candidates:
            id_col = candidates[0]
        elif sample_submission is not None and len(sample_submission.columns) == 2:
            candidate = next(col for col in sample_submission.columns if col != prediction_col)
            if candidate in test_df.columns:
                id_col = candidate
    if id_col == prediction_col:
        raise ValueError("The prediction column must not overwrite the ID column.")

    test_ids = template_ids = None
    if id_col is not None:
        for label, frame in (("Test data", test_df), ("Sample submission", sample_submission)):
            if frame is None or id_col not in frame.columns:
                continue
            ids = pd.Index(frame[id_col])
            if ids.hasnans or not ids.is_unique:
                raise ValueError(f"{label} IDs must be unique and non-missing.")
            if label == "Test data":
                test_ids = ids
            else:
                template_ids = ids
        if test_ids is None and template_ids is None:
            raise ValueError(f"ID column {id_col!r} was not found in either input.")
        if test_ids is not None and template_ids is not None and not test_ids.equals(template_ids):
            raise ValueError("Sample submission IDs must match test data in the same order.")

    if sample_submission is not None:
        submission = sample_submission.copy()
        submission[prediction_col] = values
        return submission
    if test_ids is not None:
        return pd.DataFrame({id_col: test_ids.to_numpy(), prediction_col: values})
    if prediction_col == "row_id":
        raise ValueError("The prediction column must not overwrite the generated row ID.")
    return pd.DataFrame({"row_id": np.arange(len(values), dtype=np.int64), prediction_col: values})


def save_valid_prediction(path: Path, y_true: np.ndarray, pred: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "row_index": np.arange(len(y_true), dtype=np.int64),
            "y_true": y_true,
            "prediction": pred,
        }
    ).to_csv(path, index=False)


def read_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def safe_name(value: float | int | str) -> str:
    if isinstance(value, str):
        text = value
    else:
        text = f"{value:g}"
    return text.replace(".", "p").replace("-", "m")


def fit_ridge(alpha: float, x_train: np.ndarray, y_train: np.ndarray, x_pred: np.ndarray) -> np.ndarray:
    return NumpyRidgeRegressor(alpha=float(alpha)).fit(x_train, y_train).predict(x_pred)


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
    root: Path,
) -> np.ndarray:
    output: np.ndarray | None = None
    for component in components:
        selected = read_lines(root / component["selected_features_path"])
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
    if output is None:
        raise ValueError("No current-best components configured.")
    return output


def write_selected_features(path: Path, features: list[str]) -> None:
    write_lines(path, features)
