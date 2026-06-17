from __future__ import annotations

import numpy as np


def pearson_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    true_centered = y_true - np.nanmean(y_true)
    pred_centered = y_pred - np.nanmean(y_pred)
    denom = np.sqrt(np.nansum(true_centered**2)) * np.sqrt(np.nansum(pred_centered**2))
    if not np.isfinite(denom) or denom == 0:
        return 0.0
    return float(np.nansum(true_centered * pred_centered) / denom)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.nanmean((y_true - y_pred) ** 2)))

