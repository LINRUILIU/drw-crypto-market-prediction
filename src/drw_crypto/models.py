from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class NumpyRidgeRegressor:
    alpha: float = 1.0
    coef_: np.ndarray | None = None
    intercept_: float = 0.0

    def fit(self, x: np.ndarray, y: np.ndarray) -> "NumpyRidgeRegressor":
        x64 = np.asarray(x, dtype=np.float64)
        y64 = np.asarray(y, dtype=np.float64)
        self.intercept_ = float(np.mean(y64))
        y_centered = y64 - self.intercept_

        xtx = x64.T @ x64
        xty = x64.T @ y_centered
        reg = float(self.alpha) * np.eye(xtx.shape[0], dtype=np.float64)
        try:
            self.coef_ = np.linalg.solve(xtx + reg, xty)
        except np.linalg.LinAlgError:
            self.coef_ = np.linalg.lstsq(xtx + reg, xty, rcond=None)[0]
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("NumpyRidgeRegressor must be fitted before predict().")
        return np.asarray(x, dtype=np.float64) @ self.coef_ + self.intercept_

