from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PurgedGroupSplit:
    fold: int
    valid_group: int
    train_indices: np.ndarray
    valid_indices: np.ndarray
    purged_groups: list[int]


def purged_group_time_series_splits(
    n_rows: int,
    n_groups: int = 6,
    gap: int = 1,
) -> list[PurgedGroupSplit]:
    """Create two-sided diagnostic folds with adjacent groups purged.

    Historical experiments use training groups both before and after validation.
    This is not a forward-only forecast evaluation; its behavior is preserved.
    """
    if n_rows < n_groups:
        raise ValueError("n_rows must be at least n_groups.")
    if n_groups < 2:
        raise ValueError("n_groups must be at least 2.")
    if gap < 0:
        raise ValueError("gap must be non-negative.")

    groups = np.array_split(np.arange(n_rows, dtype=np.int64), n_groups)
    splits: list[PurgedGroupSplit] = []
    for valid_group, valid_indices in enumerate(groups):
        purge_start = max(0, valid_group - gap)
        purge_end = min(n_groups - 1, valid_group + gap)
        purged = list(range(purge_start, purge_end + 1))
        train_groups = [idx for idx in range(n_groups) if idx not in purged]
        if not train_groups:
            raise ValueError("Purged split produced an empty training fold.")
        train_indices = np.concatenate([groups[idx] for idx in train_groups])
        splits.append(
            PurgedGroupSplit(
                fold=valid_group,
                valid_group=valid_group,
                train_indices=train_indices.astype(np.int64, copy=False),
                valid_indices=valid_indices.astype(np.int64, copy=False),
                purged_groups=purged,
            )
        )
    return splits


def rolling_time_window_indices(
    n_rows: int,
    *,
    train_start: float,
    train_end: float,
    valid_start: float,
    valid_end: float,
    embargo_rows: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return forward-only windows in the existing input row order.

    Remove embargo_rows from the end of the training window. Reject invalid
    windows instead of silently disabling an embargo that empties training.
    """
    fractions = np.asarray([train_start, train_end, valid_start, valid_end], dtype=float)
    if not np.isfinite(fractions).all() or not (
        0 <= train_start < train_end <= valid_start < valid_end <= 1
    ):
        raise ValueError("Rolling windows must satisfy 0 <= train_start < train_end <= valid_start < valid_end <= 1.")
    if not isinstance(n_rows, (int, np.integer)) or n_rows <= 0:
        raise ValueError("n_rows must be a positive integer.")
    if not isinstance(embargo_rows, (int, np.integer)) or embargo_rows < 0:
        raise ValueError("embargo_rows must be a non-negative integer.")
    train_first, train_stop, valid_first, valid_stop = (int(n_rows * value) for value in fractions)
    train_stop -= embargo_rows
    if train_first >= train_stop or valid_first >= valid_stop:
        raise ValueError("Rolling windows or embargo produced an empty training or validation set.")
    return (
        np.arange(train_first, train_stop, dtype=np.int64),
        np.arange(valid_first, valid_stop, dtype=np.int64),
    )
