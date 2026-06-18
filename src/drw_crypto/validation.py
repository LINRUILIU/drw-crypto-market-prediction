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
    """Create contiguous group folds with adjacent groups purged from training."""
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
