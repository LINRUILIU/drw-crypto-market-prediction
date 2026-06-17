from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


TRAIN_CANDIDATES = (
    "train.parquet",
    "train.csv",
    "train.csv.zip",
    "train.pkl",
    "train.pickle",
)

TEST_CANDIDATES = (
    "test.parquet",
    "test.csv",
    "test.csv.zip",
    "test.pkl",
    "test.pickle",
)

SAMPLE_SUBMISSION_CANDIDATES = (
    "sample_submission.csv",
    "sample_submission.parquet",
    "sample_submission.pkl",
)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def find_first_existing(raw_dir: str | Path, configured: str | None, candidates: Iterable[str]) -> Path | None:
    raw_dir = Path(raw_dir)
    if configured:
        path = Path(configured)
        if not path.is_absolute():
            path = raw_dir / path
        return path if path.exists() else None

    for name in candidates:
        path = raw_dir / name
        if path.exists():
            return path
    return None


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    name = path.name.lower()
    suffix = path.suffix.lower()

    if name.endswith(".csv") or name.endswith(".csv.zip"):
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    if suffix == ".feather":
        return pd.read_feather(path)

    raise ValueError(f"Unsupported data file type: {path}")


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True, default=str)


def write_lines(path: str | Path, lines: Iterable[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for line in lines:
            f.write(f"{line}\n")

