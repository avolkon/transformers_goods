from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


def validate_submission(
    submission: pd.DataFrame,
    test: pd.DataFrame,
    label_space: set[str],
) -> None:
    if list(submission.columns) != ["id", "subject"]:
        raise ValueError(f"columns must be id,subject; got {list(submission.columns)}")
    if len(submission) != len(test):
        raise ValueError(f"row count {len(submission)} != test {len(test)}")
    if set(submission["id"]) != set(test["id"]):
        raise ValueError("submission id set does not match test")
    unknown = set(submission["subject"]) - label_space
    if unknown:
        raise ValueError(f"subjects not in train label space: {sorted(unknown)[:5]} ...")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_submission(df: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return sha256_file(path)