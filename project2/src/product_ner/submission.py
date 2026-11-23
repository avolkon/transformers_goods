from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from product_ner.data import SLOTS

COLUMNS = ("id", *SLOTS)


def parse_slot_cell(value: object) -> list[str]:
    if isinstance(value, list):
        items = value
    else:
        try:
            items = json.loads(str(value))
        except json.JSONDecodeError as exc:
            raise ValueError(f"slot cell is not JSON: {value!r}") from exc
    if not isinstance(items, list) or not all(isinstance(item, str) for item in items):
        raise ValueError(f"slot cell must be a JSON list of strings: {value!r}")
    return items


def validate_submission(
    submission: pd.DataFrame,
    test: pd.DataFrame,
    vocabs: dict[str, list[str]],
) -> None:
    if list(submission.columns) != list(COLUMNS):
        raise ValueError(f"columns must be {list(COLUMNS)}; got {list(submission.columns)}")
    if len(submission) != len(test):
        raise ValueError(f"row count {len(submission)} != test {len(test)}")
    if submission["id"].duplicated().any():
        raise ValueError("submission has duplicate ids")
    if submission["id"].tolist() != test["id"].tolist():
        raise ValueError("submission id order does not match test")
    for slot in SLOTS:
        allowed = set(vocabs[slot])
        for raw in submission[slot]:
            items = parse_slot_cell(raw)
            if len(items) != len(set(items)):
                raise ValueError(f"duplicate values in {slot}: {items}")
            unknown = [item for item in items if item not in allowed]
            if unknown:
                raise ValueError(f"{slot} values not in vocab: {unknown[:5]}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_submission(df: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, lineterminator="\n")
    return sha256_file(path)
