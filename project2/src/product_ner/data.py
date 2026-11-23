from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

SLOTS = ("colors", "materials", "length", "size")


def load_vocabs(path: Path) -> dict[str, list[str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("vocabs.json must be an object of four lists")
    missing = [slot for slot in SLOTS if slot not in raw]
    if missing:
        raise ValueError(f"vocabs.json missing keys: {missing}")
    vocabs: dict[str, list[str]] = {}
    for slot in SLOTS:
        values = raw[slot]
        if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
            raise ValueError(f"vocabs[{slot}] must be a list of strings")
        if len(values) != len(set(values)):
            raise ValueError(f"vocabs[{slot}] has duplicate entries")
        vocabs[slot] = list(values)
    return vocabs


def load_frames(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    train = pd.read_csv(data_dir / "train.csv", dtype=str, keep_default_na=False)
    test = pd.read_csv(data_dir / "test.csv", dtype=str, keep_default_na=False)
    sample = pd.read_csv(data_dir / "sample_submission.csv", dtype=str, keep_default_na=False)
    vocabs = load_vocabs(data_dir / "vocabs.json")
    for name, df, cols in (
        ("train", train, {"id", "title", "description"}),
        ("test", test, {"id", "title", "description"}),
        ("sample", sample, {"id", "colors", "materials", "length", "size"}),
    ):
        missing = cols - set(df.columns)
        if missing:
            raise ValueError(f"{name}.csv missing columns: {sorted(missing)}")
    overlap = set(train["id"]) & set(test["id"])
    if overlap:
        raise ValueError(f"train/test id overlap: {len(overlap)}")
    logger.info(
        "loaded train=%s test=%s sample=%s vocab_sizes=%s",
        train.shape,
        test.shape,
        sample.shape,
        {slot: len(values) for slot, values in vocabs.items()},
    )
    return train, test, sample, vocabs
