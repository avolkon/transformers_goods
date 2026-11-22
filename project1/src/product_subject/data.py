from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from product_subject.text import normalize_text

logger = logging.getLogger(__name__)


def load_frames(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    sample = pd.read_csv(data_dir / "sample_submission.csv")
    for name, df, cols in (
        ("train", train, {"id", "text", "subject"}),
        ("test", test, {"id", "text"}),
        ("sample", sample, {"id", "subject"}),
    ):
        missing = cols - set(df.columns)
        if missing:
            raise ValueError(f"{name}.csv missing columns: {sorted(missing)}")
    train = train.copy()
    test = test.copy()
    train["text_norm"] = train["text"].map(normalize_text)
    test["text_norm"] = test["text"].map(normalize_text)
    overlap_raw = set(train["text"]) & set(test["text"])
    if overlap_raw:
        raise ValueError(f"train/test leakage on raw text: {len(overlap_raw)} titles")
    norm_overlap = set(train["text_norm"]) & set(test["text_norm"])
    if norm_overlap and not overlap_raw:
        logger.warning(
            "%s train/test share normalized text without raw match", len(norm_overlap)
        )
    logger.info(
        "loaded train=%s test=%s classes=%s unique_train_texts=%s",
        train.shape,
        test.shape,
        train["subject"].nunique(),
        train["text_norm"].nunique(),
    )
    return train, test, sample


def training_frame(train: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Select rows used for final fit (all rows vs one row per unique title)."""
    if mode == "all":
        return train
    if mode == "unique_text":
        deduped = train.drop_duplicates(subset=["text_norm"], keep="first")
        logger.info(
            "train_fit unique_text: %s rows (from %s)", len(deduped), len(train)
        )
        return deduped
    raise ValueError(f"unknown training mode: {mode}")