from pathlib import Path

import pandas as pd
import pytest

from product_subject.config import DATA_DIR
from product_subject.data import load_frames, training_frame
from product_subject.submission import validate_submission


@pytest.fixture(scope="module")
def frames():
  return load_frames(DATA_DIR)


def test_train_test_no_raw_text_overlap(frames):
    train, test, _ = frames
    assert set(train["text"]).isdisjoint(set(test["text"]))


def test_training_unique_text_smaller(frames):
    train, _, _ = frames
    unique = training_frame(train, "unique_text")
    assert len(unique) < len(train)
    assert unique["text_norm"].is_unique


def test_submission_schema(frames):
    train, test, _ = frames
    label_space = set(train["subject"].unique())
    sub = test[["id"]].copy()
    sub["subject"] = train["subject"].mode().iloc[0]
    validate_submission(sub, test, label_space)