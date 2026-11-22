from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from product_subject.config import RANDOM_STATE

logger = logging.getLogger(__name__)


def build_features(
    *,
    max_features_word: int,
    max_features_char: int,
    ngram_word: tuple[int, int],
    char_ngram: tuple[int, int],
    min_df: int,
) -> ColumnTransformer:
    word_vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=ngram_word,
        min_df=min_df,
        max_df=0.98,
        max_features=max_features_word,
        sublinear_tf=True,
    )
    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=char_ngram,
        min_df=min_df,
        max_features=max_features_char,
        sublinear_tf=True,
    )
    return ColumnTransformer(
        [
            ("word", word_vec, "text_norm"),
            ("char", char_vec, "text_norm"),
        ],
        sparse_threshold=0.3,
    )


def make_classifier(name: str, **params):
    if name == "logreg":
        return LogisticRegression(
            C=params.get("C", 2.0),
            max_iter=params.get("max_iter", 400),
            class_weight="balanced",
            solver="lbfgs",
            random_state=RANDOM_STATE,
        )
    if name == "linearsvc":
        return LinearSVC(
            C=params.get("C", 1.0),
            class_weight="balanced",
            dual="auto",
            max_iter=params.get("max_iter", 4000),
            random_state=RANDOM_STATE,
        )
    if name == "ridge":
        return RidgeClassifier(
            alpha=params.get("alpha", 1.0),
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
    if name == "sgd":
        return SGDClassifier(
            loss="modified_huber",
            alpha=params.get("alpha", 1e-5),
            max_iter=params.get("max_iter", 30),
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
    raise ValueError(name)


def build_pipeline(
    *,
    max_features: int,
    ngram_word: tuple[int, int],
    char_ngram: tuple[int, int],
    C: float,
    max_iter: int,
    clf_name: str = "logreg",
    min_df: int = 2,
    max_features_char: int | None = None,
    **clf_params,
) -> Pipeline:
    features = build_features(
        max_features_word=max_features,
        max_features_char=max_features_char or max_features,
        ngram_word=ngram_word,
        char_ngram=char_ngram,
        min_df=min_df,
    )
    params = {"C": C, "max_iter": max_iter, **clf_params}
    clf = make_classifier(clf_name, **params)
    return Pipeline([("features", features), ("clf", clf)])


def run_group_cv(
    train: pd.DataFrame,
    pipeline_template: Pipeline,
    *,
    n_splits: int,
    seed: int,
) -> dict[str, Any]:
    X = train[["text_norm"]]
    y = train["subject"].to_numpy()
    groups = train["text_norm"].to_numpy()
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_scores: list[float] = []
    t0 = time.perf_counter()
    for fold, (tr, va) in enumerate(cv.split(X, y, groups=groups), start=1):
        model = clone(pipeline_template)
        model.fit(X.iloc[tr], y[tr])
        pred = model.predict(X.iloc[va])
        score = f1_score(y[va], pred, average="macro", zero_division=0)
        fold_scores.append(float(score))
        logger.info("cv fold %s/%s macro-F1=%.5f", fold, n_splits, score)
    elapsed = time.perf_counter() - t0
    summary = {
        "n_splits": n_splits,
        "macro_f1_mean": float(np.mean(fold_scores)),
        "macro_f1_std": float(np.std(fold_scores)),
        "macro_f1_per_fold": fold_scores,
        "cv_seconds": round(elapsed, 2),
    }
    logger.info(
        "cv macro-F1 mean=%.5f std=%.5f (%.1fs)",
        summary["macro_f1_mean"],
        summary["macro_f1_std"],
        elapsed,
    )
    return summary