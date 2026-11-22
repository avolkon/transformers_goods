"""Fine-tune LinearSVC C on the winning feature set."""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from product_subject.baseline import build_pipeline  # noqa: E402
from product_subject.config import DATA_DIR, RANDOM_STATE  # noqa: E402
from product_subject.data import load_frames, training_frame  # noqa: E402
from product_subject.submission import validate_submission, write_submission  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

OUT = ROOT / "submissions" / "linear_v2"


def score_c(C: float, df: pd.DataFrame) -> float:
    pipe = build_pipeline(
        max_features=120_000,
        max_features_char=150_000,
        ngram_word=(1, 2),
        char_ngram=(2, 6),
        min_df=1,
        C=C,
        max_iter=4000,
        clf_name="linearsvc",
    )
    X = df[["text_norm"]]
    y = df["subject"].to_numpy()
    groups = df["text_norm"].to_numpy()
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    tr, va = next(cv.split(X, y, groups=groups))
    model = clone(pipe)
    t0 = time.perf_counter()
    model.fit(X.iloc[tr], y[tr])
    pred = model.predict(X.iloc[va])
    s = float(f1_score(y[va], pred, average="macro", zero_division=0))
    logger.info("C=%s holdout=%.5f (%.1fs)", C, s, time.perf_counter() - t0)
    return s


def main() -> int:
    train, test, _ = load_frames(DATA_DIR)
    fit_df = training_frame(train, "unique_text")
    grid = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    scores = {c: score_c(c, fit_df) for c in grid}
    best_c = max(scores, key=scores.get)
    logger.info("best C=%s f1=%.5f", best_c, scores[best_c])

    pipe = build_pipeline(
        max_features=120_000,
        max_features_char=150_000,
        ngram_word=(1, 2),
        char_ngram=(2, 6),
        min_df=1,
        C=best_c,
        max_iter=4000,
        clf_name="linearsvc",
    )
    pipe.fit(fit_df[["text_norm"]], fit_df["subject"].to_numpy())
    pred = pipe.predict(test[["text_norm"]])
    submission = pd.DataFrame({"id": test["id"], "subject": pred})
    validate_submission(submission, test, set(train["subject"].unique()))
    sub_hash = write_submission(submission, OUT / "submission.csv")
    joblib.dump(pipe, OUT / "model.joblib")
    (OUT / "c_grid.json").write_text(json.dumps({"scores": scores, "best_C": best_c, "sha": sub_hash}, indent=2), encoding="utf-8")
    old = pd.read_csv(ROOT / "submissions" / "baseline" / "submission.csv")
    merged = old.merge(submission, on="id", suffixes=("_o", "_n"))
    n_diff = int((merged["subject_o"] != merged["subject_n"]).sum())
    logger.info("diff vs baseline %s sha=%s", n_diff, sub_hash[:12])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())