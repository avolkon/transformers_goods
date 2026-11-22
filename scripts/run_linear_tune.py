"""Tune linear TF-IDF models; write the best submission."""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from product_subject.baseline import build_pipeline  # noqa: E402
from product_subject.config import DATA_DIR, RANDOM_STATE  # noqa: E402
from product_subject.data import load_frames, training_frame  # noqa: E402
from product_subject.submission import validate_submission, write_submission  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

OUT = ROOT / "submissions" / "linear_v2"

CANDIDATES = [
    {
        "name": "baseline_logreg",
        "clf_name": "logreg",
        "train_rows": "unique_text",
        "max_features": 80_000,
        "max_features_char": 80_000,
        "ngram_word": (1, 2),
        "char_ngram": (3, 5),
        "min_df": 2,
        "C": 2.0,
        "max_iter": 300,
    },
    {
        "name": "svc_c1_unique",
        "clf_name": "linearsvc",
        "train_rows": "unique_text",
        "max_features": 120_000,
        "max_features_char": 150_000,
        "ngram_word": (1, 2),
        "char_ngram": (2, 6),
        "min_df": 1,
        "C": 1.0,
        "max_iter": 4000,
    },
    {
        "name": "svc_c2_all",
        "clf_name": "linearsvc",
        "train_rows": "all",
        "max_features": 120_000,
        "max_features_char": 150_000,
        "ngram_word": (1, 2),
        "char_ngram": (2, 6),
        "min_df": 1,
        "C": 2.0,
        "max_iter": 4000,
    },
    {
        "name": "svc_word3_unique",
        "clf_name": "linearsvc",
        "train_rows": "unique_text",
        "max_features": 150_000,
        "max_features_char": 150_000,
        "ngram_word": (1, 3),
        "char_ngram": (3, 5),
        "min_df": 1,
        "C": 1.0,
        "max_iter": 4000,
    },
    {
        "name": "ridge_unique",
        "clf_name": "ridge",
        "train_rows": "unique_text",
        "max_features": 120_000,
        "max_features_char": 120_000,
        "ngram_word": (1, 2),
        "char_ngram": (2, 6),
        "min_df": 1,
        "alpha": 0.5,
        "max_iter": 1,
        "C": 1.0,
    },
    {
        "name": "sgd_all",
        "clf_name": "sgd",
        "train_rows": "all",
        "max_features": 120_000,
        "max_features_char": 150_000,
        "ngram_word": (1, 2),
        "char_ngram": (2, 6),
        "min_df": 1,
        "alpha": 1e-5,
        "max_iter": 40,
        "C": 1.0,
    },
]


def holdout_score(pipe: Pipeline, df: pd.DataFrame, seed: int) -> float:
    X = df[["text_norm"]]
    y = df["subject"].to_numpy()
    groups = df["text_norm"].to_numpy()
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    tr, va = next(cv.split(X, y, groups=groups))
    model = clone(pipe)
    model.fit(X.iloc[tr], y[tr])
    pred = model.predict(X.iloc[va])
    return float(f1_score(y[va], pred, average="macro", zero_division=0))


def pipeline_from_cfg(cfg: dict) -> Pipeline:
    return build_pipeline(
        max_features=cfg["max_features"],
        ngram_word=tuple(cfg["ngram_word"]),
        char_ngram=tuple(cfg["char_ngram"]),
        C=cfg.get("C", 1.0),
        max_iter=cfg.get("max_iter", 400),
        clf_name=cfg["clf_name"],
        min_df=cfg.get("min_df", 1),
        max_features_char=cfg.get("max_features_char"),
        alpha=cfg.get("alpha", 1.0),
    )


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    train, test, _ = load_frames(DATA_DIR)
    label_space = set(train["subject"].unique())
    results = []

    for cfg in CANDIDATES:
        fit_df = training_frame(train, cfg["train_rows"])
        pipe = pipeline_from_cfg(cfg)
        t0 = time.perf_counter()
        score = holdout_score(pipe, fit_df, RANDOM_STATE)
        elapsed = time.perf_counter() - t0
        row = {**cfg, "holdout_macro_f1": score, "seconds": round(elapsed, 1)}
        results.append(row)
        logger.info("%s holdout_macro_f1=%.5f (%.1fs)", cfg["name"], score, elapsed)

    ranked = sorted(results, key=lambda r: r["holdout_macro_f1"], reverse=True)
    best = ranked[0]
    logger.info("best=%s f1=%.5f", best["name"], best["holdout_macro_f1"])

    fit_df = training_frame(train, best["train_rows"])
    pipe = pipeline_from_cfg(best)
    t0 = time.perf_counter()
    pipe.fit(fit_df[["text_norm"]], fit_df["subject"].to_numpy())
    fit_sec = time.perf_counter() - t0
    pred = pipe.predict(test[["text_norm"]])
    submission = pd.DataFrame({"id": test["id"], "subject": pred})
    validate_submission(submission, test, label_space)
    sub_path = OUT / "submission.csv"
    sub_hash = write_submission(submission, sub_path)
    joblib.dump(pipe, OUT / "model.joblib")

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "best": best,
        "ranked": ranked,
        "fit_seconds": round(fit_sec, 2),
        "submission_sha256": sub_hash,
        "n_changed_vs_baseline": None,
    }
    baseline_path = ROOT / "submissions" / "baseline" / "submission.csv"
    if baseline_path.exists():
        old = pd.read_csv(baseline_path)
        merged = old.merge(submission, on="id", suffixes=("_old", "_new"))
        n_diff = int((merged["subject_old"] != merged["subject_new"]).sum())
        report["n_changed_vs_baseline"] = n_diff
        report["agree_vs_baseline"] = 1.0 - n_diff / len(merged)
        logger.info("diff vs baseline: %s / %s", n_diff, len(merged))

    (OUT / "tune_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT / "meta.json").write_text(
        json.dumps(
            {
                "experiment": "linear_v2",
                "best_name": best["name"],
                "holdout_macro_f1": best["holdout_macro_f1"],
                "submission_sha256": sub_hash,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("wrote %s", sub_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())