"""FE-0: TF-IDF + logistic regression baseline with experiment artifacts."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from product_subject import __version__  # noqa: E402
from product_subject.baseline import build_pipeline, run_group_cv  # noqa: E402
from product_subject.config import DATA_DIR, RANDOM_STATE  # noqa: E402
from product_subject.data import load_frames, training_frame  # noqa: E402
from product_subject.submission import sha256_file, validate_submission, write_submission  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

EXPERIMENT = "baseline"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TF-IDF logistic baseline (structured)")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "submissions" / EXPERIMENT,
    )
    parser.add_argument("--no-cv", action="store_true")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--max-features", type=int, default=80_000)
    parser.add_argument("--C", type=float, default=2.0)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument(
        "--train-rows",
        choices=("all", "unique_text"),
        default="unique_text",
        help="unique_text avoids overweighting duplicate titles (recommended)",
    )
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = args.out_dir / "artifacts"
    models_dir.mkdir(parents=True, exist_ok=True)

    train, test, _ = load_frames(args.data_dir)
    label_space = set(train["subject"].unique())
    fit_train = training_frame(train, args.train_rows)

    pipeline = build_pipeline(
        max_features=args.max_features,
        ngram_word=(1, 2),
        char_ngram=(3, 5),
        C=args.C,
        max_iter=args.max_iter,
    )

    metrics: dict = {
        "model": "tfidf_logreg",
        "package_version": __version__,
        "train_rows": args.train_rows,
    }
    if not args.no_cv:
        metrics["cv"] = run_group_cv(
            train, pipeline, n_splits=args.cv_folds, seed=args.seed
        )

    X_fit = fit_train[["text_norm"]]
    y_fit = fit_train["subject"].to_numpy()
    t0 = time.perf_counter()
    pipeline.fit(X_fit, y_fit)
    fit_sec = time.perf_counter() - t0
    metrics["fit_seconds"] = round(fit_sec, 2)
    metrics["n_train_rows_fit"] = len(fit_train)
    metrics["n_train_rows_total"] = len(train)
    metrics["n_classes"] = len(label_space)

    pred_test = pipeline.predict(test[["text_norm"]])
    submission = pd.DataFrame({"id": test["id"], "subject": pred_test})
    validate_submission(submission, test, label_space)

    sub_path = args.out_dir / "submission.csv"
    sub_hash = write_submission(submission, sub_path)
    metrics["submission_sha256"] = sub_hash

    model_path = models_dir / "tfidf_logreg.joblib"
    joblib.dump(pipeline, model_path)

    meta = {
        "experiment": EXPERIMENT,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "submission_path": str(sub_path.relative_to(ROOT)),
        "submission_sha256": sub_hash,
        "model_path": str(model_path.relative_to(ROOT)),
        "train_rows": args.train_rows,
        "seed": args.seed,
        "hyperparams": {
            "max_features": args.max_features,
            "C": args.C,
            "max_iter": args.max_iter,
        },
    }
    (args.out_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.out_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    legacy_sub = ROOT / "submissions" / "baseline_submission.csv"
    legacy_sub.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(legacy_sub, index=False)

    logger.info("wrote %s sha256=%s...", sub_path, sub_hash[:12])
    logger.info("metrics cv_mean=%s", metrics.get("cv", {}).get("macro_f1_mean"))
    return 0


if __name__ == "__main__":
    sys.exit(main())