"""Fine-tune a Russian transformer for subject classification."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from product_subject import __version__  # noqa: E402
from product_subject.config import DATA_DIR, RANDOM_STATE  # noqa: E402
from product_subject.data import load_frames, training_frame  # noqa: E402
from product_subject.encoder_train import (  # noqa: E402
    EncoderTrainConfig,
    predict_texts,
    train_encoder,
)
from product_subject.labels import fit_label_encoder, save_label_map  # noqa: E402
from product_subject.submission import validate_submission, write_submission  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_MODELS = {
    "tiny": "cointegrated/rubert-tiny2",
    "base": "ai-forever/ruBert-base",
    "rob": "DeepPavlov/rubert-base-cased",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Transformer fine-tune")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "submissions" / "encoder_v1")
    parser.add_argument(
        "--model",
        choices=list(DEFAULT_MODELS),
        default="tiny",
        help="preset HF model id",
    )
    parser.add_argument("--model-name", type=str, default="", help="override HF hub id")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--train-rows", choices=("unique_text", "all"), default="unique_text")
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args(argv)

    model_name = args.model_name or DEFAULT_MODELS[args.model]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    artifacts = args.out_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)

    train, test, _ = load_frames(args.data_dir)
    fit_train = training_frame(train, args.train_rows)
    label2id, id2label, class_weights = fit_label_encoder(fit_train["subject"])
    fit_train = fit_train.copy()
    fit_train["subject_id"] = fit_train["subject"].map(label2id)

    cfg = EncoderTrainConfig(
        model_name=model_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        fp16=args.fp16,
    )

    t0 = time.perf_counter()
    model, tokenizer, train_summary = train_encoder(
        fit_train,
        artifacts,
        label2id,
        id2label,
        class_weights,
        cfg,
    )
    train_sec = time.perf_counter() - t0

    pred_ids = predict_texts(
        model,
        tokenizer,
        test["text_norm"].tolist(),
        max_length=cfg.max_length,
        batch_size=cfg.batch_size,
    )
    subjects = [id2label[int(i)] for i in pred_ids]
    submission = pd.DataFrame({"id": test["id"], "subject": subjects})
    validate_submission(submission, test, set(label2id))

    sub_path = args.out_dir / "submission.csv"
    sub_hash = write_submission(submission, sub_path)
    save_label_map(label2id, id2label, artifacts / "label_map.json")

    metrics = {
        "model": model_name,
        "preset": args.model,
        "package_version": __version__,
        "train_rows": args.train_rows,
        "train_seconds": round(train_sec, 2),
        "submission_sha256": sub_hash,
        **train_summary,
    }
    (args.out_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    meta = {
        "experiment": args.out_dir.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "seed": RANDOM_STATE,
        "submission_sha256": sub_hash,
    }
    (args.out_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("wrote %s eval_f1=%.4f", sub_path, metrics.get("eval_macro_f1", 0))
    return 0


if __name__ == "__main__":
    sys.exit(main())