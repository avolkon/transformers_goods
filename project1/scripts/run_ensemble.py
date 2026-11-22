"""Blend TF-IDF and transformer probabilities (OOF-free, simple average)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.sparse import hstack

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from product_subject.config import DATA_DIR  # noqa: E402
from product_subject.data import load_frames  # noqa: E402
from product_subject.submission import validate_submission, write_submission  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--baseline-model", type=Path, default=ROOT / "submissions/baseline/artifacts/tfidf_logreg.joblib")
    parser.add_argument("--encoder-dir", type=Path, default=ROOT / "submissions/encoder_v1/artifacts/model")
    parser.add_argument("--w-encoder", type=float, default=0.65)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "submissions/ensemble_v1")
    args = parser.parse_args()

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    train, test, _ = load_frames(args.data_dir)
    label2id = {s: i for i, s in enumerate(sorted(train["subject"].unique()))}
    id2label = {i: s for s, i in label2id.items()}

    baseline = joblib.load(args.baseline_model)
    bl_proba = baseline.predict_proba(test[["text_norm"]])
    # align columns to label2id order
    if hasattr(baseline.named_steps["clf"], "classes_"):
        clf_classes = list(baseline.named_steps["clf"].classes_)
        bl_full = np.zeros((len(test), len(label2id)), dtype=np.float64)
        for j, c in enumerate(clf_classes):
            bl_full[:, label2id[c]] = bl_proba[:, j]
        bl_proba = bl_full

    tokenizer = AutoTokenizer.from_pretrained(args.encoder_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.encoder_dir)
    model.eval()
    enc_proba = np.zeros((len(test), len(label2id)), dtype=np.float64)
    bs = 32
    texts = test["text_norm"].tolist()
    with torch.inference_mode():
        for i in range(0, len(texts), bs):
            batch = texts[i : i + bs]
            enc = tokenizer(batch, truncation=True, max_length=96, padding=True, return_tensors="pt")
            logits = model(**enc).logits.numpy()
            probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
            enc_proba[i : i + len(batch)] = probs

    w = args.w_encoder
    mix = (1 - w) * bl_proba + w * enc_proba
    pred_ids = mix.argmax(axis=1)
    subjects = [id2label[int(i)] for i in pred_ids]
    submission = pd.DataFrame({"id": test["id"], "subject": subjects})
    validate_submission(submission, test, set(label2id))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sub_path = args.out_dir / "submission.csv"
    sub_hash = write_submission(submission, sub_path)
    meta = {"w_encoder": w, "submission_sha256": sub_hash}
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()