from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from product_subject.config import RANDOM_STATE

logger = logging.getLogger(__name__)


class TextLabelDataset(Dataset):
    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer: Any,
        max_length: int,
    ) -> None:
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        enc["labels"] = self.labels[idx]
        return enc


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        if self.class_weights is not None:
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
            loss = loss_fct(logits, labels)
        else:
            loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred: Any) -> dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"macro_f1": f1_score(labels, preds, average="macro", zero_division=0)}


def make_group_split(
    df: pd.DataFrame,
    *,
    val_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """One group-aware holdout split for early stopping."""
    n_splits = max(2, int(round(1.0 / val_fraction)))
    groups = df["text_norm"].to_numpy()
    y = df["subject_id"].to_numpy()
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    tr, va = next(cv.split(df, y, groups=groups))
    return tr, va


@dataclass
class EncoderTrainConfig:
    model_name: str
    max_length: int = 96
    batch_size: int = 32
    epochs: int = 8
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    val_fraction: float = 0.1
    fp16: bool = False
    gradient_accumulation_steps: int = 1


def train_encoder(
    train_df: pd.DataFrame,
    out_dir: Any,
    label2id: dict[str, int],
    id2label: dict[int, str],
    class_weights: np.ndarray,
    cfg: EncoderTrainConfig,
) -> tuple[Any, Any, dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device=%s model=%s", device, cfg.model_name)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=False,
    )
    model.to(device)

    tr_idx, va_idx = make_group_split(
        train_df, val_fraction=cfg.val_fraction, seed=RANDOM_STATE
    )
    tr_df = train_df.iloc[tr_idx]
    va_df = train_df.iloc[va_idx]

    train_ds = TextLabelDataset(
        tr_df["text_norm"].tolist(),
        tr_df["subject_id"].tolist(),
        tokenizer,
        cfg.max_length,
    )
    val_ds = TextLabelDataset(
        va_df["text_norm"].tolist(),
        va_df["subject_id"].tolist(),
        tokenizer,
        cfg.max_length,
    )

    cw = torch.tensor(class_weights, dtype=torch.float32, device=device)

    steps_per_epoch = max(1, (len(train_ds) + cfg.batch_size - 1) // cfg.batch_size)
    warmup_steps = max(1, int(cfg.epochs * steps_per_epoch * cfg.warmup_ratio))

    args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size * 2,
        num_train_epochs=cfg.epochs,
        weight_decay=cfg.weight_decay,
        warmup_steps=warmup_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=50,
        fp16=cfg.fp16 and torch.cuda.is_available(),
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        report_to=[],
        save_total_limit=2,
        seed=RANDOM_STATE,
        dataloader_num_workers=0,
    )

    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
        class_weights=cw,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    train_result = trainer.train()
    eval_metrics = trainer.evaluate()

    model.save_pretrained(out_dir / "model")
    tokenizer.save_pretrained(out_dir / "model")

    summary = {
        "train_loss": train_result.training_loss,
        "eval_macro_f1": float(eval_metrics.get("eval_macro_f1", 0)),
        "eval_loss": float(eval_metrics.get("eval_loss", 0)),
        "n_train": len(train_ds),
        "n_val": len(val_ds),
    }
    logger.info("encoder train done eval_macro_f1=%.5f", summary["eval_macro_f1"])
    return model, tokenizer, summary


@torch.inference_mode()
def predict_texts(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    *,
    max_length: int,
    batch_size: int,
) -> np.ndarray:
    device = next(model.parameters()).device
    model.eval()
    all_ids: list[int] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        enc = tokenizer(
            batch,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits
        all_ids.extend(logits.argmax(-1).cpu().tolist())
    return np.array(all_ids, dtype=np.int64)