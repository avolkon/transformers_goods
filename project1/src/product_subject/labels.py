from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def fit_label_encoder(subjects: pd.Series) -> tuple[dict[str, int], dict[int, str], np.ndarray]:
    labels = sorted(subjects.unique())
    label2id = {s: i for i, s in enumerate(labels)}
    id2label = {i: s for s, i in label2id.items()}
    counts = subjects.value_counts().reindex(labels).to_numpy(dtype=np.float64)
    # sqrt inverse-frequency: milder than 1/n so singletons do not dominate loss
    weights = 1.0 / np.sqrt(counts)
    weights = weights / weights.mean()
    return label2id, id2label, weights.astype(np.float32)


def save_label_map(
    label2id: dict[str, int],
    id2label: dict[int, str],
    path: Path,
) -> None:
    payload = {
        "label2id": label2id,
        "id2label": {str(k): v for k, v in id2label.items()},
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")