"""EDA and weak-label table for product attribute NER."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from product_ner.config import DATA_DIR, RANDOM_STATE  # noqa: E402
from product_ner.data import SLOTS, load_frames  # noqa: E402
from product_ner.gazetteer import Gazetteer, slots_to_json  # noqa: E402
from product_ner.text import normalize_text  # noqa: E402

FIG_DIR = ROOT / "notebooks" / "eda"
WEAK_PATH = DATA_DIR / "weak_train.csv"


def _lengths(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["title_len"] = out["title"].str.len()
    out["description_len"] = out["description"].str.len()
    out["description_empty"] = out["description"].str.strip().eq("")
    return out


def _bar(labels: list[str], values: list[float], path: Path, title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, values, color="#3d5a80")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    train, test, _sample, vocabs = load_frames(DATA_DIR)
    train_s = _lengths(train)
    test_s = _lengths(test)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(train_s["title_len"], bins=30, alpha=0.7, label="title")
    ax.hist(train_s["description_len"].clip(upper=1500), bins=30, alpha=0.7, label="description (clip 1500)")
    ax.set_title("Длина текста, train")
    ax.set_xlabel("символы")
    ax.set_ylabel("карточки")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "length_hist.png", dpi=120)
    plt.close(fig)

    gazetteer = Gazetteer(vocabs)
    records = []
    for row in train.itertuples(index=False):
        slots = gazetteer.extract(row.title, row.description, canonical=True)
        records.append(
            {
                "id": row.id,
                "title_norm": normalize_text(row.title),
                "description_norm": normalize_text(row.description),
                **slots_to_json(slots),
            }
        )
    weak = pd.DataFrame(records)
    parsed = {slot: weak[slot].map(json.loads) for slot in SLOTS}
    fill = {slot: float(parsed[slot].map(bool).mean()) for slot in SLOTS}
    _bar(
        list(fill),
        [fill[slot] for slot in SLOTS],
        FIG_DIR / "slot_fill_train.png",
        "Доля карточек train со слабой меткой слота",
        "доля",
    )

    color_counts = parsed["colors"].explode().dropna().value_counts()
    fig, ax = plt.subplots(figsize=(8, 4))
    color_counts.iloc[::-1].plot.barh(ax=ax, color="#3d5a80")
    ax.set_title("Слабые метки цвета на train")
    ax.set_xlabel("карточки")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "color_counts.png", dpi=120)
    plt.close(fig)

    presence = pd.DataFrame({slot: parsed[slot].map(bool).astype(int) for slot in SLOTS})
    corr = presence.corr(method="pearson")
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(corr.to_numpy(), vmin=-0.2, vmax=1, cmap="Blues")
    ax.set_xticks(range(4), SLOTS, rotation=30, ha="right")
    ax.set_yticks(range(4), SLOTS)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center")
    ax.set_title("Корреляция наличия слотов (слабые метки)")
    fig.colorbar(image, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "slot_corr.png", dpi=120)
    plt.close(fig)

    rng = np.random.default_rng(RANDOM_STATE)
    weak["split"] = np.where(rng.random(len(weak)) < 0.2, "holdout", "train")
    weak.to_csv(WEAK_PATH, index=False)

    summary = {
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_description_empty": int(train_s["description_empty"].sum()),
        "test_description_empty": int(test_s["description_empty"].sum()),
        "title_empty": int(train_s["title"].str.strip().eq("").sum()),
        "title_len": train_s["title_len"].quantile([0.5, 0.95, 0.99]).round(1).to_dict(),
        "description_len": train_s["description_len"].quantile([0.5, 0.95, 0.99]).round(1).to_dict(),
        "description_len_max": int(train_s["description_len"].max()),
        "vocab_sizes": {slot: len(values) for slot, values in vocabs.items()},
        "weak_fill_train": {slot: round(value, 4) for slot, value in fill.items()},
        "weak_holdout_rows": int((weak["split"] == "holdout").sum()),
        "slot_corr": corr.round(3).to_dict(),
        "color_top": color_counts.head(8).to_dict(),
    }
    (FIG_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
