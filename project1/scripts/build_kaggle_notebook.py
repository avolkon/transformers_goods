#!/usr/bin/env python
"""Сборка notebooks/kaggle_product_subject.ipynb (Colab → Kaggle Run All)."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "kaggle_product_subject.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


CELLS = [
    md("""# Классификация категории товара по названию

**Соревнование:** [MEPhI: NLP: Classification: Subject Detection](https://www.kaggle.com/competitions/mephi-product-title-subject-detection)  
**Метрика:** F1-macro (`sklearn`, `zero_division=0`)  
**ФИО / группа:** _(заполнить в markdown до публикации; Run All не требует правок кода)_

Этот ноутбук самодостаточный: **Run All** без ручных правок ячеек.

| Файл (правило курса §8) | Что это |
|-------------------------|---------|
| `baseline_submission.csv` | короткий baseline (TF-IDF + LogisticRegression) |
| `submission.csv` | финальная модель (TF-IDF + LinearSVC), тот файл, который сдан на leaderboard (**public 0.90084**) |

Данные: только `train.csv` / `test.csv` соревнования. Внешняя разметка теста, LLM без модели на train и prod-классификатор Naitii **не используются**.
"""),
    md("""**Перед Run All положите CSV в один из путей** (без интерактивного `upload` в ячейках — иначе Run All остановится):

| Среда | Куда класть `train.csv` и `test.csv` |
|--------|--------------------------------------|
| **Kaggle** | Add Data → это соревнование (обычно `/kaggle/input/mephi-product-title-subject-detection/`) |
| **Colab** | `/content/data/train.csv` и `/content/data/test.csv` (Files → загрузить заранее) |
| Локально | `data/` рядом с ноутбуком |
"""),
    code("""# @title Установка зависимостей
import importlib
import subprocess
import sys

for pkg, mod in (
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("scikit-learn", "sklearn"),
    ("matplotlib", "matplotlib"),
):
    try:
        importlib.import_module(mod)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])
print("зависимости OK")"""),
    code("""# @title §1. Импорты и константы
import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

RANDOM_STATE = 42
_WS = re.compile(r"\\s+")


def output_dir() -> Path:
    for p in (Path("/kaggle/working"), Path("/content")):
        if p.exists():
            return p
    return Path(".")


OUT_DIR = output_dir()
BASELINE_PATH = OUT_DIR / "baseline_submission.csv"
SUBMIT_PATH = OUT_DIR / "submission.csv"
print("OUT_DIR =", OUT_DIR.resolve())"""),
    code("""# @title §2. Загрузка train.csv / test.csv
def find_csv(name: str) -> Path:
    explicit = [
        Path("/kaggle/input/mephi-product-title-subject-detection") / name,
        Path("/content/data") / name,
        Path("/content") / name,
        Path("data") / name,
        Path("../data") / name,
        Path(name),
    ]
    kaggle_root = Path("/kaggle/input")
    if kaggle_root.exists():
        explicit.extend(sorted(kaggle_root.glob(f"**/{name}")))
    for path in explicit:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"{name} не найден. Положите файл в /content/data/ (Colab) "
        "или подключите данные соревнования (Kaggle)."
    )


def normalize_text(text) -> str:
    if not isinstance(text, str):
        text = "" if text is None or (isinstance(text, float) and np.isnan(text)) else str(text)
    return _WS.sub(" ", text.strip().lower().replace("ё", "е"))


train_path = find_csv("train.csv")
test_path = find_csv("test.csv")
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

need_train, need_test = {"id", "text", "subject"}, {"id", "text"}
if not need_train.issubset(train.columns):
    raise ValueError(f"train.csv колонки {list(train.columns)}")
if not need_test.issubset(test.columns):
    raise ValueError(f"test.csv колонки {list(test.columns)}")

train = train.copy()
test = test.copy()
train["text_norm"] = train["text"].map(normalize_text)
test["text_norm"] = test["text"].map(normalize_text)

overlap_raw = set(train["text"]) & set(test["text"])
overlap_norm = set(train["text_norm"]) & set(test["text_norm"])
print("train", train_path)
print("test ", test_path)
print("строк train/test:", len(train), len(test), "| классов:", train["subject"].nunique())
print("пересечение raw text train∩test:", len(overlap_raw))
print("пересечение text_norm train∩test:", len(overlap_norm))
if overlap_raw:
    raise ValueError(f"утечка raw text: {len(overlap_raw)}")"""),
    md("""**Вывод: загрузка.** Пути и размеры — из `read_csv`. Утверждение «нет пересечения сырых названий» следует **не** из числа строк, а из `len(set(train["text"]) & set(test["text"])) == 0` (ячейка печатает это число и падает, если оно > 0). После нормализации (`ё→е`, lower) пересечения `text_norm` могут быть — это не то же самое, что raw `text`.
"""),
    code("""# @title §3. EDA: качество данных, распределения, визуализации
print("=== пропуски (NaN) ===")
miss = pd.DataFrame({"train": train.isna().sum(), "test": test.isna().sum()})
print(miss.to_string())
if miss.to_numpy().max() > 0:
    print("  → числовые NaN: заполнение не требуется, см. normalize_text для не-строк.")

text_tr = train["text"].astype(str)
text_te = test["text"].astype(str)
empty_tr = int((text_tr.str.strip() == "").sum())
empty_te = int((text_te.str.strip() == "").sum())
print("\\n=== пустые названия (после str) ===")
print("  train:", empty_tr, "| test:", empty_te)
if empty_tr or empty_te:
  print("  → пустые строки попадут в text_norm как ''; в модели остаются как есть.")

n_classes = int(train["subject"].nunique())
vc = train["subject"].value_counts()
text_lens = text_tr.str.len()
print("\\n=== структура train ===")
print("  классов:", n_classes)
print("  строк:", len(train), "| unique text_norm:", train["text_norm"].nunique())
print("  дубликатов text_norm (лишние строки):", len(train) - train["text_norm"].nunique())
print("  singleton-классов (1 строка в train):", int((vc == 1).sum()))
print(
    "  длина text (символы): median",
    float(text_lens.median()),
    "| p95",
    float(text_lens.quantile(0.95)),
    "| p99",
    float(text_lens.quantile(0.99)),
    "| max",
    int(text_lens.max()),
)
print(
    "  интерпретация длины: экстремально длинные названия не отрезаем — они несут сигнал; "
    "отдельной обрезки «выбросов» нет."
)

print("\\n=== признаки для моделирования ===")
print("  id — идентификатор, в обучение и TF-IDF не входит.")
print("  subject — целевая метка (только train).")
print("  text → text_norm — единственный вход модели (нормализация в §2).")
print("  корреляционная матрица числовых признаков не строится: нет табличных фич, кроме текста.")

fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
axes[0].hist(text_lens, bins=40, color="steelblue", edgecolor="white", linewidth=0.3)
axes[0].set_title("Длина названий (train)")
axes[0].set_xlabel("символов")
axes[0].set_ylabel("число строк")

per_class = vc.to_numpy()
axes[1].hist(per_class, bins=50, color="coral", edgecolor="white", linewidth=0.3)
axes[1].set_yscale("log")
axes[1].set_title("Примеров на класс (train, log по оси Y)")
axes[1].set_xlabel("число строк с этим subject")
axes[1].set_ylabel("число классов (log)")

plt.tight_layout()
try:
    get_ipython()
    plt.show()
except NameError:
    plt.close()

print("\\nтоп-5 классов по частоте:")
print(vc.head(5).to_string())"""),
    md("""**Вывод: EDA (для моделей).**

- **Пропуски и пустые значения** проверены явно (`isna`, пустой `text`); для не-строк используется нормализация в §2. Отдельного imputation нет — признаков с массовыми NaN нет.
- **Выбросы:** для текста смотрим хвост **длины** (p95/p99); агрессивная обрезка не применяется — длинные названия остаются в обучении.
- **Нерелевантные поля:** `id` не используется как признак; таргет `subject` не подаётся в векторизатор. В модель идёт только **`text_norm`**.
- **Корреляции** в классическом смысле не считаются (одномодальная NLP-задача); вместо этого — **дисбаланс классов** (гистограмма «примеров на класс», log-шкала) и **дубликаты** названий → решение обучаться на **уникальных** `text_norm` (§4).
- **Feature engineering:** нормализация текста, TF-IDF (word + char) — в §5–§6; выводы EDA объясняют выбор `FIT`, `class_weight='balanced'` и риск macro-F1 на редких классах.
"""),
    code("""# @title §4. Общие функции (fit-набор, проверка CSV)
def unique_text_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates(subset=["text_norm"], keep="first")


def validate_submission(submission: pd.DataFrame, test_df: pd.DataFrame, labels: set) -> None:
    if list(submission.columns) != ["id", "subject"]:
        raise ValueError(f"колонки {list(submission.columns)}")
    if len(submission) != len(test_df) or set(submission["id"]) != set(test_df["id"]):
        raise ValueError("id не совпадают с test")
    extra = set(submission["subject"]) - labels
    if extra:
        raise ValueError(f"метки вне train: {list(extra)[:5]}")


LABELS = set(train["subject"].unique())
FIT = unique_text_frame(train)
print("строк для обучения (unique text):", len(FIT), "из", len(train))"""),
    md("""## Baseline (example submit)

TF-IDF (word 1–2 + char 3–5) + LogisticRegression, `class_weight='balanced'`. Пишет **`baseline_submission.csv`** — обязательный example submit по правилам курса.
"""),
    code("""# @title §5. Baseline → baseline_submission.csv
def make_tfidf(word_n, char_n, max_w, max_c, min_df) -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=word_n,
                    min_df=min_df,
                    max_df=0.98,
                    max_features=max_w,
                    sublinear_tf=True,
                ),
                "text_norm",
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=char_n,
                    min_df=min_df,
                    max_df=0.98,
                    max_features=max_c,
                    sublinear_tf=True,
                ),
                "text_norm",
            ),
        ],
        sparse_threshold=0.3,
    )


baseline = Pipeline(
    [
        ("features", make_tfidf((1, 2), (3, 5), 80_000, 80_000, 2)),
        (
            "clf",
            LogisticRegression(
                C=2.0,
                max_iter=300,
                class_weight="balanced",
                solver="lbfgs",
                random_state=RANDOM_STATE,
            ),
        ),
    ]
)
BASELINE_CV_FOLDS = 3
_cv = StratifiedKFold(n_splits=BASELINE_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
BASELINE_CV_SCORES = cross_val_score(
    baseline,
    FIT[["text_norm"]],
    FIT["subject"].to_numpy(),
    cv=_cv,
    scoring="f1_macro",
    n_jobs=-1,
)
print("=== CV baseline (macro-F1 на FIT) ===")
print("  fold:", [round(float(x), 5) for x in BASELINE_CV_SCORES])
print(
    "  mean ± std:",
    round(float(BASELINE_CV_SCORES.mean()), 5),
    "±",
    round(float(BASELINE_CV_SCORES.std()), 5),
)

baseline.fit(FIT[["text_norm"]], FIT["subject"].to_numpy())
pred_fit = baseline.predict(FIT[["text_norm"]])
BASELINE_FIT_F1 = f1_score(FIT["subject"], pred_fit, average="macro", zero_division=0)
print(
    "macro-F1 на FIT (обучение, завышена):",
    round(BASELINE_FIT_F1, 5),
)
base_pred = baseline.predict(test[["text_norm"]])
print("уникальных subject в pred на test:", int(pd.Series(base_pred).nunique()))
baseline_sub = pd.DataFrame({"id": test["id"], "subject": base_pred})
validate_submission(baseline_sub, test, LABELS)
baseline_sub.to_csv(BASELINE_PATH, index=False)
print("записан", BASELINE_PATH, "строк", len(baseline_sub))"""),
    md("""## Финальная модель (файл для Submit Predictions)

**LinearSVC**, TF-IDF word 1–2 (`max_features=120000`) + char_wb 2–6 (`150000`), `min_df=1`, `C=3`, обучение на уникальных названиях. Это модель с **public 0.90084**. Пишет **`submission.csv`**.
"""),
    code("""# @title §6. Финальная модель → submission.csv
import time

DO_CV_FOLDS = 3  # 0 — пропустить CV (быстрый прогон)

FINAL_CFG = {
    "word_ngram": (1, 2),
    "char_ngram": (2, 6),
    "max_features_word": 120_000,
    "max_features_char": 150_000,
    "min_df": 1,
    "C": 3.0,
}

print("=== §6. Финальная модель: конфиг ===")
print("  векторизация: word", FINAL_CFG["word_ngram"], "char_wb", FINAL_CFG["char_ngram"])
print("  max_features:", FINAL_CFG["max_features_word"], "/", FINAL_CFG["max_features_char"])
print("  min_df:", FINAL_CFG["min_df"], "| C:", FINAL_CFG["C"])
print("  clf: LinearSVC, class_weight=balanced, dual=auto, max_iter=4000")
print("  train: FIT (unique text_norm):", len(FIT), "строк,", FIT["subject"].nunique(), "классов")
subj_fit = FIT["subject"].value_counts()
print(
    "  примеров на класс (FIT): min",
    int(subj_fit.min()),
    "median",
    int(subj_fit.median()),
    "max",
    int(subj_fit.max()),
)

final_model = Pipeline(
    [
        (
            "features",
            make_tfidf(
                FINAL_CFG["word_ngram"],
                FINAL_CFG["char_ngram"],
                FINAL_CFG["max_features_word"],
                FINAL_CFG["max_features_char"],
                FINAL_CFG["min_df"],
            ),
        ),
        (
            "clf",
            LinearSVC(
                C=FINAL_CFG["C"],
                class_weight="balanced",
                dual="auto",
                max_iter=4000,
                random_state=RANDOM_STATE,
            ),
        ),
    ]
)

if DO_CV_FOLDS:
    print("\\n=== кросс-валидация macro-F1 на FIT ===")
    cv = StratifiedKFold(n_splits=DO_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    t0 = time.perf_counter()
    cv_scores = cross_val_score(
        final_model,
        FIT[["text_norm"]],
        FIT["subject"].to_numpy(),
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
    )
    FINAL_CV_SCORES = cv_scores
    print("  fold:", [round(float(x), 5) for x in FINAL_CV_SCORES])
    print("  mean ± std:", round(float(FINAL_CV_SCORES.mean()), 5), "±", round(float(FINAL_CV_SCORES.std()), 5))
    print("  время CV, сек:", round(time.perf_counter() - t0, 1))
else:
    FINAL_CV_SCORES = None

print("\\n=== обучение на всём FIT ===")
t0 = time.perf_counter()
final_model.fit(FIT[["text_norm"]], FIT["subject"].to_numpy())
print("  время fit, сек:", round(time.perf_counter() - t0, 1))

feat = final_model.named_steps["features"]
X_fit = feat.transform(FIT[["text_norm"]])
n_rows, n_cols = X_fit.shape
nnz = int(X_fit.nnz)
density = nnz / max(n_rows * n_cols, 1)
word_v = len(feat.named_transformers_["word"].vocabulary_)
char_v = len(feat.named_transformers_["char"].vocabulary_)
print("\\n=== матрица TF-IDF (train) ===")
print("  shape:", f"{n_rows} × {n_cols}")
print("  ненулевых:", nnz, "| доля ненулевых в матрице:", f"{density:.6f}")
print("  словарь word:", word_v, "| словарь char:", char_v)

clf = final_model.named_steps["clf"]
print("\\n=== LinearSVC (после fit) ===")
print("  число классов:", len(clf.classes_))
if hasattr(clf, "coef_") and clf.coef_ is not None:
    print("  coef_.shape:", clf.coef_.shape)
if hasattr(clf, "n_iter_"):
    ni = np.atleast_1d(clf.n_iter_)
    print(
        "  n_iter_ min / median / max:",
        int(ni.min()),
        int(np.median(ni)),
        int(ni.max()),
    )

pred_fit = final_model.predict(FIT[["text_norm"]])
FINAL_FIT_F1 = f1_score(FIT["subject"], pred_fit, average="macro", zero_division=0)
f1_final_fit = FINAL_FIT_F1
f1_base_fit = f1_score(
    FIT["subject"],
    baseline.predict(FIT[["text_norm"]]),
    average="macro",
    zero_division=0,
)
print("\\n=== macro-F1 на FIT (те же строки, что при обучении) ===")
print("  baseline (LogReg):", round(f1_base_fit, 5))
print("  final (LinearSVC):", round(f1_final_fit, 5))
print("  прирост final − baseline:", round(f1_final_fit - f1_base_fit, 5))

t0 = time.perf_counter()
final_pred = final_model.predict(test[["text_norm"]])
print("\\n=== предсказание test ===")
print("  время predict, сек:", round(time.perf_counter() - t0, 2))
pred_s = pd.Series(final_pred, name="subject")
print("  уникальных subject в pred:", int(pred_s.nunique()))
vc = pred_s.value_counts()
print("  топ-5 классов по числу строк:", {str(k): int(v) for k, v in vc.head().items()})
print("  singleton-классов в pred (ровно 1 строка):", int((vc == 1).sum()))

submission = pd.DataFrame({"id": test["id"], "subject": final_pred})
validate_submission(submission, test, LABELS)
submission.to_csv(SUBMIT_PATH, index=False)
print("\\n=== сабмит ===")
print("  записан", SUBMIT_PATH, "строк", len(submission))

agree = baseline_sub["subject"].to_numpy() == submission["subject"].to_numpy()
print("\\n=== baseline vs final (test) ===")
print("  доля совпадающих меток:", round(float(agree.mean()), 6))
print("  строк с расхождением:", int((~agree).sum()))
if (~agree).any():
    chg_ix = np.where(~agree)[0]
    chg = pd.DataFrame(
        {
            "baseline": baseline_sub["subject"].iloc[chg_ix].to_numpy(),
            "final": submission["subject"].iloc[chg_ix].to_numpy(),
        }
    )
    pairs = (
        chg.groupby(["baseline", "final"], as_index=False)
        .size()
        .rename(columns={"size": "n"})
        .sort_values("n", ascending=False)
    )
    print("  топ-10 переходов baseline → final:")
    for _, row in pairs.head(10).iterrows():
        print(f"    {row['baseline']} → {row['final']}: {int(row['n'])} шт.")
    print("  примеры расхождений (до 5):")
    for i in chg_ix[:5]:
        t = test["text_norm"].iloc[i]
        if len(t) > 55:
            t = t[:52] + "..."
        print(
            f"    id={test['id'].iloc[i]} | "
            f"base={baseline_sub['subject'].iloc[i]} → final={submission['subject'].iloc[i]} | {t!r}"
        )"""),
    code("""# @title §6.1 Сравнение моделей (метрики)
PUBLIC_LB = {
    "baseline": 0.87394,
    "final": 0.90084,
}


def _cv_cell(scores) -> str:
    if scores is None:
        return "—"
    return f"{scores.mean():.5f} ± {scores.std():.5f}"


agree_rate = float((baseline_sub["subject"].to_numpy() == submission["subject"].to_numpy()).mean())
rows = [
    [
        "Baseline (example)",
        "TF-IDF word 1–2, char 3–5; LogReg C=2",
        _cv_cell(BASELINE_CV_SCORES),
        f"{BASELINE_FIT_F1:.5f}",
        f"{PUBLIC_LB['baseline']:.5f}",
    ],
    [
        "Финал (сабмит)",
        "TF-IDF word 1–2, char 2–6, min_df=1; LinearSVC C=3",
        _cv_cell(FINAL_CV_SCORES),
        f"{FINAL_FIT_F1:.5f}",
        f"{PUBLIC_LB['final']:.5f}",
    ],
]
cols = ["Модель", "Признаки / clf", "CV macro-F1 (3-fold FIT)", "macro-F1 на FIT", "Public LB"]
compare = pd.DataFrame(rows, columns=cols)
print(compare.to_string(index=False))
print("\\nдоля совпадающих меток на test (baseline vs final):", round(agree_rate, 6))
print(
    "\\nЛокальный перебор (вне ноутбука): char 2–6, min_df=1, LinearSVC C=3 на unique_text — "
    "лучший holdout ~0.57; обучение на всех 31k строк без дедупа ~0.40."
)"""),
    md("""## Сравнение моделей

Таблица заполняется в **§6.1** при Run All (3-fold **StratifiedKFold**, `random_state=42`, macro-F1 на **FIT**).

| Модель | CV macro-F1 (ориентир) | macro-F1 на FIT | Public LB |
|--------|-------------------------|-----------------|-----------|
| Baseline | ~0.466 ± 0.013 | ~0.929 | ~0.874 |
| Финал | ~0.520 ± 0.024 | ~0.999 | ~0.901 |

**Вывод:** финал лучше по CV и на лидерборде; macro-F1 на FIT у финала завышен (переобучение). Baseline сохранён как example submit.
"""),
    code("""# @title §7. Проверка файлов сдачи
def peek(path: Path) -> None:
    df = pd.read_csv(path)
    validate_submission(df, test, LABELS)
    print(path.name, "OK", df.shape, "пример:", df.iloc[0].to_dict())


peek(BASELINE_PATH)
peek(SUBMIT_PATH)
print("готово: baseline_submission.csv (example) и submission.csv (финал / leaderboard 0.90084)")"""),
    md("""## Сдача

1. **Colab:** скачайте `baseline_submission.csv` и `submission.csv` из `/content/` (или текущей папки). Убедитесь, что Run All прошёл без ошибок.
2. **Kaggle:** File → Save Version → **Save & Run All**; подключите данные соревнования. В Output будут оба CSV.
3. На странице соревнования Submit Predictions загрузите **`submission.csv`** (финальная модель).
4. Опубликуйте notebook (public / unlisted) и прикрепите к соревнованию.

Лимит: не более **5** сабмитов в день, до **2** final. Секретов и внешних меток теста в ноутбуке нет.
"""),
]


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "cells": CELLS,
    }
    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
