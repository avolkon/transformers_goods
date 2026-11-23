"""Token NER: gazetteer vs LogReg vs LinearSVC, small hyperparameter grid.

Weak labels are the v2 strict gazetteer. Holdout F1 is agreement with that
dictionary, not the hidden Kaggle gold. Selection key is holdout exact-match
of the four slots, then macro-F1.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from product_ner.config import DATA_DIR, RANDOM_STATE  # noqa: E402
from product_ner.data import SLOTS, load_frames  # noqa: E402
from product_ner.gazetteer import Gazetteer, slots_to_json  # noqa: E402
from product_ner.submission import validate_submission, write_submission  # noqa: E402

TOKEN = __import__("re").compile(r"[0-9a-zа-я]+")
OUT = ROOT / "submissions" / "token_tune"


def tokenize(text: str) -> list[tuple[int, int, str]]:
    return [(m.start(), m.end(), m.group(0)) for m in TOKEN.finditer(text)]


def features(tokens: list[tuple[int, int, str]], index: int) -> dict[str, str]:
    word = tokens[index][2]
    prev = tokens[index - 1][2] if index else "<BOS>"
    nxt = tokens[index + 1][2] if index + 1 < len(tokens) else "<EOS>"
    return {
        "w": word,
        "pre": prev,
        "nxt": nxt,
        "suf3": word[-3:],
        "pre3": word[:3],
        "digit": "1" if word.isdigit() else "0",
        "short": "1" if len(word) <= 2 else "0",
    }


def label_token(start: int, end: int, spans: list[tuple[int, int, str, str]]) -> str:
    best = "O"
    best_len = -1
    for span_start, span_end, slot, label in spans:
        if start < span_end and end > span_start and span_end - span_start > best_len:
            best = f"{slot}|{label}"
            best_len = span_end - span_start
    return best


def document_examples(gazetteer: Gazetteer, title: str, description: str) -> tuple[list[dict[str, str]], list[str], dict[str, set[str]]]:
    text, spans = gazetteer.strict_spans(title, description)
    tokens = tokenize(text)
    xs = [features(tokens, i) for i in range(len(tokens))]
    ys = [label_token(start, end, spans) for start, end, _word in tokens]
    gold = {slot: set(gazetteer.extract(title, description, strict=True)[slot]) for slot in SLOTS}
    return xs, ys, gold


def prf(gold: list[dict[str, set[str]]], pred: list[dict[str, set[str]]]) -> dict[str, dict[str, float]]:
    report: dict[str, dict[str, float]] = {}
    for slot in SLOTS:
        tp = fp = fn = 0
        for truth, guess in zip(gold, pred, strict=True):
            tp += len(truth[slot] & guess[slot])
            fp += len(guess[slot] - truth[slot])
            fn += len(truth[slot] - guess[slot])
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        report[slot] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
    return report


def exact_match(gold: list[dict[str, set[str]]], pred: list[dict[str, set[str]]]) -> float:
    hits = 0
    for truth, guess in zip(gold, pred, strict=True):
        if all(truth[slot] == guess[slot] for slot in SLOTS):
            hits += 1
    return hits / len(gold) if gold else 0.0


def decode(tokens: list[tuple[int, int, str]], labels: np.ndarray, gazetteer: Gazetteer) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {slot: [] for slot in SLOTS}
    for (_start, _end, _word), raw in zip(tokens, labels, strict=True):
        if raw == "O" or "|" not in str(raw):
            continue
        slot, label = str(raw).split("|", 1)
        if slot in found and label in gazetteer.order[slot]:
            found[slot].append(label)
    return {slot: gazetteer._in_vocab_order(slot, labels) for slot, labels in found.items()}


def predict_docs(model, vectorizer, docs: list[tuple[str, str]], gazetteer: Gazetteer) -> list[dict[str, set[str]]]:
    from product_ner.text import normalize_text

    predicted: list[dict[str, set[str]]] = []
    for title, description in docs:
        tokens = tokenize(normalize_text(f"{title or ''} {description or ''}"))
        if not tokens:
            predicted.append({slot: set() for slot in SLOTS})
            continue
        matrix = vectorizer.transform([features(tokens, i) for i in range(len(tokens))])
        labels = model.predict(matrix)
        slots = decode(tokens, labels, gazetteer)
        predicted.append({slot: set(slots[slot]) for slot in SLOTS})
    return predicted


def fit_model(name: str, params: dict, x_train, y_train):
    if name == "logreg":
        clf = SGDClassifier(
            loss="log_loss",
            alpha=params["alpha"],
            class_weight=params.get("class_weight"),
            max_iter=40,
            random_state=RANDOM_STATE,
        )
    else:
        clf = LinearSVC(
            C=params["C"],
            class_weight=params.get("class_weight"),
            dual=False,
            max_iter=2000,
            random_state=RANDOM_STATE,
        )
    clf.fit(x_train, y_train)
    return clf


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    train, test, _sample, vocabs = load_frames(DATA_DIR)
    gazetteer = Gazetteer(vocabs)
    split = pd.read_csv(DATA_DIR / "weak_train.csv", dtype=str, usecols=["id", "split"])
    holdout_ids = set(split.loc[split["split"] == "holdout", "id"])

    train_x: list[dict[str, str]] = []
    train_y: list[str] = []
    hold_docs: list[tuple[str, str]] = []
    hold_gold: list[dict[str, set[str]]] = []
    started = time.perf_counter()
    for row in train.itertuples(index=False):
        xs, ys, gold = document_examples(gazetteer, row.title, row.description)
        if row.id in holdout_ids:
            hold_docs.append((row.title, row.description))
            hold_gold.append(gold)
            continue
        train_x.extend(xs)
        train_y.extend(ys)
    print(f"labeled tokens={len(train_y)} holdout_docs={len(hold_docs)} sec={time.perf_counter()-started:.1f}", flush=True)

    y_arr = np.array(train_y)
    pos = np.flatnonzero(y_arr != "O")
    neg = np.flatnonzero(y_arr == "O")
    rng = np.random.default_rng(RANDOM_STATE)
    keep_neg = rng.choice(neg, size=min(len(neg), 4 * len(pos)), replace=False)
    keep = np.concatenate([pos, keep_neg])
    rng.shuffle(keep)
    train_x = [train_x[i] for i in keep]
    train_y = [train_y[i] for i in keep]
    print(f"fit rows={len(train_y)} positives={len(pos)}", flush=True)

    vectorizer = DictVectorizer(sparse=True)
    matrix = vectorizer.fit_transform(train_x)
    print(f"features={matrix.shape[1]}", flush=True)

    grid = [
        ("logreg", {"alpha": alpha, "class_weight": weight})
        for alpha in (1e-5, 1e-4, 1e-3)
        for weight in (None, "balanced")
    ] + [
        ("linearsvc", {"C": c, "class_weight": None})
        for c in (0.5, 1.0, 2.0)
    ]

    rows = []
    fitted = {}
    for name, params in grid:
        t0 = time.perf_counter()
        model = fit_model(name, params, matrix, train_y)
        pred = predict_docs(model, vectorizer, hold_docs, gazetteer)
        per_slot = prf(hold_gold, pred)
        macro = sum(per_slot[slot]["f1"] for slot in SLOTS) / len(SLOTS)
        exact = exact_match(hold_gold, pred)
        param_name = "alpha" if name == "logreg" else "C"
        key = f"{name}|{param_name}={params[param_name]}|w={params['class_weight']}"
        fitted[key] = model
        rows.append(
            {
                "model": name,
                "C": params.get("C"),
                "alpha": params.get("alpha"),
                "class_weight": params["class_weight"],
                "holdout_exact_match": round(exact, 4),
                "macro_f1": round(macro, 4),
                "per_slot": per_slot,
                "fit_seconds": round(time.perf_counter() - t0, 1),
            }
        )
        print(
            f"{key} exact={exact:.4f} macro_f1={macro:.4f} sec={rows[-1]['fit_seconds']}",
            flush=True,
        )

    gazetteer_pred = [hold_gold[i] for i in range(len(hold_gold))]
    gazetteer_row = {
        "model": "gazetteer_v2",
        "C": None,
        "class_weight": None,
        "holdout_exact_match": 1.0,
        "macro_f1": 1.0,
        "per_slot": prf(hold_gold, gazetteer_pred),
        "note": "weak labels are this gazetteer, so holdout F1 is 1 by construction",
    }
    best = max(rows, key=lambda item: (item["holdout_exact_match"], item["macro_f1"]))
    default = next(
        item for item in rows if item["model"] == "logreg" and item["alpha"] == 1e-4 and item["class_weight"] is None
    )

    def test_frame(model) -> pd.DataFrame:
        records = []
        for row in test.itertuples(index=False):
            pred = predict_docs(model, vectorizer, [(row.title, row.description)], gazetteer)[0]
            ordered = {slot: gazetteer._in_vocab_order(slot, list(pred[slot])) for slot in SLOTS}
            records.append({"id": row.id, **slots_to_json(ordered)})
        frame = pd.DataFrame(records, columns=["id", *SLOTS])
        validate_submission(frame, test, vocabs)
        return frame

    best_param = "alpha" if best["model"] == "logreg" else "C"
    best_value = best["alpha"] if best["model"] == "logreg" else best["C"]
    best_key = f"{best['model']}|{best_param}={best_value}|w={best['class_weight']}"
    default_key = "logreg|alpha=0.0001|w=None"
    print("predict test best and default", flush=True)
    best_sub = test_frame(fitted[best_key])
    default_sub = test_frame(fitted[default_key])
    v2 = pd.read_csv(ROOT / "submissions" / "gazetteer_v2" / "submission.csv", dtype=str, keep_default_na=False)

    def n_diff(left: pd.DataFrame, right: pd.DataFrame) -> int:
        changed = 0
        for slot in SLOTS:
            changed_rows = left[slot].to_numpy() != right[slot].to_numpy()
            # row differs if any slot differs; count rows not cells
            if slot == SLOTS[0]:
                mask = changed_rows
            else:
                mask = mask | changed_rows
        return int(mask.sum())

    diff_best_default = n_diff(best_sub, default_sub)
    diff_best_v2 = n_diff(best_sub, v2)
    diff_default_v2 = n_diff(default_sub, v2)
    sub_path = OUT / "submission.csv"
    write_submission(best_sub, sub_path)
    report = {
        "weak_label_source": "gazetteer strict v2",
        "holdout_docs": len(hold_docs),
        "selection": "max holdout exact-match, then macro-F1",
        "gazetteer_v2": gazetteer_row,
        "grid": rows,
        "best": {k: best[k] for k in ("model", "C", "class_weight", "holdout_exact_match", "macro_f1")},
        "default_logreg_C1": {
            "holdout_exact_match": default["holdout_exact_match"],
            "macro_f1": default["macro_f1"],
        },
        "test_rows_best_vs_default": diff_best_default,
        "test_rows_best_vs_v2": diff_best_v2,
        "test_rows_default_vs_v2": diff_default_v2,
        "submission_changes_with_tuning": diff_best_default > 0,
        "tuned_submission_equals_v2": diff_best_v2 == 0,
    }
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in report if k != "grid"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
