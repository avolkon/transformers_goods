"""Compare legacy (all rows) vs structured (unique_text) baseline."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def run_mode(train_rows: str, tag: str, cv_folds: int = 3) -> dict:
    out = ROOT / "submissions" / f"cmp_{tag}"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_baseline.py"),
        "--train-rows",
        train_rows,
        "--out-dir",
        str(out),
        "--cv-folds",
        str(cv_folds),
    ]
    subprocess.run(cmd, check=True, cwd=ROOT)
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    metrics["out_dir"] = str(out)
    return metrics


def agreement(path_a: Path, path_b: Path) -> float:
    a = pd.read_csv(path_a)
    b = pd.read_csv(path_b)
    merged = a.merge(b, on="id", suffixes=("_a", "_b"))
    return float((merged["subject_a"] == merged["subject_b"]).mean())


def main() -> None:
    legacy_path = ROOT / "submissions" / "baseline_submission.csv"
    m_all = run_mode("all", "all_rows")
    m_uniq = run_mode("unique_text", "unique_text")
    agree = agreement(
        Path(m_all["out_dir"]) / "submission.csv",
        Path(m_uniq["out_dir"]) / "submission.csv",
    )
    report = {
        "legacy_exists": legacy_path.exists(),
        "all_rows_cv_mean": m_all.get("cv", {}).get("macro_f1_mean"),
        "unique_text_cv_mean": m_uniq.get("cv", {}).get("macro_f1_mean"),
        "test_prediction_agreement": agree,
        "fit_seconds_all": m_all.get("fit_seconds"),
        "fit_seconds_unique": m_uniq.get("fit_seconds"),
    }
    if legacy_path.exists():
        legacy = pd.read_csv(legacy_path)
        report["legacy_vs_unique_agreement"] = agreement(
            legacy_path, Path(m_uniq["out_dir"]) / "submission.csv"
        )
        report["legacy_vs_all_agreement"] = agreement(
            legacy_path, Path(m_all["out_dir"]) / "submission.csv"
        )
    out = ROOT / "submissions" / "baseline_comparison.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()