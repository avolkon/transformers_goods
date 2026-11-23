"""Dictionary baseline: vocabs.json longest match -> baseline_submission.csv."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from product_ner import __version__  # noqa: E402
from product_ner.config import DATA_DIR, RANDOM_STATE  # noqa: E402
from product_ner.data import SLOTS, load_frames  # noqa: E402
from product_ner.gazetteer import MATCHER_RULE, Gazetteer, slots_to_json  # noqa: E402
from product_ner.submission import sha256_file, validate_submission, write_submission  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

EXPERIMENT = "baseline"


def nonempty_rates(frame: pd.DataFrame) -> dict[str, float]:
    rates = {}
    for slot in SLOTS:
        parsed = frame[slot].map(json.loads)
        rates[slot] = round(float(parsed.map(bool).mean()), 4)
    return rates


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Vocab gazetteer baseline")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "submissions" / EXPERIMENT)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="drop one-size ranges, brand letters, centimeters, size-grid сетка, бархатистый",
    )
    parser.add_argument(
        "--canonical",
        action="store_true",
        help="v3: strict plus color aliases, skin/eco leather, jeans->denim",
    )
    args = parser.parse_args(argv)
    if args.canonical:
        args.out_dir = ROOT / "submissions" / "gazetteer_v3"
    elif args.strict:
        args.out_dir = ROOT / "submissions" / "gazetteer_v2"

    train, test, _sample, vocabs = load_frames(args.data_dir)
    gazetteer = Gazetteer(vocabs)
    records = [
        {
            "id": row.id,
            **slots_to_json(
                gazetteer.extract(
                    row.title,
                    row.description,
                    strict=args.strict,
                    canonical=args.canonical,
                )
            ),
        }
        for row in test.itertuples(index=False)
    ]
    submission = pd.DataFrame(records, columns=["id", *SLOTS])
    validate_submission(submission, test, vocabs)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.canonical:
        file_name = "submission.csv"
        upload_name = "submission_v3.csv"
    elif args.strict:
        file_name = "submission.csv"
        upload_name = "submission_v2.csv"
    else:
        file_name = "baseline_submission.csv"
        upload_name = "baseline_submission.csv"
    sub_path = args.out_dir / file_name
    sub_hash = write_submission(submission, sub_path)
    upload_path = ROOT / upload_name
    upload_path.write_bytes(sub_path.read_bytes())

    rates = nonempty_rates(submission)
    all_empty = int(
        submission[list(SLOTS)].map(json.loads).map(bool).any(axis=1).eq(False).sum()
    )
    meta = {
        "experiment": "gazetteer_v3" if args.canonical else ("gazetteer_v2" if args.strict else EXPERIMENT),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "package_version": __version__,
        "seed": args.seed,
        "strict": args.strict or args.canonical,
        "canonical": args.canonical,
        "matcher": MATCHER_RULE,
        "n_rows": len(submission),
        "submission_path": str(sub_path.relative_to(ROOT)),
        "submission_sha256": sub_hash,
        "upload_copy": str(upload_path.relative_to(ROOT)),
        "nonempty_rate": rates,
        "rows_all_slots_empty": all_empty,
        "vocab_sizes": {slot: len(values) for slot, values in vocabs.items()},
    }
    (args.out_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("wrote %s sha256=%s", sub_path, sub_hash)
    logger.info("nonempty_rate=%s all_empty=%s", rates, all_empty)
    if sha256_file(upload_path) != sub_hash:
        raise RuntimeError("upload copy hash differs from submission")
    return 0


if __name__ == "__main__":
    sys.exit(main())
