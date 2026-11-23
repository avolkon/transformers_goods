from product_ner.config import DATA_DIR
from product_ner.data import load_frames
from product_ner.submission import validate_submission


def test_frames_match_spec():
    train, test, sample, vocabs = load_frames(DATA_DIR)
    assert len(train) == 22654
    assert len(test) == 400
    assert len(sample) == 400
    assert list(train.columns)[:3] == ["id", "title", "description"]
    assert list(test["id"]) == list(sample["id"])
    assert set(train["id"]).isdisjoint(set(test["id"]))
    assert set(vocabs) == {"colors", "materials", "length", "size"}
    assert vocabs["length"] == ["Мини", "Миди", "Макси"]


def test_sample_submission_is_valid_empty_baseline():
    _train, test, sample, vocabs = load_frames(DATA_DIR)
    validate_submission(sample, test, vocabs)
