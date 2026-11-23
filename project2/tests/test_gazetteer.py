import json

import pandas as pd
import pytest

from product_ner.gazetteer import Gazetteer, slots_to_json
from product_ner.submission import parse_slot_cell, validate_submission

VOCABS = {
    "colors": ["Белый", "Серый", "Зеленый", "Голубой", "Красный"],
    "materials": ["Шерсть", "Кожа", "Экокожа", "Деним", "Лен"],
    "length": ["Мини", "Миди", "Макси"],
    "size": ["L", "M", "OS", "S", "XL", "42"],
}


@pytest.fixture
def gazetteer() -> Gazetteer:
    return Gazetteer(VOCABS)


def test_color_and_material_inflection(gazetteer: Gazetteer):
    slots = gazetteer.extract("Свитер - Серый", "100% шерсти мериноса")
    assert slots["colors"] == ["Серый"]
    assert slots["materials"] == ["Шерсть"]
    assert slots["length"] == []
    assert slots["size"] == []


def test_compound_color_and_eco_leather(gazetteer: Gazetteer):
    assert gazetteer.extract("Зелено-голубой", "")["colors"] == ["Зеленый", "Голубой"]
    assert gazetteer.extract("куртка из экокожи", "")["materials"] == ["Экокожа"]


def test_length_does_not_match_min_max_words(gazetteer: Gazetteer):
    slots = gazetteer.extract("минимальный максимум", "юбка миди")
    assert slots["length"] == ["Миди"]


def test_letter_size_and_one_size(gazetteer: Gazetteer):
    assert gazetteer.extract("Топ мини - M", "")["size"] == ["M"]
    assert gazetteer.extract("Топ мини - M", "")["length"] == ["Мини"]
    assert gazetteer.extract("размер one size", "")["size"] == ["OS"]


def test_number_size_needs_a_cue(gazetteer: Gazetteer):
    assert gazetteer.extract("цена 21990", "")["size"] == []
    assert gazetteer.extract("Платье", "размер 42")["size"] == ["42"]
    assert gazetteer.extract("Платье (42)", "")["size"] == ["42"]


def test_strict_drops_false_sizes_and_materials():
    gazetteer = Gazetteer(
        {
            **VOCABS,
            "materials": [*VOCABS["materials"], "Бархат", "Сетка"],
            "size": [*VOCABS["size"], "38", "46", "50"],
        }
    )
    loose = gazetteer.extract("Юбка", "onesize (38-46), бархатистая кожа")
    strict = gazetteer.extract("Юбка", "onesize (38-46), бархатистая кожа", strict=True)
    assert "38" in loose["size"]
    assert "Бархат" in loose["materials"]
    assert strict["size"] == ["OS"]
    assert strict["materials"] == ["Кожа"]

    grid = gazetteer.extract("Сапоги размер S", "Размерная сетка в галерее", strict=True)
    assert grid["size"] == ["S"]
    assert grid["materials"] == []
    assert gazetteer.extract("Платье из сетки", "", strict=True)["materials"] == ["Сетка"]

    assert gazetteer.extract("Дорожка", "размер 50 см", strict=True)["size"] == []
    assert gazetteer.extract("Платье шелковое L Etoile", "", strict=True)["size"] == []
    assert gazetteer.extract("Топ мини - M", "", strict=True)["size"] == ["M"]


def test_canonical_aliases_and_false_materials():
    gazetteer = Gazetteer(
        {
            "colors": ["Бежевый", "Голубой", "Зеленый", "Серый", "Коричневый", "Молочный"],
            "materials": ["Кожа", "Экокожа", "Деним", "Лен", "Хлопок"],
            "length": ["Мини"],
            "size": ["M", "OS"],
        }
    )
    shore = gazetteer.extract("Костюм Изумрудный берег", "", canonical=True)
    assert "Зеленый" not in shore["colors"]
    assert gazetteer.extract("Платье силуэт песочные часы", "", canonical=True)["colors"] == []
    boots = gazetteer.extract(
        "казаки",
        "велюр песочного цвета. носите с голубыми джинсами",
        canonical=True,
    )
    assert boots["colors"] == ["Бежевый"]
    assert gazetteer.extract("Джинсы графитовые", "", canonical=True)["colors"] == ["Серый"]
    assert gazetteer.extract("Джинсы графитовые", "", canonical=True)["materials"] == ["Деним"]
    skin = gazetteer.extract("Майка из хлопка", "устраняет на коже стафилококк", canonical=True)
    assert skin["materials"] == ["Хлопок"]
    eco = gazetteer.extract("Бомбер", "из матовой эко-кожи", canonical=True)
    assert eco["materials"] == ["Экокожа"]
    assert gazetteer.extract("Ремень", "льняной мешок для хранения", canonical=True)["materials"] == []


def test_red_does_not_match_inside_another_word(gazetteer: Gazetteer):
    assert gazetteer.extract("прекрасный день", "")["colors"] == []


def test_submission_roundtrip_and_rejections(gazetteer: Gazetteer):
    test = pd.DataFrame(
        [{"id": "a", "title": "Топ мини - M", "description": "шерсть"}]
    )
    slots = slots_to_json(gazetteer.extract("Топ мини - M", "шерсть"))
    good = pd.DataFrame([{"id": "a", **slots}])
    validate_submission(good, test, VOCABS)
    assert parse_slot_cell(good.loc[0, "materials"]) == ["Шерсть"]

    bad = good.copy()
    bad["colors"] = json.dumps(["Несуществующий"], ensure_ascii=False)
    with pytest.raises(ValueError, match="not in vocab"):
        validate_submission(bad, test, VOCABS)

    broken = good.copy()
    broken["size"] = "[M"
    with pytest.raises(ValueError, match="not JSON"):
        validate_submission(broken, test, VOCABS)

    wrong_id = good.copy()
    wrong_id["id"] = "b"
    with pytest.raises(ValueError, match="id order"):
        validate_submission(wrong_id, test, VOCABS)
