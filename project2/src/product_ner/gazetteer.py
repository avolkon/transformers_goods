from __future__ import annotations

import json
import re

from product_ner.data import SLOTS
from product_ner.text import normalize_text

# Closed inflection for short lemmas. Longer stems use a bounded Cyrillic tail.
_HAND_MATERIAL = {
    "лен": r"(?<!\w)(?:льнян[а-я]{0,3}|льна|льну|льном|льне|лен)(?!\w)",
    "мех": r"(?<!\w)мех(?:овой|овая|овое|овые|ового|овой|ами|ах|ом|а|у|е)?(?!\w)",
    "шелк": r"(?<!\w)шелк(?:овый|овая|овое|овые|ового|овой|овым|ами|ах|ом|а|у|е|и)?(?!\w)",
    "кожа": r"(?<!\w)кож(?:аный|аная|аное|аные|аного|аной|аным|ами|ах|ей|ею|ой|а|и|у|е)?(?!\w)",
    "замша": r"(?<!\w)замш(?:евый|евая|евое|евые|евого|а|и|у|ей|е)?(?!\w)",
    "сетка": r"(?<!\w)сетк(?:ами|ах|ой|а|и|у|е)?(?!\w)",
    "твид": r"(?<!\w)твид(?:овый|овая|овое|овые|а|у|е|ом)?(?!\w)",
}
_COLOR_SPECIAL = {
    "металлик": r"(?<!\w)металлик(?:а|у|ом|е)?(?!\w)",
    "мультиколор": r"(?<!\w)мультиколор(?:а|у|ом|е)?(?!\w)",
}
_ADJECTIVE_ENDINGS = {
    "ый": "ый|ая|ое|ые|ого|ой|ым|ую|ом|ых|ыми|о",
    "ий": "ий|яя|ее|ие|его|ей|им|юю|ем|их|ими|е",
    "ой": "ой|ая|ое|ые|ого|ой|ым|ую|ом|ых|ыми|о",
}
_LENGTH = {"мини": "Мини", "миди": "Миди", "макси": "Макси"}
_ONE_SIZE = re.compile(r"(?<!\w)one[\s\-]*size(?!\w)")
_SIZE_CUE = re.compile(r"(?<!\w)(?:размеры|размер|sizes|size|р\s*-\s*р|р\s*р)(?!\w)")
_LETTER_CONTEXT = re.compile(r"(?:размер\w*|sizes|size|[-–—/(,:])\s*$", re.IGNORECASE)
_GRID_SETKA = re.compile(r"(?:размерн\w*|широк\w*)\s+$", re.IGNORECASE)

MATCHER_RULE = {
    "fields": ["title", "description"],
    "casefold": True,
    "yo_to_e": True,
    "longest_match": True,
    "unit_converter": False,
    "morphology": "vocab stem plus a closed set of Russian endings",
    "length": "exact token мини/миди/макси",
    "size_letters": "standalone token, longest first",
    "size_numbers": "only after a size cue or in parentheses",
    "one_size_alias": "one size -> OS",
    "strict": {
        "os_drops_other_sizes": True,
        "single_letter_needs_size_context": True,
        "number_before_cm_is_not_size": True,
        "size_grid_is_not_setka": True,
        "barhatistyj_is_not_velvet": True,
    },
}


def _color_pattern(canonical: str) -> str:
    word = normalize_text(canonical)
    if word in _COLOR_SPECIAL:
        return _COLOR_SPECIAL[word]
    for suffix, endings in _ADJECTIVE_ENDINGS.items():
        if word.endswith(suffix):
            stem = word[: -len(suffix)]
            return rf"(?<!\w){re.escape(stem)}(?:{endings})(?!\w)"
    return rf"(?<!\w){re.escape(word)}(?!\w)"


def _material_patterns(canonical: str) -> list[str]:
    word = normalize_text(canonical)
    if word in _HAND_MATERIAL:
        return [_HAND_MATERIAL[word]]
    stems = [word]
    if word.endswith("ок") and len(word) > 4:
        stems.append(word[:-2] + "к")
    elif word.endswith("ь"):
        stems.append(word[:-1])
    elif word[-1] in "аяоеиы":
        stems.append(word[:-1])
    patterns = []
    for stem in dict.fromkeys(stems):
        if len(stem) >= 5:
            patterns.append(rf"(?<!\w){re.escape(stem)}[а-я]{{0,6}}(?!\w)")
        else:
            patterns.append(rf"(?<!\w){re.escape(stem)}(?!\w)")
    return patterns


def _keep_spans(matches: list[tuple[int, int, str]]) -> list[tuple[int, int, str]]:
    chosen: list[tuple[int, int, str]] = []
    occupied: list[tuple[int, int]] = []
    for start, end, label in sorted(matches, key=lambda item: (-(item[1] - item[0]), item[0])):
        if any(start < right and end > left for left, right in occupied):
            continue
        occupied.append((start, end))
        chosen.append((start, end, label))
    return chosen


def _resolve(matches: list[tuple[int, int, str]]) -> list[str]:
    return [label for _start, _end, label in _keep_spans(matches)]


class Gazetteer:
    def __init__(self, vocabs: dict[str, list[str]]) -> None:
        self.order = {
            slot: {name: index for index, name in enumerate(vocabs[slot])} for slot in SLOTS
        }
        self._colors = [
            (re.compile(_color_pattern(name)), name) for name in vocabs["colors"]
        ]
        self._materials = [
            (re.compile(pattern), name)
            for name in vocabs["materials"]
            for pattern in _material_patterns(name)
        ]
        self._lengths = [
            (re.compile(rf"(?<!\w){re.escape(surface)}(?!\w)"), canonical)
            for surface, canonical in _LENGTH.items()
            if canonical in self.order["length"]
        ]
        letters = [name for name in vocabs["size"] if not name.isdigit()]
        letters = sorted(letters, key=len, reverse=True)
        self._letters = re.compile(
            r"(?<!\w)(" + "|".join(re.escape(name) for name in letters) + r")(?!\w)",
            re.IGNORECASE,
        )
        self._letter_canonical = {name.upper(): name for name in letters}
        numbers = [name for name in vocabs["size"] if name.isdigit()]
        self._number_canonical = {name: name for name in numbers}
        number_alt = "|".join(re.escape(name) for name in sorted(numbers, key=len, reverse=True))
        self._numbers = re.compile(rf"(?<!\d)({number_alt})(?!\d)")
        self._os = "OS" if "OS" in self.order["size"] else None

    def extract(
        self,
        title: object,
        description: object,
        *,
        strict: bool = False,
        canonical: bool = False,
    ) -> dict[str, list[str]]:
        title_norm = normalize_text(title)
        text = normalize_text(f"{title or ''} {description or ''}")
        found = {
            "colors": self._scan(self._colors, text),
            "materials": self._scan(self._materials, text),
            "length": self._scan(self._lengths, text),
            "size": self._sizes(text),
        }
        if strict or canonical:
            found["size"] = self._strict_sizes(text, found["size"])
            found["materials"] = self._strict_materials(text, found["materials"])
        if canonical:
            found = self._canonical(title_norm, text, found)
        return {slot: self._in_vocab_order(slot, labels) for slot, labels in found.items()}

    def strict_spans(self, title: object, description: object) -> tuple[str, list[tuple[int, int, str, str]]]:
        """Character spans kept by the v2 strict gazetteer: (start, end, slot, label)."""
        text = normalize_text(f"{title or ''} {description or ''}")
        allowed = self.extract(title, description, strict=True)
        spans: list[tuple[int, int, str, str]] = []
        for slot, patterns in (
            ("colors", self._colors),
            ("materials", self._materials),
            ("length", self._lengths),
        ):
            matches = [
                (match.start(), match.end(), label)
                for pattern, label in patterns
                for match in pattern.finditer(text)
                if label in allowed[slot]
            ]
            for start, end, label in _keep_spans(matches):
                spans.append((start, end, slot, label))
        for match in self._letters.finditer(text):
            label = self._letter_canonical[match.group(1).upper()]
            if label in allowed["size"]:
                spans.append((match.start(), match.end(), "size", label))
        if self._os and self._os in allowed["size"]:
            found = _ONE_SIZE.search(text)
            if found:
                spans.append((found.start(), found.end(), "size", self._os))
        for match in self._numbers.finditer(text):
            label = self._number_canonical[match.group(1)]
            window = text[max(0, match.start() - 24) : match.start()]
            parenthetical = match.start() > 0 and text[match.start() - 1] in "(["
            if label in allowed["size"] and (parenthetical or _SIZE_CUE.search(window)):
                spans.append((match.start(), match.end(), "size", label))
        return text, spans

    @staticmethod
    def _scan(patterns: list[tuple[re.Pattern[str], str]], text: str) -> list[str]:
        matches = [
            (match.start(), match.end(), label)
            for pattern, label in patterns
            for match in pattern.finditer(text)
        ]
        return _resolve(matches)

    def _sizes(self, text: str) -> list[str]:
        labels: list[str] = []
        labels.extend(self._letter_canonical[m.group(1).upper()] for m in self._letters.finditer(text))
        if self._os and _ONE_SIZE.search(text):
            labels.append(self._os)
        for match in self._numbers.finditer(text):
            window = text[max(0, match.start() - 24) : match.start()]
            parenthetical = match.start() > 0 and text[match.start() - 1] in "(["
            if parenthetical or _SIZE_CUE.search(window):
                labels.append(self._number_canonical[match.group(1)])
        return labels

    def _strict_sizes(self, text: str, labels: list[str]) -> list[str]:
        if "OS" in labels:
            return ["OS"]
        kept: list[str] = []
        for label in labels:
            if label.isdigit():
                if _number_is_only_centimeters(text, label):
                    continue
            elif len(label) == 1 and not _letter_has_size_context(text, label):
                continue
            kept.append(label)
        return kept

    def _strict_materials(self, text: str, labels: list[str]) -> list[str]:
        kept: list[str] = []
        for label in labels:
            if label == "Бархат" and not re.search(r"(?<!\w)бархат(?!ист)", text):
                continue
            if label == "Сетка" and not _setka_is_fabric(text):
                continue
            kept.append(label)
        return kept

    def _canonical(self, title: str, text: str, found: dict[str, list[str]]) -> dict[str, list[str]]:
        colors = list(found["colors"])
        for pattern, label, blocked_after in _COLOR_ALIASES:
            if label not in self.order["colors"]:
                continue
            for match in pattern.finditer(text):
                after = text[match.end() : match.end() + 12]
                if blocked_after is not None and blocked_after.match(after):
                    continue
                colors.append(label)
        colors = _drop_outfit_colors(title, text, colors, self._colors)
        materials = _fix_leather(text, list(found["materials"]))
        materials = _fix_denim(title, text, materials)
        materials = _drop_linen_bag(text, materials)
        found = dict(found)
        found["colors"] = colors
        found["materials"] = materials
        return found

    def _in_vocab_order(self, slot: str, labels: list[str]) -> list[str]:
        unique = [label for label in dict.fromkeys(labels) if label in self.order[slot]]
        return sorted(unique, key=self.order[slot].__getitem__)


def _letter_has_size_context(text: str, letter: str) -> bool:
    pattern = re.compile(rf"(?<!\w){re.escape(letter)}(?!\w)", re.IGNORECASE)
    for match in pattern.finditer(text):
        before = text[max(0, match.start() - 18) : match.start()]
        after = text[match.end() : match.end() + 1]
        if _LETTER_CONTEXT.search(before) or after in ")-,/":
            return True
    return False


def _number_is_only_centimeters(text: str, number: str) -> bool:
    hits = list(re.finditer(rf"(?<!\d){re.escape(number)}(?!\d)", text))
    if not hits:
        return False
    return all(re.match(r"\s*см(?!\w)", text[hit.end() :]) for hit in hits)


def _setka_is_fabric(text: str) -> bool:
    for match in re.finditer(r"(?<!\w)сетк", text):
        before = text[max(0, match.start() - 24) : match.start()]
        if _GRID_SETKA.search(before):
            continue
        return True
    return False


_COLOR_ALIASES: list[tuple[re.Pattern[str], str, re.Pattern[str] | None]] = [
    (
        re.compile(r"(?<!\w)изумрудн(?:ый|ая|ое|ые|ого|ой|ым|ую)(?!\w)"),
        "Зеленый",
        re.compile(r"\s+берег"),
    ),
    (
        re.compile(r"(?<!\w)шоколад(?:н(?:ый|ая|ое|ые|ого|ой|ым|ую|ых))?(?!\w)"),
        "Коричневый",
        None,
    ),
    (
        re.compile(r"(?<!\w)песочн(?:ый|ая|ое|ые|ого|ой|ым|ую)(?!\w)"),
        "Бежевый",
        re.compile(r"\s+час"),
    ),
    (
        re.compile(r"(?<!\w)сливочн(?:ый|ая|ое|ые|ого|ой|ым|ую)(?!\w)"),
        "Молочный",
        None,
    ),
    (
        re.compile(r"(?<!\w)графит(?:овый|овая|овое|овые|ового|овой|овым)?(?!\w)"),
        "Серый",
        None,
    ),
    (re.compile(r"(?<!\w)хаки(?!\w)"), "Зеленый", None),
]
_OUTFIT_PREP = re.compile(r"(?:^|\s)(?:с|к|или)\s+$")
_OUTFIT_GARMENT = re.compile(r"\s*(?:джинс|брюк|пиджак|пальто|юбк|рубаш|худи)\w*")
_JEANS = re.compile(r"(?<!\w)джинс")
_DENIM_WORD = re.compile(r"(?<!\w)деним\w*")
_DENIM_STYLING = re.compile(r"(?:^|\s)(?:с|к)\s+$")
_LINEN_BAG = re.compile(r"\s+меш")


def _drop_outfit_colors(
    title: str,
    text: str,
    labels: list[str],
    patterns: list[tuple[re.Pattern[str], str]],
) -> list[str]:
    kept: list[str] = []
    for label in dict.fromkeys(labels):
        spans = [match for pattern, name in patterns if name == label for match in pattern.finditer(text)]
        in_title = any(pattern.search(title) for pattern, name in patterns if name == label)
        if spans and not in_title and all(_span_is_outfit(text, match.start(), match.end()) for match in spans):
            continue
        kept.append(label)
    return kept


def _span_is_outfit(text: str, start: int, end: int) -> bool:
    before = text[max(0, start - 16) : start]
    after = text[end : end + 24]
    return bool(_OUTFIT_PREP.search(before) and _OUTFIT_GARMENT.match(after))


def _fix_leather(text: str, labels: list[str]) -> list[str]:
    spans = list(re.finditer(r"(?<!\w)кож\w*", text))
    if not spans:
        return labels
    kinds = [_leather_kind(text, match) for match in spans]
    labels = list(labels)
    if "eco" in kinds and "Экокожа" not in labels:
        labels.append("Экокожа")
    real = [kind for kind in kinds if kind == "real"]
    if "Кожа" in labels and not real:
        labels = [label for label in labels if label != "Кожа"]
    elif "eco" in kinds and "Кожа" in labels and real == []:
        labels = [label for label in labels if label != "Кожа"]
    return labels


def _leather_kind(text: str, match: re.Match[str]) -> str:
    pre = text[max(0, match.start() - 8) : match.start()]
    post = text[match.end() : match.end() + 16]
    if re.search(r"эко[-\s]?$", pre):
        return "eco"
    if re.search(r"имитир", text[max(0, match.start() - 40) : match.start()]):
        return "imit"
    if re.search(r"(?:на|по)\s+$", pre) or "дыша" in post:
        return "skin"
    return "real"


def _fix_denim(title: str, text: str, labels: list[str]) -> list[str]:
    labels = list(labels)
    if _JEANS.search(title) or _DENIM_WORD.search(title):
        labels.append("Деним")
        return labels
    spans = list(_DENIM_WORD.finditer(text))
    if "Деним" in labels and spans and all(_DENIM_STYLING.search(text[max(0, m.start() - 4) : m.start()]) for m in spans):
        labels = [label for label in labels if label != "Деним"]
    return labels


def _drop_linen_bag(text: str, labels: list[str]) -> list[str]:
    spans = list(re.finditer(r"(?<!\w)льнян\w*", text))
    if "Лен" in labels and spans and all(_LINEN_BAG.match(text[m.end() : m.end() + 8]) for m in spans):
        return [label for label in labels if label != "Лен"]
    return labels


def slots_to_json(slots: dict[str, list[str]]) -> dict[str, str]:
    return {slot: json.dumps(slots[slot], ensure_ascii=False) for slot in SLOTS}
