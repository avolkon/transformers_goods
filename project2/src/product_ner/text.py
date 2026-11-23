from __future__ import annotations

import re

_WS = re.compile(r"\s+")


def normalize_text(text: object) -> str:
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    text = text.strip().lower().replace("ё", "е")
    return _WS.sub(" ", text)
