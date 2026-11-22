from __future__ import annotations

import re

import numpy as np

_WS = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        text = "" if text is None or (isinstance(text, float) and np.isnan(text)) else str(text)
    text = text.strip().lower().replace("ё", "е")
    return _WS.sub(" ", text)