"""Light text cleanup for indexing and display. Only spacing changes; wording is never altered."""
from __future__ import annotations

import re

_WS = re.compile(r"\s+")
_FIG_GLUE = re.compile(r"(?<=[A-Za-z0-9.,;)])(?=FIGS?\.)")   # "inFIG.1", "1B.FIG." -> space before FIG
_FIG_NUM = re.compile(r"(FIGS?\.)(\d)")              # "FIG.1"    -> "FIG. 1"
# reference numerals glued to words by the source extraction: "tire1of" -> "tire 1 of",
# "damper10placed" -> "damper 10 placed", "2shows" -> "2 shows". Lower-case letter before the digits,
# so chemical formulas (SiO2, C2H4) and ordinals (2nd, 3rd) are left alone.
_NUM_IN_WORD = re.compile(r"(?<=[a-z])(\d+[A-Z]?)(?=[a-z]{2,})")
_NUM_AFTER_WORD = re.compile(r"(?<=[a-z]{2})(\d+[A-Z]?)(?=[.,;:)\s]|$)")
_NUM_BEFORE_WORD = re.compile(r"(?<![A-Za-z\d.,\-])(\d+[A-Z]?)(?=(?!st\b|nd\b|rd\b|th\b)[a-z]{3,})")
_SPACE_PUNCT = re.compile(r"\s+([,.;:)])")           # "claim 6 ," -> "claim 6,"
_OPEN_PAREN = re.compile(r"\(\s+")


def normalize_text(text: str | None) -> str:
    if not text:
        return ""
    t = str(text)
    t = _FIG_GLUE.sub(" ", t)
    t = _FIG_NUM.sub(r"\1 \2", t)
    t = _NUM_IN_WORD.sub(r" \1 ", t)
    t = _NUM_AFTER_WORD.sub(r" \1", t)
    t = _NUM_BEFORE_WORD.sub(r"\1 ", t)
    t = _SPACE_PUNCT.sub(r"\1", t)
    t = _OPEN_PAREN.sub("(", t)
    t = _WS.sub(" ", t)
    return t.strip()


def normalize_title(title: str | None) -> str:
    """Key for exact title matching: upper-case, single spaces."""
    return _WS.sub(" ", str(title or "")).strip().upper()


def word_count(text: str) -> int:
    return len(text.split())


_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9\-]+")
STOPWORDS = frozenset("""
a an and are as at be by for from has have in into is it its of on or such that the their
then there these this to was were which with wherein whereby said each one first second
least plurality comprising comprises including includes having claim claims according
""".split())


def content_tokens(text: str) -> list[str]:
    """Lower-cased content words (for TF-IDF, BM25 queries and highlights)."""
    return [w.lower() for w in _TOKEN.findall(text) if w.lower() not in STOPWORDS and len(w) > 2]
