"""Query analysis (DESIGN.md §4.2): score the query like a claim; complex or claim-mode queries
are also split into elements, each searched as its own sub-query."""
from __future__ import annotations

from dataclasses import dataclass

from ..config import ChunkingConfig
from ..ingest.claims import Claim, assign_tier, split_elements, split_preamble, _NUM
from ..ingest.normalize import normalize_text, word_count


@dataclass
class AnalyzedQuery:
    text: str
    tier: str
    preamble: str | None
    elements: list[str]


def analyze(query: str, mode: str, cfg: ChunkingConfig) -> AnalyzedQuery:
    text = normalize_text(query)
    m = _NUM.match(text)
    if m:
        text = text[m.end():].strip()
    preamble, body, _ = split_preamble(text)
    elements = split_elements(body) or [text]
    c = Claim(number=1, number_inferred=False, status="active", text=text, words=word_count(text),
              elements=elements, preamble=preamble)
    tier = assign_tier(c, cfg)
    # element sub-queries: always in claim mode; in text mode only for long multi-part queries
    use_elements = len(elements) > 1 and (mode == "claim" or tier == "complex")
    return AnalyzedQuery(text=text, tier=tier, preamble=preamble, elements=elements if use_elements else [])
