"""Claim parsing: numbers (incl. ranges), canceled claims, dependencies, depth,
preamble, elements and complexity tier (DESIGN.md §2.1, §2.3).

Source quirks handled (all measured on the small dataset):
  * number written inline: "7 . The tire according to claim 6 , wherein ..."
  * canceled ranges:       "1 - 5 . (canceled)"
  * no number at all (3,221 items): number inferred from position -> number_inferred
  * claim 1 preamble stripped (501 of 640 patents): starts lower-case -> preamble_missing
  * dependent claim citing a number that is not present -> unresolved_refs
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..config import ChunkingConfig
from .normalize import normalize_text, word_count

_NUM = re.compile(r"^\s*(\d+)\s*(?:[-–]\s*(\d+))?\s*\.\s*(?!\d)")
_CANCELED = re.compile(r"^\(?\s*(?:canceled|cancelled|deleted|withdrawn)\s*\)?\s*\.?\s*$", re.I)
_REF_RANGE = re.compile(r"\bclaims?\s+(\d+)\s*(?:to|through|-|–)\s*(\d+)", re.I)
_REF_LIST = re.compile(r"\bclaims?\s+(\d+(?:\s*,\s*\d+)*(?:\s*,?\s*(?:or|and)\s+\d+)?)", re.I)
_REF_ANY_PRECEDING = re.compile(
    r"\bany\s+(?:one\s+)?of\s+the\s+(?:preceding|previous|foregoing|above)\s+claims", re.I)
_TRANSITION = re.compile(
    r"^(.{0,300}?\b(?:comprising|comprises|consisting essentially of|consisting of|consists of|"
    r"including|includes|having|characterized by|characterised by|characterized in that)\b)\s*:?\s+",
    re.I)
_WHEREIN_SPLIT = re.compile(r",?\s+(?=wherein\b)", re.I)
_REF_WINDOW = 300


@dataclass
class Claim:
    number: int
    number_inferred: bool
    status: str                       # active | canceled
    text: str                         # display text, number prefix removed, normalized
    depends_on: list[int] = field(default_factory=list)       # resolved parents
    unresolved_refs: list[int] = field(default_factory=list)  # cited but not present / canceled
    depth: int = 0
    preamble: str | None = None
    preamble_missing: bool = False
    elements: list[str] = field(default_factory=list)
    words: int = 0
    tier: str = "simple"

    @property
    def is_dependent(self) -> bool:
        return bool(self.depends_on or self.unresolved_refs)

    @property
    def multi_dep(self) -> bool:
        return len(self.depends_on) + len(self.unresolved_refs) > 1


def _refs(text: str, own_number: int, prior: list[int]) -> list[int]:
    head = text[:_REF_WINDOW]
    refs: set[int] = set()
    if _REF_ANY_PRECEDING.search(head):
        refs.update(prior)
    for a, b in _REF_RANGE.findall(head):
        lo, hi = sorted((int(a), int(b)))
        refs.update(range(lo, hi + 1))
    for group in _REF_LIST.findall(head):
        refs.update(int(n) for n in re.findall(r"\d+", group))
    return sorted(r for r in refs if r < own_number)


def split_elements(body: str) -> list[str]:
    """Split a claim body into limitations on ';' and before 'wherein' clauses."""
    parts: list[str] = []
    for piece in body.split(";"):
        for sub in _WHEREIN_SPLIT.split(piece):
            s = sub.strip().strip(",").strip()
            s = re.sub(r"^and\s+", "", s, flags=re.I).rstrip(".").strip()
            if not s:
                continue
            if parts and word_count(s) < 2:          # stray fragment, not a limitation
                parts[-1] = f"{parts[-1]}; {s}"
            else:
                parts.append(s)
    return parts


def split_preamble(text: str) -> tuple[str | None, str, bool]:
    """Return (preamble, body, preamble_missing)."""
    if not text:
        return None, "", True
    if text[0].islower():                     # opening words lost in the source
        return None, text, True
    m = _TRANSITION.match(text)
    if m:
        return m.group(1).strip(), text[m.end():].strip(), False
    return None, text, False


def parse_claims(raw_claims: list) -> list[Claim]:
    claims: list[Claim] = []
    used: set[int] = set()
    last = 0
    for item in raw_claims or []:
        raw = str(item or "")
        m = _NUM.match(raw)
        if m:
            start = int(m.group(1))
            end = int(m.group(2)) if m.group(2) else start
            body = raw[m.end():]
            numbers, inferred = list(range(start, end + 1)), False
            if any(n in used for n in numbers):          # duplicate numbering -> infer
                numbers, inferred = [max(used | {last}) + 1], True
        else:
            body = raw
            numbers, inferred = [last + 1], True
        text = normalize_text(body)
        canceled = bool(_CANCELED.match(text)) or not text
        for n in numbers:
            used.add(n)
            last = max(last, n)
            claims.append(Claim(number=n, number_inferred=inferred,
                                status="canceled" if canceled else "active",
                                text="" if canceled else text, words=0 if canceled else word_count(text)))

    active = {c.number: c for c in claims if c.status == "active"}
    for c in active.values():
        prior = sorted(n for n in active if n < c.number)
        refs = _refs(c.text, c.number, prior)
        c.depends_on = [r for r in refs if r in active]
        c.unresolved_refs = [r for r in refs if r not in active]
        if c.is_dependent:
            c.elements = [c.text]
        else:
            c.preamble, body, c.preamble_missing = split_preamble(c.text)
            c.elements = split_elements(body) or [c.text]

    memo: dict[int, int] = {}

    def depth(n: int, seen: frozenset = frozenset()) -> int:
        if n in memo:
            return memo[n]
        c = active[n]
        parents = [p for p in c.depends_on if p not in seen]
        d = 0 if not parents else 1 + max(depth(p, seen | {n}) for p in parents)
        memo[n] = d
        return d

    for n in active:
        active[n].depth = depth(n)
    return claims


def assign_tier(c: Claim, cfg: ChunkingConfig) -> str:
    n_el = len(c.elements)
    if (c.depth >= cfg.complex_min_depth or c.multi_dep
            or n_el >= cfg.complex_min_elements or c.words >= cfg.complex_min_words):
        return "complex"
    if not c.is_dependent and c.words <= cfg.simple_max_words and n_el <= cfg.simple_max_elements:
        return "simple"
    return "moderate"


def ancestor_chain(c: Claim, by_number: dict[int, Claim]) -> list[Claim]:
    """Root -> ... -> parent (excluding c), following the first resolved parent,
    plus any extra parents of multi-dependent claims (deduplicated, in number order)."""
    chain: list[Claim] = []
    seen: set[int] = {c.number}
    stack = list(c.depends_on)
    while stack:
        p = stack.pop(0)
        if p in seen or p not in by_number:
            continue
        seen.add(p)
        chain.append(by_number[p])
        stack.extend(by_number[p].depends_on)
    return sorted(chain, key=lambda x: x.number)
