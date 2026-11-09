"""Adaptive chunking (DESIGN.md §2.3).

Each claim / passage gets as much context as it needs to be understood on its own:
  simple claims          -> one small chunk (title + claim)
  dependent claims       -> claim + its ancestor chain (budgeted), so "wherein X is 75 mol %"
                            carries the claims it refines
  moderate/complex indep -> whole claim + one chunk per element (for element-level matching)
  description            -> paragraphs merged while on-topic, long ones split at sentence
                            boundaries where the topic shifts; target size depends on density
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from ..config import ChunkingConfig
from .claims import Claim, ancestor_chain, assign_tier
from .normalize import content_tokens, normalize_text, word_count

_NUMERAL = re.compile(r"\b\d{1,4}[a-z]?\b")
_SENTENCE = re.compile(r"(?<=[.;])\s+(?=[A-Z(])")
_WHEREIN = re.compile(r"\bwherein\b", re.I)
_BOILERPLATE = [re.compile(p, re.I) for p in (
    r"not intended to be (?:exhaustive|limit)", r"for (?:the )?purposes? of illustration",
    r"without departing from the (?:spirit|scope)", r"it (?:is|should) (?:to )?be (?:noted|understood) that",
    r"various (?:changes|modifications)", r"incorporated (?:herein )?by reference",
    r"the terminology used herein", r"those skilled in the art", r"are presented herein",
)]


@dataclass
class Chunk:
    key: str
    doc_number: str
    kind: str                       # abstract | claim | element | description
    embed_text: str
    display_text: str
    claim_number: int | None = None
    element_idx: int | None = None
    section: str | None = None
    para_span: str | None = None
    parent_key: str | None = None
    tier: str | None = None
    weight: float = 1.0
    n_tokens_est: int = 0
    context_claims: str | None = None   # e.g. "1,6" = ancestors included in embed_text


def _est_tokens(text: str, cfg: ChunkingConfig) -> int:
    return int(math.ceil(word_count(text) * cfg.words_to_tokens))


def _delta(c: Claim) -> str:
    """The limitation a dependent claim adds: text from its first 'wherein', else all."""
    m = _WHEREIN.search(c.text)
    return c.text[m.start():] if m else c.text


def _chain_text(title: str, c: Claim, chain: list[Claim], cfg: ChunkingConfig) -> tuple[str, list[int]]:
    parts = [(a, f"Claim {a.number}: {a.text}") for a in chain]
    tail = f"Claim {c.number}: {c.text}"

    def assemble(ps):
        return " ".join([f"{title}."] + [t for _, t in ps] + [tail])

    text = assemble(parts)
    if _est_tokens(text, cfg) <= cfg.chain_token_budget:
        return text, [a.number for a, _ in parts]
    # compress intermediates (all but the root) to their added limitation
    parts = [parts[0]] + [(a, f"Claim {a.number}: {_delta(a)}") for a, _ in parts[1:]] if parts else []
    while len(parts) > 1 and _est_tokens(assemble(parts), cfg) > cfg.chain_token_budget:
        parts.pop(1)                       # drop the oldest intermediate first
    text = assemble(parts)
    if _est_tokens(text, cfg) > cfg.chain_token_budget:     # root itself too long: truncate it
        budget_words = int(cfg.chain_token_budget / cfg.words_to_tokens) - word_count(tail) - 10
        root_words = " ".join(parts[0][1].split()[:max(budget_words, 50)]) if parts else ""
        text = " ".join([f"{title}.", root_words, tail])
    return text, [a.number for a, _ in parts]


def claim_chunks(doc: str, title: str, claims: list[Claim], cfg: ChunkingConfig) -> list[Chunk]:
    active = {c.number: c for c in claims if c.status == "active"}
    out: list[Chunk] = []
    for c in sorted(active.values(), key=lambda x: x.number):
        c.tier = assign_tier(c, cfg)
        key = f"{doc}|c{c.number}"
        if c.is_dependent:
            chain = ancestor_chain(c, active)
            embed, ctx = _chain_text(title, c, chain, cfg)
        else:
            embed, ctx = f"{title}. Claim {c.number}: {c.text}", []
        out.append(Chunk(key=key, doc_number=doc, kind="claim", embed_text=embed, display_text=c.text,
                         claim_number=c.number, tier=c.tier, n_tokens_est=_est_tokens(embed, cfg),
                         context_claims=",".join(map(str, ctx)) or None))
        if not c.is_dependent and c.tier != "simple" and len(c.elements) > 1:
            lead = c.preamble or title
            for i, el in enumerate(c.elements):
                et = f"{title}. {lead}: {el}" if lead != title else f"{title}: {el}"
                out.append(Chunk(key=f"{key}|e{i}", doc_number=doc, kind="element", embed_text=et,
                                 display_text=el, claim_number=c.number, element_idx=i,
                                 parent_key=key, tier=c.tier, n_tokens_est=_est_tokens(et, cfg)))
    return out


# ---------------------------------------------------------------- description passages
def _is_heading(text: str, cfg: ChunkingConfig) -> bool:
    n = word_count(text)
    if n == 0 or n > cfg.heading_max_words:
        return False
    if text.isupper():
        return True
    letters = sum(ch.isalpha() for ch in text)
    if "=" in text or letters < 0.7 * len(text.replace(" ", "")):     # formulas, symbols, tables
        return False
    return not text.rstrip().endswith((".", ":", ";", "?", "!", ")")) and not _NUMERAL.search(text)


class _Tfidf:
    """Per-patent TF-IDF over paragraphs; cosine between adjacent units decides merges."""

    def __init__(self, texts: list[str]):
        docs = [Counter(content_tokens(t)) for t in texts]
        df = Counter(w for d in docs for w in d)
        n = len(docs)
        self.idf = {w: math.log((n + 1) / (c + 1)) + 1 for w, c in df.items()}

    def vec(self, text: str) -> dict[str, float]:
        tf = Counter(content_tokens(text))
        v = {w: c * self.idf.get(w, 1.0) for w, c in tf.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {w: x / norm for w, x in v.items()}

    @staticmethod
    def cos(a: dict, b: dict) -> float:
        if len(a) > len(b):
            a, b = b, a
        return sum(x * b.get(w, 0.0) for w, x in a.items())


def _target(text: str, cfg: ChunkingConfig) -> int:
    n = max(word_count(text), 1)
    dense = len(_NUMERAL.findall(text)) / n >= cfg.dense_numeral_ratio
    return cfg.passage_target_dense if dense else cfg.passage_target_narrative


def _split_long(text: str, tf: _Tfidf, target: int) -> list[str]:
    sents = [s for s in _SENTENCE.split(text) if s.strip()]
    if len(sents) <= 1:                                  # no sentence boundary: hard window
        words = text.split()
        return [" ".join(words[i:i + target]) for i in range(0, len(words), target)]
    pieces, cur, cur_w = [], [], 0
    for i, s in enumerate(sents):
        w = word_count(s)
        topic_shift = cur and cur_w >= 0.6 * target and tf.cos(tf.vec(cur[-1]), tf.vec(s)) < 0.05
        if cur and (cur_w + w > target or topic_shift):
            pieces.append(" ".join(cur))
            cur, cur_w = [], 0
        cur.append(s)
        cur_w += w
    if cur:
        pieces.append(" ".join(cur))
    return pieces


def description_chunks(doc: str, title: str, paragraphs: list, cfg: ChunkingConfig) -> list[Chunk]:
    units: list[tuple[int, str, str | None]] = []     # (para_idx, text, section)
    section = None
    cleaned = [(i, normalize_text(p)) for i, p in enumerate(paragraphs)]
    cleaned = [(i, t) for i, t in cleaned if t]       # empty paragraphs are structural blanks
    tf = _Tfidf([t for _, t in cleaned] or [""])
    for i, t in cleaned:
        if _is_heading(t, cfg):
            section = t
            continue
        tgt = _target(t, cfg)
        if word_count(t) > int(tgt * 1.5):
            units.extend((i, piece, section) for piece in _split_long(t, tf, tgt))
        else:
            units.append((i, t, section))

    out: list[Chunk] = []
    group: list[tuple[int, str, str | None]] = []

    def flush():
        if not group:
            return
        text = " ".join(t for _, t, _ in group)
        sec = group[0][2]
        first, last = group[0][0], group[-1][0]
        span = str(first) if first == last else f"{first}-{last}"
        weight = cfg.boilerplate_weight if sum(bool(p.search(text)) for p in _BOILERPLATE) >= (
            1 if word_count(text) < 120 else 2) else 1.0
        embed = f"{title}. {sec}: {text}" if sec else f"{title}. {text}"
        out.append(Chunk(key=f"{doc}|d{len(out)}", doc_number=doc, kind="description", embed_text=embed,
                         display_text=text, section=sec, para_span=span, weight=weight,
                         n_tokens_est=_est_tokens(embed, cfg)))
        group.clear()

    for u in units:
        if group:
            cur_text = " ".join(t for _, t, _ in group)
            cur_w = word_count(cur_text)
            same_section = u[2] == group[-1][2]
            fits = cur_w + word_count(u[1]) <= _target(cur_text, cfg)
            on_topic = tf.cos(tf.vec(group[-1][1]), tf.vec(u[1])) >= cfg.merge_similarity
            if not (same_section and fits and (on_topic or cur_w < cfg.passage_min_words)):
                flush()
        group.append(u)
    flush()
    return out


def fixed_chunks(doc: str, title: str, claims: list[Claim], paragraphs: list, cfg: ChunkingConfig) -> list[Chunk]:
    """Baseline for the Part 3 evaluation: every claim alone (no ancestor context, no element chunks)
    and the description cut into fixed word windows, ignoring structure and topic."""
    out = [Chunk(key=f"{doc}|c{c.number}", doc_number=doc, kind="claim", embed_text=f"{title}. Claim {c.number}: {c.text}",
                 display_text=c.text, claim_number=c.number, tier=None,
                 n_tokens_est=_est_tokens(c.text, cfg) + 5)
           for c in sorted((c for c in claims if c.status == "active"), key=lambda x: x.number)]
    words, owner = [], []
    for i, p in enumerate(paragraphs):
        t = normalize_text(p)
        if t:
            ws = t.split()
            words += ws
            owner += [i] * len(ws)
    step = cfg.fixed_window_words - cfg.fixed_overlap_words
    for k, start in enumerate(range(0, max(len(words) - cfg.fixed_overlap_words, 1), step)):
        piece = words[start:start + cfg.fixed_window_words]
        if not piece:
            break
        a, b = owner[start], owner[start + len(piece) - 1]
        text = " ".join(piece)
        out.append(Chunk(key=f"{doc}|d{k}", doc_number=doc, kind="description", embed_text=f"{title}. {text}",
                         display_text=text, para_span=str(a) if a == b else f"{a}-{b}",
                         n_tokens_est=_est_tokens(text, cfg)))
    return out


def abstract_chunk(doc: str, title: str, abstract: str, cfg: ChunkingConfig) -> list[Chunk]:
    text = normalize_text(abstract)
    if not text:
        return []
    embed = f"{title}. {text}"
    return [Chunk(key=f"{doc}|abs", doc_number=doc, kind="abstract", embed_text=embed,
                  display_text=text, n_tokens_est=_est_tokens(embed, cfg))]
