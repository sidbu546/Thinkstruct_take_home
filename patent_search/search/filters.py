"""Filter stage (DESIGN.md §4.5): runs first, on indexes built for it, and returns the set of
allowed patents. CPC prefixes use an index-backed range scan, keywords and title phrases use
the FTS5 inverted index, exact titles use the title_norm B-tree index."""
from __future__ import annotations

from dataclasses import dataclass, field

from ..ingest.normalize import normalize_title
from ..notices import filter_gap_notice
from ..schemas import Filters

_HIGH = "\U0010ffff"


def _quote(term: str) -> str:
    return '"' + term.replace('"', '""') + '"'


def keyword_fts_query(terms: list[str], fld: str, op: str) -> str:
    terms = [t.strip() for t in terms if t and t.strip()]
    if not terms:
        raise ValueError("keyword filter needs at least one term")
    if op == "phrase":
        body = _quote(" ".join(terms))
    else:
        body = f" {'AND' if op == 'all' else 'OR'} ".join(_quote(t) for t in terms)
    cols = {"title": "title", "abstract": "abstract", "both": "{title abstract}"}[fld]
    return f"{cols} : ({body})"


@dataclass
class FilterResult:
    allowed: set[str] | None                 # None = no filter
    descriptions: list[str] = field(default_factory=list)
    notices: list[str] = field(default_factory=list)


def resolve(con, f: Filters) -> FilterResult:
    if f.is_empty():
        return FilterResult(None)
    where, params, desc, notices = [], [], [], []

    if f.cpc_prefix:
        ors = []
        for p in f.cpc_prefix:
            p = p.strip().upper()
            ors.append("(classification >= ? AND classification < ?)")
            params += [p, p + _HIGH]
        where.append("(" + " OR ".join(ors) + ")")
        desc.append("cpc_prefix:" + "|".join(x.upper() for x in f.cpc_prefix))
    if f.cpc_exclude:
        for p in f.cpc_exclude:
            p = p.strip().upper()
            where.append("(classification IS NULL OR NOT (classification >= ? AND classification < ?))")
            params += [p, p + _HIGH]
        desc.append("cpc_exclude:" + "|".join(x.upper() for x in f.cpc_exclude))
    if f.cpc_prefix or f.cpc_exclude:
        n_null = con.execute("SELECT COUNT(*) FROM patents WHERE classification IS NULL").fetchone()[0]
        if n_null:
            notices.append(filter_gap_notice(n_null, "classification"))

    if f.week_from or f.week_to:
        if f.week_from:
            where.append("filing_week >= ?")
            params.append(f.week_from.isoformat())
        if f.week_to:
            where.append("filing_week <= ?")
            params.append(f.week_to.isoformat())
        desc.append(f"filing_week:{f.week_from or '…'}..{f.week_to or '…'}")
        n_null = con.execute("SELECT COUNT(*) FROM patents WHERE filing_week IS NULL").fetchone()[0]
        if n_null:
            notices.append(filter_gap_notice(n_null, "filing date"))

    if f.title:
        if f.title.mode == "exact":
            where.append("title_norm = ?")
            params.append(normalize_title(f.title.value))
            desc.append(f"title = '{f.title.value}'")
        else:
            where.append("rowid IN (SELECT rowid FROM patents_fts WHERE patents_fts MATCH ?)")
            params.append(f"title : {_quote(f.title.value)}")
            desc.append(f"title contains '{f.title.value}'")

    if f.keywords:
        where.append("rowid IN (SELECT rowid FROM patents_fts WHERE patents_fts MATCH ?)")
        params.append(keyword_fts_query(f.keywords.terms, f.keywords.field, f.keywords.op))
        joiner = {"all": " AND ", "any": " OR ", "phrase": " "}[f.keywords.op]
        desc.append(f"{f.keywords.field} contains {joiner.join(repr(t) for t in f.keywords.terms)}")

    sql = "SELECT doc_number FROM patents WHERE " + " AND ".join(where)
    allowed = {r[0] for r in con.execute(sql, params)}
    return FilterResult(allowed, desc, notices)
