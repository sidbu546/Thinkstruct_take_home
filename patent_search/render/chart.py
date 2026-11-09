"""Claim chart exports (Markdown / CSV): query element × reference × evidence × status."""
from __future__ import annotations

import csv
import io

from ..schemas import SearchResponse

_ICON = {"found": "found", "weak": "weak", "not_found": "not found", "undetermined": "undetermined"}


def _rows(resp: SearchResponse):
    """Yield (query_claim, element_idx, element_text, doc, title, status, score, evidence, data_status, notes)."""
    if resp.claim_map:
        for e in resp.claim_map:
            for m in e.matches:
                for c in m.coverage or []:
                    yield (e.claim_number, c.element, c.element_text, m.doc_number, m.title, c.status, c.score,
                           _ev(c), m.data_status, c.reason or "")
    else:
        for r in resp.results:
            for c in r.coverage or []:
                yield ("query", c.element, c.element_text, r.doc_number, r.title, c.status, c.score, _ev(c),
                       r.data_status, c.reason or "")


def _ev(c) -> str:
    if not c.evidence:
        return ""
    e = c.evidence
    where = {"claim": f"claim {e.claim_number}", "element": f"claim {e.claim_number} el. {(e.element_idx or 0) + 1}",
             "description": f"desc para {e.para_span}", "abstract": "abstract"}.get(e.kind, e.kind)
    return f"{where}: {e.text[:160]}"


def to_csv(resp: SearchResponse) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["query_claim", "element", "element_text", "doc_number", "title", "status", "score", "evidence",
                "data_status", "notes"])
    for row in _rows(resp):
        w.writerow([row[0], row[1] + 1, *row[2:]])
    if resp.data_notice:
        w.writerow([])
        w.writerow(["NOTE", resp.data_notice])
    return buf.getvalue()


def to_markdown(resp: SearchResponse) -> str:
    rows = list(_rows(resp))
    out = ["# Claim chart", ""]
    if not rows:
        out.append("_No element coverage in this response (use --mode claim or --mode patent)._")
    docs = list(dict.fromkeys(r[3] for r in rows))
    by_claim: dict = {}
    for r in rows:
        by_claim.setdefault(r[0], {}).setdefault((r[1], r[2]), {})[r[3]] = r
    status_of = {r[3]: r[8] for r in rows}
    footnotes = []
    for qc, elements in by_claim.items():
        cols = [d for d in docs if any(d in v for v in elements.values())]
        out.append(f"## Query {'claim ' + str(qc) if qc != 'query' else 'claim'}")
        out.append("")
        out.append("| Element | " + " | ".join(f"{d}" + (" ¹" if status_of[d] != "complete" else "") for d in cols) + " |")
        out.append("|---|" + "---|" * len(cols))
        for (idx, text), cells in elements.items():
            vals = []
            for d in cols:
                r = cells.get(d)
                vals.append("" if r is None else f"{_ICON[r[5]]}" + (f" ({r[6]:.2f})" if r[6] is not None else ""))
            out.append(f"| {idx + 1}. {text[:80].replace('|', '/')} | " + " | ".join(vals) + " |")
        out.append("")
    if any(s != "complete" for s in status_of.values()):
        footnotes.append("¹ Source data for this patent is incomplete or partial (pending data research). Its column "
                         "is provisional and will be updated once the data is researched; re-examine it then. "
                         "'undetermined' means the element could not be checked because of a data gap; it is not "
                         "a finding of 'not disclosed'.")
    if resp.data_notice:
        footnotes.append(resp.data_notice)
    return "\n".join(out + footnotes) + "\n"
