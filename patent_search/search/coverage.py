"""Element coverage (DESIGN.md §4.4) and match highlighting.

found        best evidence >= coverage_found
weak         between coverage_weak and coverage_found
not_found    below coverage_weak AND the candidate's data is complete for the sections that
             could disclose it
undetermined below coverage_weak AND the candidate has open gaps there. Reported as pending
             data research, never as not_found: a data gap is not evidence of novelty."""
from __future__ import annotations

import re

import numpy as np

from ..ingest.normalize import content_tokens
from ..schemas import CoverageItem, Evidence

GAP_TYPES = {"description_thin", "claim1_preamble_missing", "claim_opening_missing", "parent_claim_unresolved",
             "embed_failed"}
GAP_MISSING_FIELDS = {"abstract", "claims", "detailed_description"}


def gap_issues(issues: list[dict]) -> list[dict]:
    return [i for i in issues if i["status"] == "open" and (
        i["issue_type"] in GAP_TYPES or (i["issue_type"] == "missing_field" and i["location"] in GAP_MISSING_FIELDS))]


def evidence_for(engine, row: int) -> Evidence:
    return Evidence(kind=engine.kind[row], claim_number=engine.claim_number[row],
                    element_idx=engine.element_idx[row], para_span=engine.para_span[row],
                    text=engine.display_text[row])


def element_coverage(engine, doc: str, elements: list[str], elem_vecs: np.ndarray) -> list[CoverageItem]:
    sc = engine.cfg.search
    rows = engine.doc_rows.get(doc)
    gaps = gap_issues(engine.issues.get(doc, []))
    gap_ids = [i["id"] for i in gaps]
    gap_reason = ", ".join(sorted({i["issue_type"] if i["issue_type"] != "missing_field"
                                   else f"missing {i['location']}" for i in gaps}))
    out = []
    if rows is None or len(rows) == 0:
        for j, el in enumerate(elements):
            out.append(CoverageItem(element=j, element_text=el, status="undetermined",
                                    reason="no searchable text for this patent (pending data research)",
                                    issue_ids=gap_ids))
        return out
    sims = engine.vectors[rows] @ np.atleast_2d(elem_vecs).T          # (n_rows, n_elements)
    for j, el in enumerate(elements):
        b = int(np.argmax(sims[:, j]))
        s = float(sims[b, j])
        row = int(rows[b])
        if s >= sc.coverage_found:
            out.append(CoverageItem(element=j, element_text=el, status="found", score=round(s, 4),
                                    evidence=evidence_for(engine, row)))
        elif s >= sc.coverage_weak:
            out.append(CoverageItem(element=j, element_text=el, status="weak", score=round(s, 4),
                                    evidence=evidence_for(engine, row),
                                    reason=(f"provisional: open data gaps ({gap_reason})" if gaps else None),
                                    issue_ids=gap_ids if gaps else []))
        elif gaps:
            out.append(CoverageItem(element=j, element_text=el, status="undetermined", score=round(s, 4),
                                    reason=f"no match in available text; data gaps open ({gap_reason}), "
                                           "pending data research", issue_ids=gap_ids))
        else:
            out.append(CoverageItem(element=j, element_text=el, status="not_found", score=round(s, 4)))
    return out


def highlight_spans(text: str, query: str, limit: int = 60) -> list[list[int]]:
    stems = set()
    for t in content_tokens(query):
        stems.add(t[:max(4, len(t) - 3)] if len(t) > 5 else t)
    if not stems:
        return []
    pat = re.compile(r"\b(?:" + "|".join(re.escape(s) for s in sorted(stems, key=len, reverse=True)) + r")\w*",
                     re.I)
    return [[m.start(), m.end()] for m in pat.finditer(text)][:limit]
