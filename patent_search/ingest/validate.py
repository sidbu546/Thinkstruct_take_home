"""Data-issue detection (DESIGN.md §2.2). Every defect becomes an issue row; records are
kept and indexed with whatever they have. Outputs later turn these into notices."""
from __future__ import annotations

import json
import re

from .claims import Claim
from .load import RawPatent
from .normalize import word_count

SEVERITY_ORDER = {"blocking": 4, "high": 3, "medium": 2, "low": 1, "info": 0}
MISSING_FIELD_SEVERITY = {"abstract": "high", "claims": "high", "detailed_description": "high",
                          "title": "medium", "classification": "medium"}
THIN_DESCRIPTION_WORDS = 50
_AMBIGUOUS_CPC = re.compile(r"^[A-H]\d{2}[A-Z]\d{3,}")


def _issue(doc, issue_type, severity, location, detail) -> dict:
    return {"doc_number": doc, "issue_type": issue_type, "severity": severity,
            "location": location, "status": "open",
            "detail": detail if isinstance(detail, str) else json.dumps(detail)}


def detect_issues(p: RawPatent, claims: list[Claim], description_words: int) -> list[dict]:
    doc = p.doc_number
    out: list[dict] = []
    values = {"title": p.title, "abstract": p.abstract, "classification": p.classification,
              "claims": [c for c in claims if c.status == "active"], "detailed_description": p.paragraphs}
    for fld, sev in MISSING_FIELD_SEVERITY.items():
        if not values[fld]:
            why = "empty" if fld in p.present_keys else "absent from the record"
            if fld == "classification" and fld in p.present_keys:
                why = "empty or 'N/A'"
            out.append(_issue(doc, "missing_field", sev, fld, f"{fld} is {why}"))

    by_num = {c.number: c for c in claims}
    c1 = by_num.get(1)
    if c1 is not None and c1.status == "active" and c1.preamble_missing:
        out.append(_issue(doc, "claim1_preamble_missing", "high", "claims[1]",
                          "claim 1 starts mid-sentence; its opening words (preamble) are missing"))
    lost_opening = [c.number for c in claims if c.status == "active" and c.number != 1
                    and c.text[:1].islower() and not c.is_dependent]
    if lost_opening:
        out.append(_issue(doc, "claim_opening_missing", "high", "claims",
                          {"claims": lost_opening, "note": "claim starts mid-sentence; its opening words, "
                           "possibly including 'according to claim N', are missing, so it may be a "
                           "dependent claim treated as independent"}))
    inferred = [c.number for c in claims if c.number_inferred]
    if inferred:
        out.append(_issue(doc, "claim_number_inferred", "medium", "claims",
                          {"claims": inferred, "note": "number not in source; inferred from position"}))
    unresolved = {c.number: c.unresolved_refs for c in claims if c.unresolved_refs}
    if unresolved:
        out.append(_issue(doc, "parent_claim_unresolved", "high", "claims",
                          {"claim_to_missing_parents": unresolved}))
    if p.paragraphs and description_words < THIN_DESCRIPTION_WORDS:
        out.append(_issue(doc, "description_thin", "high", "detailed_description",
                          f"only {description_words} words of description"))
    canceled = [c.number for c in claims if c.status == "canceled"]
    if canceled:
        out.append(_issue(doc, "claim_canceled", "info", "claims", {"claims": canceled}))
    if p.classification and _AMBIGUOUS_CPC.match(p.classification):
        out.append(_issue(doc, "classification_ambiguous", "info", "classification",
                          "group/subgroup split is ambiguous without '/'; drill-down below "
                          "subclass uses raw prefixes"))
    if p.filing_week is None:
        out.append(_issue(doc, "date_missing", "medium", "filename",
                          "no filing week in the source file name or XML name"))
    for li in p.load_issues:
        out.append(_issue(doc, li["issue_type"], li["severity"], li["location"], li["detail"]))
    return out


def data_status(issues: list[dict]) -> str:
    open_ = [i for i in issues if i["status"] == "open"]
    if any(i["severity"] == "blocking" for i in open_) or any(
            i["issue_type"] == "missing_field" and i["location"] in ("abstract", "claims", "detailed_description")
            for i in open_):
        return "incomplete"
    if any(SEVERITY_ORDER[i["severity"]] >= SEVERITY_ORDER["medium"] for i in open_):
        return "partial"
    return "complete"


def description_word_count(paragraphs: list) -> int:
    return sum(word_count(str(x)) for x in paragraphs if x)
