"""Fixed wording for data-gap notices (DESIGN.md §5.3).

Rules: say what is missing, that the assessment is provisional, that it will be updated once
data research is complete, and to re-examine after the update. Never say a record or field is
ignored or excluded."""
from __future__ import annotations

import json

_TAIL = ("The overlap/novelty assessment for this patent is provisional and will be updated once "
         "data research on this record is complete. Re-examine this result after the update.")

_TEMPLATES = {
    "missing_field": "The {field} is missing from the source data for this patent (pending data research).",
    "claim1_preamble_missing": ("Claim 1 of this patent is missing its opening words (preamble) in the source "
                                "data (pending data research). Matches involving claim 1 are provisional."),
    "claim_opening_missing": ("Claims {claims} of this patent start mid-sentence in the source data: their opening "
                              "words, possibly including which claim they depend on, are missing (pending data "
                              "research). They are treated as independent claims until the data is researched."),
    "claim_number_inferred": ("Claim numbers {claims} of this patent are not in the source data and were "
                              "inferred from their position (pending data research)."),
    "parent_claim_unresolved": ("Some dependent claims of this patent refer to a claim that is not in the source "
                                "data ({detail}); they were indexed without that parent's text (pending data research)."),
    "description_thin": ("The detailed description of this patent is thin or not yet available ({detail}; "
                         "pending data research). Claim elements that could only be disclosed there are "
                         "marked undetermined, not 'not found'."),
    "date_missing": "The filing date of this patent is not in the source data (pending data research).",
    "duplicate_conflict": ("Another record with this document number has different content ({detail}); "
                           "the more complete record is shown (pending data research)."),
    "embed_failed": "Part of this patent could not be embedded and is searchable by keywords only (pending repair).",
}
_FIELD_NAMES = {"abstract": "abstract", "claims": "claims", "detailed_description": "detailed description",
                "title": "title", "classification": "classification code"}


def issue_notice(issue_type: str, location: str | None, detail: str | None) -> str:
    tpl = _TEMPLATES.get(issue_type, "This patent has an open data issue of type '{type}' (pending data research).")
    d = detail or ""
    claims = ""
    try:
        parsed = json.loads(d)
        if isinstance(parsed, dict) and "claims" in parsed:
            nums = parsed["claims"]
            claims = ", ".join(map(str, nums[:8])) + (" …" if len(nums) > 8 else "")
        if isinstance(parsed, dict) and "claim_to_missing_parents" in parsed:
            pairs = parsed["claim_to_missing_parents"]
            d = "; ".join(f"claim {k} cites claim {', '.join(map(str, v))}" for k, v in list(pairs.items())[:5])
    except (json.JSONDecodeError, TypeError):
        pass
    text = tpl.format(field=_FIELD_NAMES.get(location or "", location or "field"), claims=claims or "some",
                      detail=d, type=issue_type)
    return f"{text} {_TAIL}"


def response_notice(n_affected: int, n_results: int) -> str | None:
    if not n_affected:
        return None
    return (f"{n_affected} of {n_results} results have incomplete or partial source data. Their "
            "overlap/novelty assessment is provisional and will be updated once data research on those "
            "records is complete. Re-examine them after the update.")


def filter_gap_notice(n_missing: int, what: str) -> str:
    return (f"{n_missing} patents have no {what} in the source data yet (pending data research) and cannot "
            f"match this {what} filter. Results may be incomplete; re-run the search after the update.")
