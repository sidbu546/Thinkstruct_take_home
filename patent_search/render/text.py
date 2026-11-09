"""Terminal rendering of a SearchResponse (the CLI default; --format json prints the JSON)."""
from __future__ import annotations

import textwrap

from ..schemas import SearchResponse

_MARK = {"found": "found", "weak": "weak", "not_found": "not found", "undetermined": "undetermined"}
_SHORT = {"claim1_preamble_missing": "claim 1 lost its opening words", "claim_number_inferred": "claim numbers inferred",
          "claim_opening_missing": "other claims start mid-sentence",
          "parent_claim_unresolved": "a dependent claim cites a missing claim",
          "description_thin": "description thin / not yet available", "missing_field": "a field is missing",
          "date_missing": "filing date missing", "duplicate_conflict": "conflicting duplicate record"}


def _where(bm) -> str:
    if bm is None:
        return ""
    if bm.kind == "claim":
        inf = " (number inferred)" if bm.claim_number_inferred else ""
        ctx = f", with context of claim(s) {', '.join(map(str, bm.context_claims))}" if bm.context_claims else ""
        return f"claim {bm.claim_number}{inf}{ctx}"
    if bm.kind == "element":
        return f"claim {bm.claim_number}, element {bm.element_idx + 1}"
    if bm.kind == "description":
        span = bm.para_span or ""
        a, _, b = span.partition("-")
        para = f"para {int(a) + 1}" + (f"-{int(b) + 1}" if b else "")
        return f"description {para}" + (f" ({bm.section})" if bm.section else "")
    return "abstract"


def _snippet(text: str, highlights: list, width: int = 220) -> str:
    start = max(0, highlights[0][0] - 60) if highlights else 0
    s = text[start:start + width]
    return ("…" if start else "") + s + ("…" if start + width < len(text) else "")


def _cov_summary(cov) -> str:
    if not cov:
        return ""
    n = len(cov)
    found = sum(c.status == "found" for c in cov)
    und = sum(c.status == "undetermined" for c in cov)
    s = f"{found}/{n} found"
    return s + (f", {und} undetermined" if und else "")


def render(resp: SearchResponse, width: int = 100) -> str:
    out = []
    q = resp.query
    filt = ", ".join(f"{k}={v}" for k, v in resp.filters_applied.items()) or "none"
    out.append(f"Query: {q.mode} mode" + (f" · tier: {q.tier}" if q.tier else "") +
               (f" · {len(q.elements)} elements" if q.elements else "") + f" · filters: {filt}")
    out.append(f"Pipeline: {' → '.join(resp.pipeline)}   ({resp.total_matched} matched · {resp.took_ms} ms)")
    if resp.rerank:
        rk = resp.rerank
        out.append(f"Phase 2: {rk['model']} reranked {rk['scored']} of {rk['depth']} candidates "
                   f"({rk['pairs']} pairs, {resp.stage_ms.get('rerank', 0)} ms); '(was N)' = phase-1 rank")
    for n in resp.pipeline_notices:
        out.append(textwrap.fill("! " + n, width, subsequent_indent="  "))
    for n in resp.filter_notices:
        out.append(textwrap.fill("! " + n, width, subsequent_indent="  "))
    if resp.data_notice:
        out.append(textwrap.fill("! " + resp.data_notice, width, subsequent_indent="  "))
        out.append("  (Full per-patent notices: --format json, or `show <doc_number>`.)")
    out.append("")
    if not resp.results:
        out.append("No results.")
    for r in resp.results:
        cov = _cov_summary(r.coverage)
        moved = (f" (was {r.signals.phase1_rank})" if resp.rerank and r.signals.phase1_rank
                 and r.signals.phase1_rank != r.rank else "")
        out.append(f"{r.rank:>3}.{moved} {r.doc_number}  {r.title[:52]:<52}  {r.classification or '(no CPC)':<11} "
                   f"{r.filing_week or '(no date)'}  score {r.score:.4f}" + (f"  [{cov}]" if cov else ""))
        if r.best_match:
            bm = r.best_match
            out.append(f"     Best match: {_where(bm)}")
            if bm.element_text:
                out.append(textwrap.fill(f"element: \"{bm.element_text[:200]}\"", width,
                                         initial_indent="       ", subsequent_indent="       "))
            out.append(textwrap.fill(f"\"{_snippet(bm.display_text, bm.highlights)}\"", width,
                                     initial_indent="       ", subsequent_indent="       "))
        if r.coverage:
            for c in r.coverage:
                if c.status != "found":
                    line = f"element {c.element + 1} \"{c.element_text[:70]}\": {_MARK[c.status]}"
                    if c.reason:
                        line += f" — {c.reason}"
                    out.append(textwrap.fill(line, width, initial_indent="       ", subsequent_indent="         "))
        if r.data_issues:
            gaps = "; ".join(_SHORT.get(i.type, i.type) for i in r.data_issues)
            out.append(textwrap.fill(f"! Data {r.data_status}: {gaps}. Provisional, pending data research; "
                                     f"re-examine after the update.", width,
                                     initial_indent="     ", subsequent_indent="       "))
    if resp.claim_map:
        out.append("")
        out.append("Claim map")
        for e in resp.claim_map:
            out.append(f"  Claim {e.claim_number} ({e.tier or '-'}): {e.claim_text[:90]}…")
            for m in e.matches:
                cov = _cov_summary(m.coverage)
                out.append(f"     → {m.doc_number} {m.title[:40]:<40} {_where(m.match):<28} "
                           f"score {m.score:.4f}" + (f" [{cov}]" if cov else "") +
                           ("" if m.data_status == "complete" else f" (data {m.data_status})"))
    if resp.facets:
        f = resp.facets
        out.append("")
        out.append("Facets: CPC " + ", ".join(f"{k} {v}" for k, v in list(f.cpc_subclass.items())[:6]) +
                   " | data status " + ", ".join(f"{k} {v}" for k, v in f.data_status.items()))
    return "\n".join(out)


def render_patent(p: dict, width: int = 100) -> str:
    out = [f"{p['doc_number']}  {p['title']}", f"CPC {p['classification'] or '(none)'} · filing week "
           f"{p['filing_week'] or '(none)'} · data status {p['data_status']} · record v{p['record_version']}", ""]
    out.append(textwrap.fill("Abstract: " + (p["abstract"] or "(missing, pending data research)"), width))
    out.append("")
    for c in p["claims"]:
        if c["status"] == "canceled":
            out.append(f"  Claim {c['claim_number']}: (canceled)")
            continue
        dep = f" [depends on {', '.join(map(str, c['depends_on']))}]" if c["depends_on"] else ""
        unres = f" [cites missing claim {', '.join(map(str, c['unresolved_refs']))}]" if c["unresolved_refs"] else ""
        inf = " (number inferred)" if c["number_inferred"] else ""
        out.append(textwrap.fill(f"Claim {c['claim_number']}{inf} · {c['tier']}{dep}{unres}: {c['text']}", width,
                                 initial_indent="  ", subsequent_indent="    "))
    out.append("")
    out.append(f"{len(p['passages'])} description passages.")
    open_issues = [i for i in p["data_issues"] if i["status"] == "open"]
    if open_issues:
        out.append("")
        out.append("Open data issues:")
        for i in open_issues:
            out.append(textwrap.fill(f"[{i['severity']}] {i['issue_type']}: {i['notice']}", width,
                                     initial_indent="  ", subsequent_indent="    "))
    return "\n".join(out)
