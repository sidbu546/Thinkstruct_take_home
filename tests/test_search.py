import json

import numpy as np
import pytest

from patent_search.render import chart, text
from patent_search.schemas import SearchRequest, SearchResponse
from patent_search.search.filters import keyword_fts_query
from patent_search.search.planner import dense_topk


def test_keyword_query_builder():
    assert keyword_fts_query(["spoke", "carbon"], "abstract", "all") == 'abstract : ("spoke" AND "carbon")'
    assert keyword_fts_query(["a b"], "both", "any") == '{title abstract} : ("a b")'
    assert keyword_fts_query(["sound", "absorbing"], "title", "phrase") == 'title : ("sound absorbing")'


def test_planner_subset_equals_masked_full_scan():
    rng = np.random.default_rng(0)
    v = rng.normal(size=(500, 16)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    q = v[:2]
    allowed = np.arange(0, 500, 3)
    a, pa = dense_topk(v, q, allowed, 10, exact_max=10_000)
    b, pb = dense_topk(v, q, allowed, 10, exact_max=1)
    assert pa == "exact_subset" and pb == "full_scan_masked"
    assert [[r for r, _ in x] for x in a] == [[r for r, _ in x] for x in b]
    assert all(r % 3 == 0 for x in a for r, _ in x)


def _some_claim(engine):
    row = engine.con.execute("SELECT doc_number, text FROM claims WHERE status='active' AND depends_on='[]' "
                             "AND length(text) > 200 ORDER BY doc_number LIMIT 1").fetchone()
    return row["doc_number"], row["text"]


def test_claim_copy_finds_its_patent_first(engine):
    doc, claim = _some_claim(engine)
    r = engine.search(SearchRequest(query=claim, mode="claim", top_k=5))
    assert r.results[0].doc_number == doc
    assert r.results[0].coverage and all(c.status in ("found", "weak", "not_found", "undetermined")
                                         for c in r.results[0].coverage)


def test_cpc_filter_is_strict(engine):
    r = engine.search(SearchRequest(query="wheel rim", filters={"cpc_prefix": ["B60B"]}, top_k=50))
    assert r.results and all(x.classification.startswith("B60B") for x in r.results)
    assert set(r.facets.cpc_subclass) == {"B60B"}
    r2 = engine.search(SearchRequest(query="wheel rim", filters={"cpc_exclude": ["B60B"]}, top_k=50))
    assert all(not (x.classification or "").startswith("B60B") for x in r2.results)


def test_title_and_keyword_filters(engine):
    title = engine.con.execute("SELECT title FROM patents ORDER BY doc_number LIMIT 1").fetchone()[0]
    expected = {d for d, p in engine.patents.items() if p["title_norm"] == title.upper().strip()}
    r = engine.search(SearchRequest(filters={"title": {"value": title.lower(), "mode": "exact"}}, top_k=200))
    assert {x.doc_number for x in r.results} == expected
    r = engine.search(SearchRequest(query="vehicle", filters={"keywords": {"terms": ["tire"], "field": "abstract"}},
                                    top_k=200))
    for x in r.results:
        assert "tire" in engine.patents[x.doc_number]["abstract"].lower()


def test_filter_only_request_and_requires_something(engine):
    r = engine.search(SearchRequest(filters={"cpc_prefix": ["B60C"]}, top_k=5))
    assert r.results and r.pipeline[-1] == "order:filing_week_desc"
    with pytest.raises(ValueError):
        engine.search(SearchRequest())


def test_patent_mode_excludes_self_and_maps_claims(engine):
    doc, _ = _some_claim(engine)
    r = engine.search(SearchRequest(mode="patent", doc_number=doc, top_k=10))
    assert doc not in {x.doc_number for x in r.results}
    assert r.claim_map and all(m.doc_number != doc for e in r.claim_map for m in e.matches)
    with pytest.raises(LookupError):
        engine.search(SearchRequest(mode="patent", doc_number="nope"))


def test_data_gaps_are_surfaced_and_never_not_found(engine):
    thin = [d for d, iss in engine.issues.items() if any(i["issue_type"] == "description_thin" for i in iss)]
    assert thin, "fixture data should contain a thin-description patent"
    doc = thin[0]
    claim = engine.con.execute("SELECT text FROM claims WHERE doc_number=? AND status='active' "
                               "ORDER BY claim_number LIMIT 1", (doc,)).fetchone()[0]
    r = engine.search(SearchRequest(query=claim + "; a completely unrelated quantum flux capacitor element",
                                    mode="claim", top_k=10))
    hit = next(x for x in r.results if x.doc_number == doc)
    assert hit.data_status != "complete" and hit.data_issues
    assert all(c.status != "not_found" for c in hit.coverage)
    assert any("pending data research" in i.notice for i in hit.data_issues)
    assert r.data_notice and "provisional" in r.data_notice


def test_outputs_render_and_roundtrip(engine):
    doc, claim = _some_claim(engine)
    r = engine.search(SearchRequest(query=claim, mode="claim", top_k=5))
    again = SearchResponse.model_validate(json.loads(r.model_dump_json()))
    assert again.results[0].doc_number == r.results[0].doc_number
    assert "Pipeline:" in text.render(r)
    assert chart.to_markdown(r).startswith("# Claim chart")
    assert chart.to_csv(r).splitlines()[0].startswith("query_claim,element")


def test_api(engine, fake_cfg, monkeypatch):
    from fastapi.testclient import TestClient
    monkeypatch.setenv("PS_INDEX_DIR", str(fake_cfg.index_dir))
    monkeypatch.setenv("PS_EMBED_BACKEND", "fake")
    from patent_search.api.app import app
    with TestClient(app) as c:
        assert c.get("/health").json()["status"] == "ok"
        r = c.post("/search", json={"query": "pneumatic tire tread", "filters": {"cpc_prefix": ["B60C"]}})
        assert r.status_code == 200 and r.json()["schema_version"] == "1.1"
        assert c.get("/patents/nope").status_code == 404
        assert c.post("/search", json={"mode": "patent"}).status_code == 422
        assert "summary" in c.get("/data-issues").json()


def test_eval_metrics():
    from patent_search.eval.metrics import aggregate, pair_metrics, paired_delta, per_query
    assert per_query(1)["MRR@10"] == 1.0 and per_query(None)["R@100"] == 0.0
    assert abs(per_query(3)["nDCG@10"] - 0.5) < 1e-9 and per_query(11)["MRR@10"] == 0.0
    agg = aggregate([1, 2, None, 50])
    assert agg["n"] == 4 and abs(agg["MRR@10"]["mean"] - 0.375) < 1e-9 and agg["R@100"]["mean"] == 0.75
    lo, hi = agg["MRR@10"]["ci95"]
    assert lo <= 0.375 <= hi
    assert paired_delta([2, 2], [1, 1])["delta"] == 0.5
    pm = pair_metrics([0.9, 0.6], [[0.5, 0.95], [0.7]])      # wins: 0.9>0.5 only
    assert abs(pm["triplet_accuracy"] - 1 / 3) < 1e-9 and pm["margin"] < 0.1


def test_eval_splits_keep_families_together(engine):
    from patent_search.eval.build_sets import assign_splits, families
    fam = families(engine)
    split = assign_splits(fam, 13)
    by_fam = {}
    for d, f in fam.items():
        by_fam.setdefault(f, set()).add(split[d])
    assert all(len(s) == 1 for s in by_fam.values())
    assert set(split.values()) <= {"train", "dev", "test"}


def test_training_triplets_never_use_dev_or_test_patents(engine):
    """Regression: hard negatives for training must come from the training split only."""
    from patent_search.eval.build_sets import assign_splits, build_pairs, families
    fam = families(engine)
    split = assign_splits(fam, 13)
    recs = build_pairs(engine, split, fam, "train", 13, max_anchors=40, log=lambda *_: None)
    assert recs
    for r in recs:
        assert split[r["doc"]] == "train"
        assert all(split[d] == "train" for d in r["negative_docs"])
