from patent_search.schemas import SearchRequest
from patent_search.render import text


def _claim(engine):
    r = engine.con.execute("SELECT doc_number, text FROM claims WHERE status='active' AND depends_on='[]' "
                           "AND length(text) > 200 ORDER BY doc_number LIMIT 1").fetchone()
    return r["doc_number"], r["text"]


def test_rerank_only_reorders_phase1_candidates(engine):
    _, claim = _claim(engine)
    p1 = engine.search(SearchRequest(query=claim, mode="claim", top_k=10, rerank=False))
    p2 = engine.search(SearchRequest(query=claim, mode="claim", top_k=10, rerank=True, rerank_depth=10))
    # phase 2 reorders the phase-1 top-10; it can never bring in anything else
    assert {x.doc_number for x in p2.results} == {x.doc_number for x in p1.results}
    assert p2.rerank and p2.rerank["scored"] == len(p1.results)
    assert [x.doc_number for x in p2.results] != [] and p2.results[0].signals.rerank is not None
    assert any(s.startswith("rerank:") for s in p2.pipeline)
    assert all(x.signals.phase1_rank is not None for x in p2.results)
    assert "Phase 2:" in text.render(p2)


def test_rerank_respects_filters(engine):
    r = engine.search(SearchRequest(query="wheel rim spoke", filters={"cpc_prefix": ["B60B"]}, rerank=True,
                                    top_k=20))
    assert r.results and all(x.classification.startswith("B60B") for x in r.results)


def test_rerank_budget_is_reported_not_silent(engine):
    _, claim = _claim(engine)
    from dataclasses import replace
    old = engine.cfg
    object.__setattr__(engine, "cfg", replace(old, search=replace(old.search, rerank_budget_ms=-1)))
    try:
        r = engine.search(SearchRequest(query=claim, mode="claim", rerank=True))
        assert r.pipeline_notices and "phase-1 order" in r.pipeline_notices[0]
        assert r.rerank["scored"] == 0
    finally:
        object.__setattr__(engine, "cfg", old)


def test_hide_claims_of_masks_own_claims(engine):
    doc, claim = _claim(engine)
    r = engine.search(SearchRequest(query=claim, mode="claim", top_k=50, hide_claims_of=[doc]))
    for x in r.results:
        if x.doc_number == doc:
            assert x.best_match.kind in ("abstract", "description")


def test_retrieval_modes(engine):
    for mode in ("dense", "bm25"):
        r = engine.search(SearchRequest(query="pneumatic tire tread groove", retrieval=mode))
        assert r.results and r.pipeline[0].startswith("dense" if mode == "dense" else "bm25")


def test_rerank_eval_run_writes_jsonl_and_rescores_exactly(engine, fake_cfg, tmp_path, monkeypatch):
    """Regression: a reranked evaluation run must serialize its per-candidate scores (numpy floats
    once crashed json.dumps after all queries were scored) and rescoring must reproduce the ranking."""
    import json
    from patent_search.eval import runner
    eval_dir, runs_dir = tmp_path / "eval_data", tmp_path / "runs"
    eval_dir.mkdir()
    rows = engine.con.execute("SELECT doc_number, text FROM claims WHERE status='active' AND depends_on='[]' "
                              "AND length(text) > 150 ORDER BY doc_number LIMIT 4").fetchall()
    with open(eval_dir / "queries_dev.jsonl", "w") as f:
        for i, r in enumerate(rows):
            f.write(json.dumps({"qid": f"q{i}", "task": "T1", "text": r["text"], "source_doc": r["doc_number"],
                                "claim_number": 1, "tier": "moderate", "data_status": "partial"}) + "\n")
    monkeypatch.setattr(runner, "EVAL_DIR", eval_dir)
    monkeypatch.setattr(runner, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(runner, "METRICS_PATH", tmp_path / "metrics.json")
    monkeypatch.setitem(runner.EXPERIMENTS, "EX", {"index": "x", "retrieval": "hybrid", "rerank": True, "label": "t"})
    monkeypatch.setitem(runner._ENGINES, "x", engine)
    res = runner.run("EX", "dev", fake_cfg, rerank_model="bge", alpha=0.3, beta=0.1, depth=10, log=lambda *_: None)
    assert res["overall"]["n"] == 4 and not (runs_dir / "EX_dev.partial.jsonl").exists()
    lines = [json.loads(l) for l in open(runs_dir / "EX_dev.jsonl")]
    assert all(isinstance(c["p1"], float) for x in lines for c in x["candidates"])
    runner.rescore("EX_dev", "EX_re_dev", 0.3, 0.1)
    again = [json.loads(l) for l in open(runs_dir / "EX_re_dev.jsonl")]
    assert [x["ranking"][:10] for x in again] == [x["ranking"][:10] for x in lines]
