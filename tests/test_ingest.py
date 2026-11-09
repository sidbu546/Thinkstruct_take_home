from patent_search.config import ChunkingConfig
from patent_search.ingest.chunk import claim_chunks, description_chunks
from patent_search.ingest.claims import assign_tier, parse_claims
from patent_search.ingest.load import RawPatent
from patent_search.ingest.normalize import normalize_text
from patent_search.ingest.validate import data_status, detect_issues

CFG = ChunkingConfig()


def test_numbers_ranges_canceled_and_inference():
    cl = parse_claims([
        "a spoke body having a middle segment; and two connecting segments",      # no number -> 1 inferred
        "2 - 4 . (canceled)",
        "5 . The spoke according to claim 1 , wherein the body is 1.5 mm thick.",
        "The spoke according to claim 5, wherein the segment is carbon.",          # no number -> 6 inferred
    ])
    by = {c.number: c for c in cl}
    assert sorted(by) == [1, 2, 3, 4, 5, 6]
    assert by[1].number_inferred and by[6].number_inferred and not by[5].number_inferred
    assert [by[n].status for n in (2, 3, 4)] == ["canceled"] * 3
    assert by[5].depends_on == [1] and by[6].depends_on == [5]
    assert by[6].depth == 2 and by[5].depth == 1 and by[1].depth == 0
    assert "1.5 mm" in by[5].text                     # decimal not taken as a claim number
    assert by[1].preamble_missing                     # starts lower-case: opening words lost


def test_multi_dependent_and_unresolved():
    cl = parse_claims(["1 . A tire comprising: a tread.", "2 . A tire of claim 1.",
                       "3 . The tire according to any one of claims 1 to 2, wherein x.",
                       "4 . The tire of claim 9, wherein y."])
    by = {c.number: c for c in cl}
    assert by[3].depends_on == [1, 2] and by[3].multi_dep
    assert by[4].unresolved_refs == [] and by[4].depends_on == []   # 9 > 4: not a valid back-reference
    cl = parse_claims(["1 . A tire comprising: a tread.", "2 . (canceled)", "3 . The tire of claim 2, wherein y."])
    assert {c.number: c for c in cl}[3].unresolved_refs == [2]


def test_preamble_and_elements():
    cl = parse_claims(["1 . A pneumatic tire comprising: a tread portion; a pair of sidewalls; and a pair of beads, "
                       "wherein the tread has a groove."])
    c = cl[0]
    assert c.preamble == "A pneumatic tire comprising" and not c.preamble_missing
    assert len(c.elements) == 4 and c.elements[-1].startswith("wherein")


def test_tiers():
    simple = parse_claims(["1 . A wheel comprising: a rim."])[0]
    assert assign_tier(simple, CFG) == "simple"
    chain = parse_claims(["1 . A wheel comprising: a rim; a hub; spokes.", "2 . The wheel of claim 1, wherein a.",
                          "3 . The wheel of claim 2, wherein b."])
    tiers = {c.number: assign_tier(c, CFG) for c in chain}
    assert tiers[2] == "moderate" and tiers[3] == "complex"


def test_dependent_chunk_carries_ancestor_chain():
    cl = parse_claims(["1 . A wheel comprising: a rim; a hub; and a set of spokes.", "2 . The wheel of claim 1, wherein ALPHA.",
                       "3 . The wheel of claim 2, wherein BETA."])
    chunks = {c.key: c for c in claim_chunks("D1", "WHEEL", cl, CFG)}
    c3 = chunks["D1|c3"]
    assert "Claim 1:" in c3.embed_text and "ALPHA" in c3.embed_text and "BETA" in c3.embed_text
    assert c3.context_claims == "1,2"
    assert "D1|c1|e0" in chunks and chunks["D1|c1|e0"].parent_key == "D1|c1"     # element chunks
    assert "D1|c2|e0" not in chunks                                              # none for dependents


def test_chain_budget_compresses_but_keeps_root_and_tail():
    long = " ".join(["word"] * 900)
    cl = parse_claims([f"1 . A wheel comprising: {long}.", f"2 . The wheel of claim 1, {long} wherein GAMMA.",
                       "3 . The wheel of claim 2, wherein DELTA."])
    c3 = [c for c in claim_chunks("D", "T", cl, CFG) if c.key == "D|c3"][0]
    assert c3.n_tokens_est <= CFG.chain_token_budget + 50
    assert "DELTA" in c3.embed_text and "Claim 1:" in c3.embed_text


def test_description_headings_merge_and_split():
    paras = ["BACKGROUND", "", "A tire has a tread and a carcass with cords.", "The tread of the tire has grooves.",
             "DETAILED DESCRIPTION", " ".join(["The spoke 12 connects the hub 14 to the rim 16."] * 60)]
    out = description_chunks("D", "TIRE", paras, CFG)
    assert out[0].section == "BACKGROUND" and out[0].para_span == "2-3"        # merged, empty dropped
    long_parts = [c for c in out if c.section == "DETAILED DESCRIPTION"]
    assert len(long_parts) > 1                                                 # long paragraph split
    assert all(len(c.display_text.split()) <= CFG.passage_target_dense + 20 for c in long_parts)


def test_normalize():
    assert normalize_text("As shown inFIG.1 , the claim 6 ,") == "As shown in FIG. 1, the claim 6,"


def _raw(**kw):
    base = dict(doc_number="X", title="T", abstract="abs", classification="B60B104FI", filing_week="2024-02-15",
                source_file="f", position=0, source_xml="x", bibtex="", claims=[], paragraphs=["p"],
                present_keys=frozenset({"title", "abstract", "classification", "claims", "detailed_description"}))
    base.update(kw)
    return RawPatent(**base)


def test_issues_flagged_never_dropped():
    p = _raw(claims=["an axle body; and a hub"], paragraphs=["short text"], abstract="")
    cl = parse_claims(p.claims)
    issues = detect_issues(p, cl, 2)
    types = {i["issue_type"] for i in issues}
    assert {"claim1_preamble_missing", "description_thin", "missing_field", "claim_number_inferred"} <= types
    assert data_status(issues) == "incomplete"                   # abstract missing
    p2 = _raw(claims=["1 . A hub comprising: a body."], paragraphs=[" ".join(["w"] * 80)])
    assert data_status(detect_issues(p2, parse_claims(p2.claims), 80)) == "complete"


def test_lost_opening_on_later_claims_is_flagged():
    p = _raw(claims=["1 . A hub comprising: a body.", "2 . The hub of claim 1, wherein x.",
                     "3 . a plurality of openings is provided in the body."], paragraphs=[" ".join(["w"] * 80)])
    issues = detect_issues(p, parse_claims(p.claims), 80)
    lost = [i for i in issues if i["issue_type"] == "claim_opening_missing"]
    assert lost and '"claims": [3]' in lost[0]["detail"]
