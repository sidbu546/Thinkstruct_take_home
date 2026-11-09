"""The JSON contract (DESIGN.md §5). Every output (API, CLI text, Markdown/CSV exports, eval
run files) is built from SearchResponse; nothing downstream parses text."""
from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1.1"   # 1.1: two-phase search fields (additive)


class KeywordFilter(BaseModel):
    terms: list[str]
    field: Literal["title", "abstract", "both"] = "both"
    op: Literal["all", "any", "phrase"] = "all"


class TitleFilter(BaseModel):
    value: str
    mode: Literal["exact", "phrase"] = "exact"


class Filters(BaseModel):
    cpc_prefix: list[str] = Field(default_factory=list, description="classification starts with any")
    cpc_exclude: list[str] = Field(default_factory=list, description="classification starts with none")
    keywords: KeywordFilter | None = None
    title: TitleFilter | None = None
    week_from: date | None = None
    week_to: date | None = None

    def is_empty(self) -> bool:
        return not (self.cpc_prefix or self.cpc_exclude or self.keywords or self.title
                    or self.week_from or self.week_to)


class SearchRequest(BaseModel):
    query: str | None = None
    mode: Literal["text", "claim", "patent"] = "text"
    doc_number: str | None = Field(None, description="required when mode = patent")
    filters: Filters = Field(default_factory=Filters)
    top_k: int = Field(30, ge=1, le=200)
    facets: bool = True
    exclude_docs: list[str] = Field(default_factory=list, description="never return these patents")
    retrieval: Literal["hybrid", "dense", "bm25"] = "hybrid"
    rerank: bool | None = Field(None, description="two-phase search; None = server default")
    rerank_model: Literal["bge"] | None = None
    rerank_depth: int | None = Field(None, ge=1, le=200)
    chunk_kinds: list[Literal["abstract", "claim", "element", "description"]] | None = Field(
        None, description="search only these chunk kinds (default: all)")
    hide_claims_of: list[str] = Field(default_factory=list,
                                      description="evaluation: hide these patents' own claim/element chunks")


class Evidence(BaseModel):
    kind: str
    claim_number: int | None = None
    element_idx: int | None = None
    para_span: str | None = None
    text: str


class BestMatch(BaseModel):
    kind: str
    chunk_id: int
    claim_number: int | None = None
    claim_number_inferred: bool | None = None
    element_idx: int | None = None
    element_text: str | None = None
    para_span: str | None = None
    section: str | None = None
    tier: str | None = None
    context_claims: list[int] = Field(default_factory=list)
    display_text: str
    highlights: list[list[int]] = Field(default_factory=list)


class CoverageItem(BaseModel):
    element: int
    element_text: str
    status: Literal["found", "weak", "not_found", "undetermined"]
    score: float | None = None
    evidence: Evidence | None = None
    reason: str | None = None
    issue_ids: list[int] = Field(default_factory=list)


class DataIssueOut(BaseModel):
    id: int
    type: str
    severity: str
    status: str
    location: str | None = None
    notice: str


class Signals(BaseModel):
    dense_rank: int | None = None
    bm25_rank: int | None = None
    rrf: float = 0.0
    phase1_rank: int | None = None
    rerank: float | None = None
    coverage: float | None = None


class Result(BaseModel):
    rank: int
    doc_number: str
    title: str
    classification: str | None
    filing_week: str | None
    record_version: int
    score: float
    signals: Signals
    best_match: BestMatch | None
    coverage: list[CoverageItem] | None = None
    data_status: str
    data_issues: list[DataIssueOut] = Field(default_factory=list)
    matched_filters: list[str] = Field(default_factory=list)


class QueryInfo(BaseModel):
    mode: str
    text: str | None = None
    tier: str | None = None
    elements: list[str] = Field(default_factory=list)


class ClaimMatch(BaseModel):
    doc_number: str
    title: str
    score: float
    match: BestMatch
    coverage: list[CoverageItem] | None = None
    data_status: str


class ClaimMapEntry(BaseModel):
    claim_number: int
    claim_text: str
    tier: str | None
    matches: list[ClaimMatch]


class Facets(BaseModel):
    cpc_subclass: dict[str, int] = Field(default_factory=dict)
    cpc_code: dict[str, int] = Field(default_factory=dict)
    filing_week: dict[str, int] = Field(default_factory=dict)
    data_status: dict[str, int] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    schema_version: str = SCHEMA_VERSION
    request_id: str
    took_ms: int
    pipeline: list[str]
    query: QueryInfo
    filters_applied: dict
    total_matched: int
    data_notice: str | None = None
    filter_notices: list[str] = Field(default_factory=list)
    results: list[Result]
    facets: Facets | None = None
    claim_map: list[ClaimMapEntry] | None = None
    stage_ms: dict[str, int] = Field(default_factory=dict)
    rerank: dict | None = None
    pipeline_notices: list[str] = Field(default_factory=list)
