"""HTTP API. Start with:  uvicorn patent_search.api.app:app --port 8000
Interactive docs (try queries in a browser): http://localhost:8000/docs"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from ..config import get_config
from ..schemas import SearchRequest, SearchResponse
from ..search.engine import Engine

STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    eng = Engine(get_config())
    _ = eng.encoder                     # load the model once at startup, not on the first query
    STATE["engine"] = eng
    yield
    STATE.clear()


app = FastAPI(title="Patent Claim Search", version="1.0", lifespan=lifespan)


def _eng() -> Engine:
    return STATE["engine"]


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    try:
        return _eng().search(req)
    except LookupError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(422, str(e))


@app.get("/patents/{doc_number}")
def patent(doc_number: str) -> dict:
    try:
        return _eng().patent(doc_number)
    except LookupError as e:
        raise HTTPException(404, str(e))


@app.get("/data-issues")
def data_issues(status: str | None = "open", severity: str | None = None, issue_type: str | None = None,
                limit: int = 200) -> dict:
    return _eng().data_issues(status, severity, issue_type, limit)


@app.get("/health")
def health() -> dict:
    e = _eng()
    m = e.manifest
    return {"status": "ok", "patents": len(e.patents), "chunks": e.n_chunks, "embedding": m["embedding"],
            "index_created_at": m["created_at"], "config_hash": m["config_hash"],
            "patents_by_data_status": m["report"]["patents_by_data_status"]}
