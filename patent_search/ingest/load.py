"""Read the weekly JSON files. Nothing is dropped silently: unreadable files and records
without an identifier are quarantined and reported; duplicates are counted and, if their
content differs, recorded as a data issue."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

_FILE_WEEK = re.compile(r"patents_ipa(\d{2})(\d{2})(\d{2})\.json$")
_XML_DATE = re.compile(r"-(\d{4})(\d{2})(\d{2})\.XML$", re.I)
EXPECTED_KEYS = ("title", "doc_number", "filename", "abstract", "detailed_description",
                 "claims", "bibtex", "classification")
MISSING_SENTINELS = {"", "N/A", "NA", "NONE", "NULL"}


@dataclass
class RawPatent:
    doc_number: str
    title: str
    abstract: str
    classification: str | None
    filing_week: str | None
    source_file: str
    position: int
    source_xml: str
    bibtex: str
    claims: list
    paragraphs: list
    present_keys: frozenset
    load_issues: list[dict] = field(default_factory=list)

    def completeness(self) -> tuple[int, int]:
        filled = sum(bool(x) for x in (self.title, self.abstract, self.classification,
                                       self.claims, self.paragraphs))
        size = len(self.abstract) + sum(len(str(c)) for c in self.claims) + \
            sum(len(str(p)) for p in self.paragraphs)
        return filled, size

    def content_hash(self) -> str:
        blob = json.dumps([self.title, self.abstract, self.claims, self.paragraphs,
                           self.classification], ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()


@dataclass
class LoadResult:
    patents: list[RawPatent]
    quarantine: list[dict]
    files: list[dict]            # {file, sha256, records}
    duplicates_exact: int = 0
    duplicates_conflict: int = 0


def _week(path: Path, xml_name: str) -> str | None:
    m = _FILE_WEEK.search(path.name)
    if m:
        return date(2000 + int(m.group(1)), int(m.group(2)), int(m.group(3))).isoformat()
    m = _XML_DATE.search(xml_name or "")
    if m:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3))).isoformat()
    return None


def _clean_str(v) -> str:
    s = "" if v is None else str(v).strip()
    return "" if s.upper() in MISSING_SENTINELS else s


def _as_list(v) -> list:
    if v is None:
        return []
    if isinstance(v, list):
        return v
    return [v]


def load_dir(data_dir: Path) -> LoadResult:
    files = sorted(Path(data_dir).glob("*.json"))
    if not files:
        raise FileNotFoundError(f"no *.json files in {data_dir}")
    by_doc: dict[str, RawPatent] = {}
    order: list[str] = []
    quarantine: list[dict] = []
    file_info: list[dict] = []
    dup_exact = dup_conflict = 0

    for f in files:
        raw_bytes = f.read_bytes()
        info = {"file": f.name, "sha256": hashlib.sha256(raw_bytes).hexdigest(), "records": 0}
        file_info.append(info)
        try:
            records = json.loads(raw_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            quarantine.append({"file": f.name, "position": None, "reason": f"unreadable file: {e}"})
            continue
        if not isinstance(records, list):
            records = [records]
        for i, r in enumerate(records):
            info["records"] += 1
            if not isinstance(r, dict):
                quarantine.append({"file": f.name, "position": i, "reason": "record is not an object"})
                continue
            doc = _clean_str(r.get("doc_number"))
            if not doc:
                quarantine.append({"file": f.name, "position": i, "reason": "no doc_number",
                                   "title": _clean_str(r.get("title"))[:120]})
                continue
            xml = _clean_str(r.get("filename"))
            p = RawPatent(
                doc_number=doc, title=_clean_str(r.get("title")), abstract=_clean_str(r.get("abstract")),
                classification=_clean_str(r.get("classification")) or None,
                filing_week=_week(f, xml), source_file=f.name, position=i, source_xml=xml,
                bibtex=_clean_str(r.get("bibtex")), claims=_as_list(r.get("claims")),
                paragraphs=_as_list(r.get("detailed_description")),
                present_keys=frozenset(k for k in EXPECTED_KEYS if k in r))
            if doc in by_doc:
                old = by_doc[doc]
                if old.content_hash() == p.content_hash():
                    dup_exact += 1
                    continue
                dup_conflict += 1
                keep, other = (p, old) if p.completeness() > old.completeness() else (old, p)
                keep.load_issues.append({
                    "issue_type": "duplicate_conflict", "severity": "medium",
                    "location": f"{other.source_file}[{other.position}]",
                    "detail": "another record with this doc_number has different content; "
                              "kept the more complete one"})
                by_doc[doc] = keep
                continue
            by_doc[doc] = p
            order.append(doc)

    return LoadResult(patents=[by_doc[d] for d in order], quarantine=quarantine, files=file_info,
                      duplicates_exact=dup_exact, duplicates_conflict=dup_conflict)
