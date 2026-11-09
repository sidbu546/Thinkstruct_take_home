"""Central configuration. Every tunable number lives here so the manifest can hash it.

Environment overrides (all optional):
  PS_INDEX_DIR      where the index is written/read (default ./index)
  PS_DEVICE         cpu | mps   (default cpu: bit-reproducible vectors)
  PS_EMBED_BACKEND  local | fake (fake = hashed bag-of-words vectors, for tests only)
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class ChunkingConfig:
    mode: str = "adaptive"                 # adaptive (the design) | fixed (baseline for the Part 3 eval)
    fixed_window_words: int = 200
    fixed_overlap_words: int = 40
    # claim tiers (DESIGN.md §2.3)
    simple_max_words: int = 60
    simple_max_elements: int = 2
    complex_min_depth: int = 2
    complex_min_elements: int = 11
    complex_min_words: int = 251
    chain_token_budget: int = 1536
    words_to_tokens: float = 1.35          # rough estimate, used only for budgeting
    # description passages
    heading_max_words: int = 15
    passage_target_dense: int = 150        # words, reference-numeral-dense text
    passage_target_narrative: int = 350    # words, narrative text
    passage_min_words: int = 40
    dense_numeral_ratio: float = 0.05
    merge_similarity: float = 0.10         # TF-IDF cosine between adjacent paragraphs
    boilerplate_weight: float = 0.5


@dataclass(frozen=True)
class SearchConfig:
    rrf_k: int = 60
    retrieve_per_list: int = 200
    exact_max_chunks: int = 50_000          # planner threshold (DESIGN.md §4.1)
    second_hit_bonus: float = 0.10
    kind_weights_text: dict = field(default_factory=lambda: {
        "abstract": 1.0, "claim": 1.0, "element": 1.0, "description": 1.0})
    # claim mode: tuned on the dev split (reports/kind_weight_grid_dev.log; Part 1 defaults were
    # abstract 0.8 / claim 1.0 / element 1.0 / description 0.6 -> dev MRR@10 0.644, these -> 0.907)
    kind_weights_claim: dict = field(default_factory=lambda: {
        "abstract": 1.0, "claim": 0.6, "element": 0.3, "description": 0.6})
    # element coverage thresholds (cosine), from scripts/calibrate_coverage.py on this index
    # (reports/coverage_calibration.json). found = 97.5th pct of same-subclass similarity,
    # weak = 99th pct of other-subclass similarity. Re-tuned in Part 3 against labels.
    coverage_found: float = 0.802
    coverage_weak: float = 0.744
    coverage_top_results: int = 10
    # two-phase search (DESIGN.md Part 3B): phase 1 top 100 -> bge-reranker-base (Apple GPU when available)
    # -> top 30. alpha/beta tuned on dev (reports/rerank_tuning_final_dev.json). Opt-in because the gain is
    # small and it costs several seconds per query (README §5).
    rerank_default: bool = False
    rerank_model: str = "bge"                   # key into `rerankers`
    rerankers: dict = field(default_factory=lambda: {
        "bge": ("BAAI/bge-reranker-base", "2cfc18c9415c912f9d8155881c133215df768a70")})
    rerank_depth: int = 100                     # phase 1 hands its top 100 patents to phase 2 ...
    rerank_top_k: int = 30                      # ... which returns the reranked top 30 (CLI default with --rerank)
    rerank_device: str = field(default_factory=lambda: os.environ.get("PS_RERANK_DEVICE", "auto"))
    # auto = Apple GPU (mps) if available, else cpu
    rerank_chunks_per_doc: int = 2              # best phase-1 chunks per patent scored by the cross-encoder
    rerank_budget_ms: int = 30000               # above the measured cold p95 (16.9 s) plus first-query warm-up
    rerank_alpha: float = 0.3                   # weight of z(rerank) vs z(phase-1)  (dev-tuned)
    rerank_beta: float = 0.1                    # claim mode: + beta * share of elements found (dev-tuned)
    rerank_batch: int = 16
    rerank_max_length: int = 512


@dataclass(frozen=True)
class Config:
    model_name: str = "Alibaba-NLP/gte-modernbert-base"
    model_revision: str = "e7f32e3c00f91d699e8c43b53106206bcc72bb22"
    max_seq_length: int = 2048
    embed_batch_size: int = 16
    device: str = field(default_factory=lambda: os.environ.get("PS_DEVICE", "cpu"))
    embed_backend: str = field(default_factory=lambda: os.environ.get("PS_EMBED_BACKEND", "local"))
    index_dir: Path = field(default_factory=lambda: Path(os.environ.get("PS_INDEX_DIR", ROOT / "index")))
    seed: int = 13
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    search: SearchConfig = field(default_factory=SearchConfig)

    @property
    def db_path(self) -> Path:
        return self.index_dir / "patents.db"

    @property
    def vectors_path(self) -> Path:
        return self.index_dir / "vectors.npy"

    @property
    def manifest_path(self) -> Path:
        return self.index_dir / "manifest.json"

    @property
    def cache_path(self) -> Path:
        return self.index_dir / "embed_cache.sqlite"

    @property
    def rerank_cache_path(self) -> Path | None:
        """Shared by all indexes (pairs are keyed by their text). Tests use the fake backend: no file."""
        return None if self.embed_backend == "fake" else ROOT / "cache" / "rerank_scores.sqlite"

    def hash(self) -> str:
        """Hash of everything that changes index contents (not paths/devices)."""
        d = {"model": self.model_name, "rev": self.model_revision, "max_seq": self.max_seq_length,
             "backend": self.embed_backend, "chunking": asdict(self.chunking)}
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]


def get_config(**overrides) -> Config:
    return Config(**overrides)
