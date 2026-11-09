"""index/manifest.json: what was built, from what, with what (DESIGN.md §2.4)."""
from __future__ import annotations

import hashlib
import importlib.metadata as md
import json
import platform
import sys
from dataclasses import asdict
from datetime import datetime, timezone

import numpy as np

from .config import Config


def _sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _versions() -> dict:
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for pkg in ("numpy", "torch", "transformers", "sentence-transformers", "fastapi", "pydantic"):
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            out[pkg] = None
    return out


def write_manifest(cfg: Config, encoder, files: list[dict], report: dict, timings: dict,
                   chunks_digest: str) -> dict:
    vecs = np.load(cfg.vectors_path, mmap_mode="r")
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "embedding": {"backend": cfg.embed_backend, "model": encoder.name, "revision": encoder.revision,
                      "dim": int(encoder.dim), "device": encoder.device,
                      "torch_threads": getattr(encoder, "torch_threads", None),
                      "max_seq_length": cfg.max_seq_length, "normalized": True},
        "versions": _versions(),
        "config_hash": cfg.hash(),
        "config": {"chunking": asdict(cfg.chunking), "search": asdict(cfg.search), "seed": cfg.seed},
        "data_files": files,
        "chunks_digest": chunks_digest,
        "vectors": {"shape": list(vecs.shape), "dtype": str(vecs.dtype),
                    "sha256": _sha256_file(cfg.vectors_path)},
        "report": report,
        "timings": timings,
    }
    cfg.manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


def model_dir_revision(path) -> str:
    """Revision id for a local (fine-tuned) model directory: hash of its weight files."""
    from pathlib import Path
    h = hashlib.sha256()
    for f in sorted(Path(path).rglob("*.safetensors")):
        h.update(f.read_bytes())
    return "local-sha256:" + h.hexdigest()[:16]


def load_manifest(cfg: Config) -> dict:
    return json.loads(cfg.manifest_path.read_text())
