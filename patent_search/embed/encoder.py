"""Local, in-process text encoder (DESIGN.md §2.4). No external service.

backend "local": sentence-transformers model, pinned revision, CPU by default.
backend "fake":  deterministic hashed bag-of-words vectors. Used by the test suite so tests
                 run in seconds without downloading a model. Never used for reported results.
"""
from __future__ import annotations

import hashlib
import os
import re

import numpy as np

from ..config import Config

_FAKE_DIM = 256
_WORD = re.compile(r"[a-z0-9]+")


class FakeEncoder:
    name = "fake-hashed-bow"
    revision = "v1"
    dim = _FAKE_DIM
    device = "cpu"

    def encode(self, texts: list[str], batch_size: int = 32, show_progress: bool = False) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in _WORD.findall(t.lower()):
                h = int.from_bytes(hashlib.md5(w.encode()).digest()[:4], "little")
                out[i, h % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms

    def count_tokens(self, texts: list[str]) -> list[int]:
        return [len(_WORD.findall(t.lower())) for t in texts]


class LocalEncoder:
    def __init__(self, cfg: Config):
        import torch
        from sentence_transformers import SentenceTransformer

        torch.manual_seed(cfg.seed)
        self.name = cfg.model_name
        self.revision = cfg.model_revision
        self.device = cfg.device
        # float32 explicitly: the checkpoint loads as float16 by default, which is ~13x slower
        # on CPU and less reproducible.
        from pathlib import Path
        local = Path(cfg.model_name).is_dir()
        self.model = SentenceTransformer(cfg.model_name, revision=None if local else cfg.model_revision,
                                         device=cfg.device, model_kwargs={"dtype": torch.float32})
        self.model.max_seq_length = cfg.max_seq_length
        self.dim = int(self.model.get_embedding_dimension())
        self.torch_threads = torch.get_num_threads()

    def encode(self, texts: list[str], batch_size: int = 16, show_progress: bool = False) -> np.ndarray:
        import torch

        with torch.inference_mode():
            v = self.model.encode(list(texts), batch_size=batch_size, normalize_embeddings=True,
                                  convert_to_numpy=True, show_progress_bar=show_progress)
        return np.asarray(v, dtype=np.float32)

    def count_tokens(self, texts: list[str]) -> list[int]:
        tok = self.model.tokenizer
        return [len(ids) for ids in tok(list(texts), add_special_tokens=True, truncation=False)["input_ids"]]


def get_encoder(cfg: Config):
    if cfg.embed_backend == "fake":
        return FakeEncoder()
    if cfg.embed_backend != "local":
        raise ValueError(f"unknown embed backend {cfg.embed_backend!r} (use 'local' or 'fake')")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    return LocalEncoder(cfg)
