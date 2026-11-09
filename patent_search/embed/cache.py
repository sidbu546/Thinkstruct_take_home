"""Content-hash embedding cache: key = sha256(model, revision, text). A rebuild after a data
fix only encodes chunks whose text changed (DESIGN.md §2.4)."""
from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import numpy as np


class EmbeddingCache:
    def __init__(self, path: Path, model: str, revision: str, dim: int):
        self.con = sqlite3.connect(path)
        self.con.execute("CREATE TABLE IF NOT EXISTS emb (key TEXT PRIMARY KEY, dim INTEGER, vec BLOB)")
        self.prefix = f"{model}@{revision}\n"
        self.dim = dim

    def _key(self, text: str) -> str:
        return hashlib.sha256((self.prefix + text).encode()).hexdigest()

    def get_many(self, texts: list[str]) -> dict[int, np.ndarray]:
        found: dict[int, np.ndarray] = {}
        keys = [self._key(t) for t in texts]
        for start in range(0, len(keys), 900):
            batch = keys[start:start + 900]
            q = f"SELECT key, dim, vec FROM emb WHERE key IN ({','.join('?' * len(batch))})"
            rows = {k: (d, v) for k, d, v in self.con.execute(q, batch)}
            for j, k in enumerate(batch):
                if k in rows and rows[k][0] == self.dim:
                    found[start + j] = np.frombuffer(rows[k][1], dtype=np.float32)
        return found

    def put_many(self, texts: list[str], vecs: np.ndarray):
        self.con.executemany("INSERT OR REPLACE INTO emb VALUES (?,?,?)",
                             [(self._key(t), self.dim, v.astype(np.float32).tobytes())
                              for t, v in zip(texts, vecs)])
        self.con.commit()

    def close(self):
        self.con.close()
