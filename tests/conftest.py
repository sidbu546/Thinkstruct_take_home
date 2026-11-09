"""Shared fixture: a small index (first 4 weekly files) built with the fake encoder, so the
suite runs in seconds without downloading a model. The fake encoder is never used for results."""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from patent_search.config import ROOT, Config

DATA = ROOT / "data" / "patent_data_small"


@pytest.fixture(scope="session")
def small_data(tmp_path_factory) -> Path:
    if not DATA.exists():
        pytest.skip("data/patent_data_small not extracted (unzip UTF-8patent_data_small.zip)")
    d = tmp_path_factory.mktemp("data")
    for f in sorted(DATA.glob("*.json"))[:4]:
        shutil.copy(f, d / f.name)
    return d


@pytest.fixture(scope="session")
def fake_cfg(tmp_path_factory) -> Config:
    return Config(embed_backend="fake", device="cpu", index_dir=tmp_path_factory.mktemp("index"))


@pytest.fixture(scope="session")
def engine(small_data, fake_cfg):
    from patent_search.index.build import build_index
    from patent_search.search.engine import Engine
    build_index(small_data, fake_cfg, log=lambda *_: None)
    return Engine(fake_cfg)
