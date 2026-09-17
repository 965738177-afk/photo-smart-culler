import pytest
from uuid import uuid4


@pytest.fixture(autouse=True)
def isolate_analysis_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("EVENT_CULL_CACHE_PATH", str(tmp_path / f"analysis-cache-{uuid4().hex}.json"))
