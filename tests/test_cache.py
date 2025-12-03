from pathlib import Path

from api.cache import (
    CACHE_SCHEMA_VERSION,
    QueryCache,
    decode_research_state,
    encode_research_state,
)
from api.state import ResearchState


def test_query_cache_roundtrip(tmp_path):
    cache_path = tmp_path / "cache.sqlite3"
    cache = QueryCache(cache_path, ttl_seconds=60)

    state = ResearchState(query="Cached question")
    payload = {"version": CACHE_SCHEMA_VERSION, "state": encode_research_state(state)}
    cache.set("Cached question", payload)

    loaded = cache.get("Cached question")
    assert loaded is not None
    assert loaded["version"] == CACHE_SCHEMA_VERSION

    restored = decode_research_state(loaded)
    assert restored.query == "Cached question"
    assert restored.warnings == []
    assert restored.time_sensitive is False


def test_query_cache_handles_unknown_query(tmp_path):
    cache = QueryCache(tmp_path / "cache.sqlite3", ttl_seconds=60)
    assert cache.get("missing") is None
