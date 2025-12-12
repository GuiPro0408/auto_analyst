from tools import search


def test_get_backend_rejects_unknown_backend():
    assert search.get_backend("unknown") is None


def test_canonical_source_preserves_unknown():
    assert search.canonical_source("example") == "example"
    assert search.canonical_source("gemini") == search.SOURCE_GEMINI_GROUNDING
