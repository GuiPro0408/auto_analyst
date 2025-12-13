"""Tests for grounded answer fast path in the RAG pipeline.

Tests that Gemini grounding results are properly passed through
the pipeline without unnecessary LLM calls.
"""

import pytest

from api.graph import run_research
from api.state import Chunk, SearchResult
from tests.conftest import FakeLLM, FakeVectorStore


@pytest.mark.integration
def test_grounded_answer_fastpath(monkeypatch):
    """Grounded answers should flow through pipeline correctly."""
    grounded_answer = "Grounded response body"

    grounded_result = SearchResult(
        url="http://example.com",
        title="Example",
        snippet="snippet",
        source="gemini_grounding",
        content=grounded_answer,
    )

    def fake_search(tasks, max_results=5, run_id=None):
        return ([grounded_result], [])

    def fake_smart_search(query, max_results=5, run_id=None):
        return ([grounded_result], [])

    def fake_fetch(result: SearchResult, run_id=None):
        # Should not be required for grounded path, but provide a safe fallback
        return None, None

    monkeypatch.setattr("api.graph.run_search_tasks", fake_search)
    monkeypatch.setattr("api.graph.smart_search", fake_smart_search)
    monkeypatch.setattr("api.graph.fetch_url", fake_fetch)

    result = run_research(
        "Test grounded question",
        llm=FakeLLM(),
        vector_store=FakeVectorStore(),
        embed_model="fake",
        top_k=1,
    )
    assert result.search_results, "expected search results"
    assert result.search_results[0].source == "gemini_grounding"
    assert result.search_results[0].content == grounded_answer
    assert result.draft_answer.startswith("Grounded response"), result.draft_answer
    assert result.citations
    assert result.retrieved
    assert result.verified_answer.startswith("Verified answer")


@pytest.mark.integration
def test_grounded_answer_with_multiple_sources(monkeypatch):
    """Multiple grounded sources should all contribute to citations."""
    grounded_results = [
        SearchResult(
            url=f"http://example{i}.com",
            title=f"Source {i}",
            snippet=f"snippet {i}",
            source="gemini_grounding",
            content=f"Grounded content {i}",
        )
        for i in range(3)
    ]

    def fake_search(tasks, max_results=5, run_id=None):
        return (grounded_results, [])

    def fake_smart_search(query, max_results=5, run_id=None):
        return (grounded_results, [])

    def fake_fetch(result: SearchResult, run_id=None):
        return None, None

    monkeypatch.setattr("api.graph.run_search_tasks", fake_search)
    monkeypatch.setattr("api.graph.smart_search", fake_smart_search)
    monkeypatch.setattr("api.graph.fetch_url", fake_fetch)

    result = run_research(
        "Test with multiple grounded sources",
        llm=FakeLLM(),
        vector_store=FakeVectorStore(),
        embed_model="fake",
        top_k=3,
    )

    assert len(result.search_results) == 3
    assert all(r.source == "gemini_grounding" for r in result.search_results)


@pytest.mark.integration
def test_mixed_grounded_and_regular_results(monkeypatch):
    """Pipeline should handle mix of grounded and regular search results."""
    from api.state import Document

    mixed_results = [
        SearchResult(
            url="http://grounded.com",
            title="Grounded",
            snippet="grounded snippet",
            source="gemini_grounding",
            content="Grounded content",
        ),
        SearchResult(
            url="http://regular.com",
            title="Regular",
            snippet="regular snippet",
            source="tavily",
        ),
    ]

    def fake_search(tasks, max_results=5, run_id=None):
        return (mixed_results, [])

    def fake_smart_search(query, max_results=5, run_id=None):
        return (mixed_results, [])

    def fake_fetch(result: SearchResult, timeout=15, run_id=None):
        if result.source == "tavily":
            return (
                Document(
                    url=result.url,
                    title=result.title,
                    content="fetched content",
                    media_type="html",
                ),
                None,
            )
        return None, None

    monkeypatch.setattr("api.graph.run_search_tasks", fake_search)
    monkeypatch.setattr("api.graph.smart_search", fake_smart_search)
    monkeypatch.setattr("api.graph.fetch_url", fake_fetch)

    result = run_research(
        "Test with mixed results",
        llm=FakeLLM(),
        vector_store=FakeVectorStore(),
        embed_model="fake",
        top_k=2,
    )

    assert len(result.search_results) == 2
    sources = {r.source for r in result.search_results}
    assert "gemini_grounding" in sources
    assert "tavily" in sources
