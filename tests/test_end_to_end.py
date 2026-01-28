"""End-to-end integration tests for the RAG pipeline.

Tests the complete research workflow from query to verified answer,
using mocked external dependencies (search, fetch, LLM, vector store).
"""

import pytest

from api.graph import run_research
from api.state import Document, SearchResult
from tests.conftest import FakeLLM, FakeVectorStore


@pytest.mark.integration
def test_end_to_end_pipeline(mock_search_and_fetch, monkeypatch):
    """Test complete pipeline from query to verified answer."""
    # Force non-limited backend for full verification flow
    monkeypatch.setattr("api.backend_utils.LLM_BACKEND", "gemini")

    fake_llm = FakeLLM()
    fake_store = FakeVectorStore()
    result = run_research(
        "Test question",
        llm=fake_llm,
        vector_store=fake_store,
        embed_model="fake",
        top_k=1,
    )

    assert result.plan
    assert result.search_results
    assert result.documents
    assert result.retrieved
    assert result.citations
    assert result.verified_answer.startswith("validated")
    assert result.conversation_history
    assert result.conversation_history[-1].query == "Test question"


@pytest.mark.integration
def test_end_to_end_with_empty_search_results(monkeypatch):
    """Test pipeline handles empty search results gracefully."""

    def empty_search(tasks, max_results=5, run_id=None):
        return ([], [])

    def fake_fetch(result, run_id=None):
        return (None, None)

    def empty_smart_search(query, max_results=5, run_id=None):
        return ([], [])

    monkeypatch.setattr("api.graph.run_search_tasks", empty_search)
    monkeypatch.setattr("api.graph.fetch_url", fake_fetch)
    monkeypatch.setattr("api.graph.smart_search", empty_smart_search)

    fake_llm = FakeLLM()
    fake_store = FakeVectorStore()
    result = run_research(
        "Test question with no results",
        llm=fake_llm,
        vector_store=fake_store,
        embed_model="fake",
        top_k=1,
    )

    # Pipeline should complete without crashing
    assert result.query == "Test question with no results"
    # May have warnings about no results
    assert result.errors == [] or "No search results" in str(result.warnings)


@pytest.mark.integration
def test_end_to_end_with_multiple_search_results(monkeypatch):
    """Test pipeline handles multiple search results."""

    def multi_search(tasks, max_results=5, run_id=None):
        return (
            [
                SearchResult(
                    url=f"http://example{i}.com",
                    title=f"Example {i}",
                    snippet=f"snippet {i}",
                    source="tavily",
                )
                for i in range(3)
            ],
            [],
        )

    def multi_fetch(result, timeout=15, run_id=None):
        return (
            Document(
                url=result.url,
                title=result.title,
                content=f"context for {result.title}",
                media_type="html",
            ),
            None,
        )

    def multi_smart_search(query, max_results=5, run_id=None):
        return multi_search([], max_results, run_id)

    monkeypatch.setattr("api.graph.run_search_tasks", multi_search)
    monkeypatch.setattr("api.graph.fetch_url", multi_fetch)
    monkeypatch.setattr("api.graph.smart_search", multi_smart_search)

    fake_llm = FakeLLM()
    fake_store = FakeVectorStore()
    result = run_research(
        "Test with multiple results",
        llm=fake_llm,
        vector_store=fake_store,
        embed_model="fake",
        top_k=3,
    )

    assert len(result.search_results) == 3
    assert len(result.documents) == 3
