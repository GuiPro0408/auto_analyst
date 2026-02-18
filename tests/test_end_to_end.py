"""End-to-end integration tests for the RAG pipeline.

Tests the complete research workflow from query to verified answer,
using mocked external dependencies (search, fetch, LLM, vector store).
"""

import pytest

from api.graph import run_research
from api.state import Document, SearchResult
from tests.conftest import FakeLLM, FakeVectorStore


def _disable_cache(monkeypatch):
    class MockCacheManager:
        def __init__(self, *args, **kwargs):
            pass

        def get_cached_result(self, *args, **kwargs):
            return None

        def save_result(self, *args, **kwargs):
            pass

    monkeypatch.setattr("api.graph.CacheManager", MockCacheManager)


@pytest.mark.integration
def test_end_to_end_pipeline(mock_search_and_fetch, monkeypatch):
    """Test complete pipeline from query to verified answer."""
    _disable_cache(monkeypatch)
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
    _disable_cache(monkeypatch)

    def empty_search(tasks, max_results=5, run_id=None, time_sensitive=None, **kwargs):
        return ([], [])

    def fake_fetch(result, run_id=None):
        return (None, None)

    def empty_fetch_parallel(results, max_workers=4, run_id=None):
        return ([], [])

    def empty_smart_search(query, max_results=5, run_id=None):
        return ([], [])

    monkeypatch.setattr("api.graph.run_search_tasks", empty_search)
    monkeypatch.setattr("api.graph.fetch_url", fake_fetch)
    monkeypatch.setattr("api.graph.fetch_documents_parallel", empty_fetch_parallel)
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
    _disable_cache(monkeypatch)

    def multi_search(tasks, max_results=5, run_id=None, time_sensitive=None, **kwargs):
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

    def multi_fetch_parallel(results, max_workers=4, run_id=None):
        docs = []
        for result in results:
            document, _ = multi_fetch(result, run_id=run_id)
            docs.append(document)
        return docs, []

    def multi_smart_search(query, max_results=5, run_id=None):
        return multi_search([], max_results, run_id)

    monkeypatch.setattr("api.graph.run_search_tasks", multi_search)
    monkeypatch.setattr("api.graph.fetch_url", multi_fetch)
    monkeypatch.setattr("api.graph.fetch_documents_parallel", multi_fetch_parallel)
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


@pytest.mark.integration
def test_run_research_uses_supplied_run_id(mock_search_and_fetch, monkeypatch):
    """Non-streaming API should preserve caller-provided run ID."""
    _disable_cache(monkeypatch)
    fake_llm = FakeLLM()
    fake_store = FakeVectorStore()
    result = run_research(
        "Test question",
        llm=fake_llm,
        vector_store=fake_store,
        embed_model="fake",
        top_k=1,
        run_id="run-id-abc123",
    )

    assert result.run_id == "run-id-abc123"
