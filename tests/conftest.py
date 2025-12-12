"""Shared pytest fixtures for Auto-Analyst tests.

This module provides reusable mock objects and fixtures for testing
the RAG pipeline components without external I/O.
"""

from typing import Dict, List, Optional, Tuple

import pytest

from api.state import Chunk, Document, SearchResult
from vector_store.base import ScoredChunk, VectorStore


# =============================================================================
# FAKE LLM IMPLEMENTATIONS
# =============================================================================


class FakeLLM:
    """Basic fake LLM that returns canned responses.

    Returns different responses based on prompt content to simulate
    the generator/verifier flow.
    """

    def __call__(self, prompt: str) -> List[Dict[str, str]]:
        if "Verified answer:" in prompt:
            return [{"generated_text": "Verified answer: validated [1]"}]
        return [{"generated_text": "Answer: drafted [1]"}]


class CapturingLLM:
    """Fake LLM that captures prompts for assertion.

    Useful for testing prompt construction and context inclusion.
    """

    def __init__(self, response: str = "Answer: captured response [1]"):
        self.response = response
        self.last_prompt = ""
        self.call_count = 0
        self.prompts: List[str] = []

    def __call__(self, prompt: str) -> List[Dict[str, str]]:
        self.last_prompt = prompt
        self.prompts.append(prompt)
        self.call_count += 1
        return [{"generated_text": self.response}]


class ConfigurableLLM:
    """Fake LLM with configurable responses based on prompt patterns.

    Example:
        llm = ConfigurableLLM({
            "plan": "Answer: plan response",
            "verify": "Verified answer: checked",
        })
    """

    def __init__(self, responses: Dict[str, str]):
        self.responses = responses
        self.default_response = "Answer: default response [1]"

    def __call__(self, prompt: str) -> List[Dict[str, str]]:
        prompt_lower = prompt.lower()
        for pattern, response in self.responses.items():
            if pattern.lower() in prompt_lower:
                return [{"generated_text": response}]
        return [{"generated_text": self.default_response}]


# =============================================================================
# FAKE VECTOR STORE
# =============================================================================


class FakeVectorStore(VectorStore):
    """In-memory fake vector store for testing.

    Stores chunks and returns them with score 1.0 on query.
    """

    def __init__(self):
        self.chunks: List[Chunk] = []
        self.query_count = 0
        self.upsert_count = 0

    def clear(self) -> None:
        self.chunks = []

    def upsert(self, chunks: List[Chunk]) -> None:
        self.chunks.extend(chunks)
        self.upsert_count += 1

    def query(
        self, text: str, top_k: int = 5, *, run_id: Optional[str] = None
    ) -> List[ScoredChunk]:
        self.query_count += 1
        return [ScoredChunk(chunk=c, score=1.0) for c in self.chunks[:top_k]]


class ConfigurableFakeVectorStore(VectorStore):
    """Fake vector store with configurable scores.

    Useful for testing retrieval threshold logic.
    """

    def __init__(self, default_score: float = 1.0):
        self.chunks: List[Chunk] = []
        self.default_score = default_score
        self.scores_by_id: Dict[str, float] = {}

    def clear(self) -> None:
        self.chunks = []
        self.scores_by_id = {}

    def set_score(self, chunk_id: str, score: float) -> None:
        """Set a specific score for a chunk by ID."""
        self.scores_by_id[chunk_id] = score

    def upsert(self, chunks: List[Chunk]) -> None:
        self.chunks.extend(chunks)

    def query(
        self, text: str, top_k: int = 5, *, run_id: Optional[str] = None
    ) -> List[ScoredChunk]:
        result = []
        for c in self.chunks[:top_k]:
            score = self.scores_by_id.get(c.id, self.default_score)
            result.append(ScoredChunk(chunk=c, score=score))
        return result


# =============================================================================
# FAKE SEARCH AND FETCH FUNCTIONS
# =============================================================================


def fake_search(
    tasks, max_results: int = 5, run_id: Optional[str] = None
) -> Tuple[List[SearchResult], List[str]]:
    """Fake search function returning a single example result."""
    return (
        [
            SearchResult(
                url="http://example.com",
                title="Example",
                snippet="example snippet",
                source="tavily",
            )
        ],
        [],
    )


def fake_fetch(
    result: SearchResult, timeout: int = 15, run_id: Optional[str] = None
) -> Tuple[Optional[Document], Optional[str]]:
    """Fake fetch function returning a document from any SearchResult."""
    return (
        Document(
            url=result.url,
            title=result.title,
            content="context for testing",
            media_type="html",
        ),
        None,
    )


def fake_smart_search(
    query: str, max_results: int = 10, run_id: Optional[str] = None
) -> Tuple[List[SearchResult], List[str]]:
    """Fake smart_search returning the same as fake_search."""
    return fake_search([], max_results, run_id)


# =============================================================================
# PYTEST FIXTURES
# =============================================================================


@pytest.fixture
def fake_llm():
    """Fixture providing a basic FakeLLM instance."""
    return FakeLLM()


@pytest.fixture
def capturing_llm():
    """Fixture providing a CapturingLLM instance."""
    return CapturingLLM()


@pytest.fixture
def fake_vector_store():
    """Fixture providing a FakeVectorStore instance."""
    return FakeVectorStore()


@pytest.fixture
def sample_chunk():
    """Fixture providing a sample Chunk for testing."""
    return Chunk(
        id="test-chunk-1",
        text="This is sample context text for testing.",
        metadata={"title": "Test Doc", "url": "http://example.com/test"},
    )


@pytest.fixture
def sample_chunks():
    """Fixture providing multiple sample Chunks."""
    return [
        Chunk(
            id="chunk-1",
            text="First chunk of context about topic A.",
            metadata={"title": "Doc A", "url": "http://example.com/a"},
        ),
        Chunk(
            id="chunk-2",
            text="Second chunk of context about topic B.",
            metadata={"title": "Doc B", "url": "http://example.com/b"},
        ),
        Chunk(
            id="chunk-3",
            text="Third chunk with additional information.",
            metadata={"title": "Doc C", "url": "http://example.com/c"},
        ),
    ]


@pytest.fixture
def sample_search_result():
    """Fixture providing a sample SearchResult."""
    return SearchResult(
        url="http://example.com/article",
        title="Example Article",
        snippet="This is an example article snippet.",
        source="tavily",
    )


@pytest.fixture
def sample_document():
    """Fixture providing a sample Document."""
    return Document(
        url="http://example.com/article",
        title="Example Article",
        content="Full article content for testing purposes.",
        media_type="html",
    )


@pytest.fixture
def mock_search_and_fetch(monkeypatch):
    """Fixture that patches search and fetch functions with fakes."""
    monkeypatch.setattr("api.graph.run_search_tasks", fake_search)
    monkeypatch.setattr("api.graph.fetch_url", fake_fetch)
    monkeypatch.setattr("api.graph.smart_search", fake_smart_search)
