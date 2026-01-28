"""Tests for streaming functionality in the research pipeline."""

import pytest
from typing import TYPE_CHECKING, Generator, List, Dict, Any

from api.state import Chunk, ConversationTurn, SearchResult, Document
from vector_store.base import ScoredChunk, VectorStore

if TYPE_CHECKING:
    pass


class FakeStreamingLLM:
    """Fake LLM that yields tokens one by one for testing streaming."""

    def __init__(self, response: str, token_size: int = 5):
        self.response = response
        self.token_size = token_size
        self.last_prompt = ""
        self.call_count = 0

    def __call__(self, prompt: str) -> List[Dict[str, str]]:
        """Synchronous call for compatibility."""
        self.last_prompt = prompt
        self.call_count += 1
        return [{"generated_text": self.response}]

    def stream(self, prompt: str) -> Generator[str, None, None]:
        """Stream tokens from the response."""
        self.last_prompt = prompt
        self.call_count += 1
        # Yield tokens of token_size characters
        for i in range(0, len(self.response), self.token_size):
            yield self.response[i : i + self.token_size]


class FakeNonStreamingLLM:
    """Fake LLM without streaming support."""

    def __init__(self, response: str):
        self.response = response

    def __call__(self, prompt: str) -> List[Dict[str, str]]:
        return [{"generated_text": self.response}]


class FakeVectorStore(VectorStore):
    """Fake vector store for testing."""

    def __init__(self):
        self.chunks: List[Chunk] = []

    def clear(self) -> None:
        self.chunks = []

    def upsert(self, chunks: List[Chunk]) -> None:
        self.chunks.extend(chunks)

    def query(
        self, text: str, top_k: int = 5, *, run_id: str | None = None
    ) -> List[ScoredChunk]:
        return [ScoredChunk(chunk=c, score=1.0) for c in self.chunks[:top_k]]


# =============================================================================
# GeminiLLM.stream() Tests
# =============================================================================


def test_fake_streaming_llm_yields_tokens():
    """Test that FakeStreamingLLM yields tokens correctly."""
    llm = FakeStreamingLLM("Hello, world!", token_size=3)

    tokens = list(llm.stream("test prompt"))

    assert tokens == ["Hel", "lo,", " wo", "rld", "!"]
    assert llm.last_prompt == "test prompt"
    assert llm.call_count == 1


def test_fake_streaming_llm_sync_call():
    """Test that FakeStreamingLLM works with sync calls too."""
    llm = FakeStreamingLLM("Answer: test response")

    result = llm("test prompt")

    assert result == [{"generated_text": "Answer: test response"}]


# =============================================================================
# generate_answer_stream() Tests
# =============================================================================


def test_generate_answer_stream_yields_partial_text():
    """Test that generate_answer_stream yields partial text during streaming."""
    from tools.generator import generate_answer_stream

    llm = FakeStreamingLLM(
        "Answer: This is a test response with [1] citation.", token_size=10
    )
    chunks = [
        Chunk(
            id="1",
            text="Test context content",
            metadata={"title": "Source", "url": "http://example.com"},
        )
    ]

    results = list(generate_answer_stream(llm, "What is test?", chunks))

    # Should have multiple partial yields plus final
    assert len(results) > 1

    # All but last should be incomplete
    for partial, is_complete, citations in results[:-1]:
        assert is_complete is False
        assert citations == []

    # Last should be complete with citations
    final_text, is_complete, citations = results[-1]
    assert is_complete is True
    assert len(citations) > 0


def test_generate_answer_stream_strips_prompt_echo():
    """Test that generate_answer_stream strips 'Answer:' prompt echo."""
    from tools.generator import generate_answer_stream

    llm = FakeStreamingLLM("Answer: The actual answer here.", token_size=10)
    chunks = [
        Chunk(
            id="1",
            text="Context",
            metadata={"title": "Source", "url": "http://example.com"},
        )
    ]

    results = list(generate_answer_stream(llm, "Question?", chunks))
    final_text, _, _ = results[-1]

    # Should not start with "Answer:"
    assert not final_text.startswith("Answer:")
    assert "actual answer" in final_text.lower()


def test_generate_answer_stream_fallback_for_non_streaming_llm():
    """Test fallback to sync generation for LLMs without stream method."""
    from tools.generator import generate_answer_stream

    llm = FakeNonStreamingLLM("Answer: Sync response.")
    chunks = [
        Chunk(
            id="1",
            text="Context",
            metadata={"title": "Source", "url": "http://example.com"},
        )
    ]

    results = list(generate_answer_stream(llm, "Question?", chunks))

    # Should have exactly one result (the complete answer)
    assert len(results) == 1
    final_text, is_complete, citations = results[0]
    assert is_complete is True


def test_generate_answer_stream_empty_context():
    """Test generate_answer_stream with no context."""
    from tools.generator import generate_answer_stream

    llm = FakeStreamingLLM("Answer: test", token_size=5)
    chunks = []

    results = list(generate_answer_stream(llm, "Question?", chunks))

    assert len(results) == 1
    final_text, is_complete, citations = results[0]
    assert is_complete is True
    assert "no context" in final_text.lower()


# =============================================================================
# verify_answer_stream() Tests
# =============================================================================


def test_verify_answer_stream_yields_partial_text():
    """Test that verify_answer_stream yields partial text."""
    from tools.generator import verify_answer_stream

    llm = FakeStreamingLLM("Verified answer: The verified response.", token_size=10)
    chunks = [
        Chunk(
            id="1",
            text="Context",
            metadata={"title": "Source", "url": "http://example.com"},
        )
    ]

    results = list(verify_answer_stream(llm, "Draft answer", "Question?", chunks))

    # Should have multiple partial yields plus final
    assert len(results) > 1

    # All but last should be incomplete
    for partial, is_complete in results[:-1]:
        assert is_complete is False

    # Last should be complete
    final_text, is_complete = results[-1]
    assert is_complete is True


def test_verify_answer_stream_fallback_for_non_streaming_llm():
    """Test fallback to sync verification for LLMs without stream method."""
    from tools.generator import verify_answer_stream

    llm = FakeNonStreamingLLM("Verified answer: Sync verified.")
    chunks = [
        Chunk(
            id="1",
            text="Context",
            metadata={"title": "Source", "url": "http://example.com"},
        )
    ]

    results = list(verify_answer_stream(llm, "Draft answer", "Question?", chunks))

    # Should have exactly one result
    assert len(results) == 1
    final_text, is_complete = results[0]
    assert is_complete is True


# =============================================================================
# run_research_streaming() Tests
# =============================================================================


def test_run_research_streaming_yields_step_events(monkeypatch):
    """Test that run_research_streaming yields step events for each node."""
    from api.graph import run_research_streaming

    # Mock cache to avoid hitting cached results
    class MockCacheManager:
        def __init__(self, *args, **kwargs):
            pass

        def get_cached_result(self, *args, **kwargs):
            return None  # Never return cached result

        def save_result(self, *args, **kwargs):
            pass

    monkeypatch.setattr("api.graph.CacheManager", MockCacheManager)

    # Mock search and fetch to avoid network calls
    def fake_smart_search(query, max_results=5, run_id=None):
        return (
            [
                SearchResult(
                    url="http://example.com",
                    title="Test",
                    snippet="test snippet",
                    source="tavily",
                )
            ],
            [],
        )

    def fake_fetch(result, run_id=None):
        return (
            Document(
                url=result.url,
                title=result.title,
                content="Test content for testing.",
                media_type="html",
            ),
            None,
        )

    def fake_fetch_parallel(results, max_workers=4, run_id=None):
        docs = []
        for r in results:
            doc, _ = fake_fetch(r, run_id)
            if doc:
                docs.append(doc)
        return docs, []

    monkeypatch.setattr("api.graph.smart_search", fake_smart_search)
    monkeypatch.setattr(
        "api.graph.run_search_tasks", lambda tasks, **kw: fake_smart_search("", **kw)
    )
    monkeypatch.setattr("api.graph.fetch_url", fake_fetch)
    monkeypatch.setattr("api.graph.fetch_documents_parallel", fake_fetch_parallel)

    llm = FakeStreamingLLM("Answer: Test answer with [1] citation.", token_size=10)
    store = FakeVectorStore()

    events = list(
        run_research_streaming(
            "What is test?",
            llm=llm,
            vector_store=store,
            top_k=3,
        )
    )

    # Should have step events
    step_events = [e for e in events if e.get("type") == "step"]
    assert len(step_events) > 0

    # Should have token events
    token_events = [e for e in events if e.get("type") == "token"]
    assert len(token_events) > 0

    # Should have complete event at the end
    complete_events = [e for e in events if e.get("type") == "complete"]
    assert len(complete_events) == 1
    assert complete_events[0].get("result") is not None


def test_run_research_streaming_returns_research_state(monkeypatch):
    """Test that run_research_streaming returns a valid ResearchState."""
    from api.graph import run_research_streaming
    from api.state import ResearchState

    # Mock cache to avoid hitting cached results
    class MockCacheManager:
        def __init__(self, *args, **kwargs):
            pass

        def get_cached_result(self, *args, **kwargs):
            return None

        def save_result(self, *args, **kwargs):
            pass

    monkeypatch.setattr("api.graph.CacheManager", MockCacheManager)

    def fake_smart_search(query, max_results=5, run_id=None):
        return (
            [
                SearchResult(
                    url="http://example.com",
                    title="Test",
                    snippet="test snippet",
                    source="tavily",
                )
            ],
            [],
        )

    def fake_fetch_parallel(results, max_workers=4, run_id=None):
        docs = [
            Document(
                url=r.url,
                title=r.title,
                content="Test content.",
                media_type="html",
            )
            for r in results
        ]
        return docs, []

    monkeypatch.setattr("api.graph.smart_search", fake_smart_search)
    monkeypatch.setattr(
        "api.graph.run_search_tasks", lambda tasks, **kw: fake_smart_search("", **kw)
    )
    monkeypatch.setattr("api.graph.fetch_documents_parallel", fake_fetch_parallel)

    llm = FakeStreamingLLM("Answer: Final answer.", token_size=10)
    store = FakeVectorStore()

    events = list(
        run_research_streaming(
            "Test query",
            llm=llm,
            vector_store=store,
            top_k=3,
        )
    )

    # Get the final result
    complete_event = next(e for e in events if e.get("type") == "complete")
    result = complete_event.get("result")

    assert isinstance(result, ResearchState)
    assert result.query == "Test query"
    assert result.run_id != ""


def test_run_research_streaming_token_events_have_phase(monkeypatch):
    """Test that token events include the phase (generate or verify)."""
    from api.graph import run_research_streaming

    # Mock cache to avoid hitting cached results
    class MockCacheManager:
        def __init__(self, *args, **kwargs):
            pass

        def get_cached_result(self, *args, **kwargs):
            return None

        def save_result(self, *args, **kwargs):
            pass

    monkeypatch.setattr("api.graph.CacheManager", MockCacheManager)

    def fake_smart_search(query, max_results=5, run_id=None):
        return (
            [
                SearchResult(
                    url="http://example.com",
                    title="Test",
                    snippet="test",
                    source="tavily",
                )
            ],
            [],
        )

    def fake_fetch_parallel(results, max_workers=4, run_id=None):
        return [
            Document(url=r.url, title=r.title, content="Content.", media_type="html")
            for r in results
        ], []

    monkeypatch.setattr("api.graph.smart_search", fake_smart_search)
    monkeypatch.setattr(
        "api.graph.run_search_tasks", lambda tasks, **kw: fake_smart_search("", **kw)
    )
    monkeypatch.setattr("api.graph.fetch_documents_parallel", fake_fetch_parallel)

    llm = FakeStreamingLLM("Answer: Test.", token_size=5)
    store = FakeVectorStore()

    events = list(run_research_streaming("Query", llm=llm, vector_store=store, top_k=3))

    token_events = [e for e in events if e.get("type") == "token"]

    # All token events should have a phase
    for event in token_events:
        assert "phase" in event
        assert event["phase"] in ("generate", "verify")


# =============================================================================
# Integration Tests
# =============================================================================


def test_streaming_preserves_citations(monkeypatch):
    """Test that streaming correctly preserves citation information."""
    from tools.generator import generate_answer_stream

    llm = FakeStreamingLLM(
        "Answer: First point [1]. Second point [2].",
        token_size=10,
    )
    chunks = [
        Chunk(
            id="1",
            text="First context",
            metadata={"title": "Source 1", "url": "http://example1.com"},
        ),
        Chunk(
            id="2",
            text="Second context",
            metadata={"title": "Source 2", "url": "http://example2.com"},
        ),
    ]

    results = list(generate_answer_stream(llm, "Question?", chunks))
    final_text, is_complete, citations = results[-1]

    assert is_complete is True
    assert len(citations) == 2
    assert citations[0]["url"] == "http://example1.com"
    assert citations[1]["url"] == "http://example2.com"


def test_streaming_handles_conversation_context():
    """Test that streaming respects conversation context."""
    from tools.generator import generate_answer_stream

    llm = FakeStreamingLLM("Answer: Contextual answer.", token_size=10)
    chunks = [
        Chunk(
            id="1",
            text="Context",
            metadata={"title": "Source", "url": "http://example.com"},
        )
    ]

    results = list(
        generate_answer_stream(
            llm,
            "Follow-up question?",
            chunks,
            conversation_context="Previous: We discussed X.",
        )
    )

    # Check that conversation context was included in the prompt
    assert (
        "conversation" in llm.last_prompt.lower() or "prior" in llm.last_prompt.lower()
    )
