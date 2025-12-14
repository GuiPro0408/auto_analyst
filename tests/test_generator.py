"""Tests for answer generation and verification.

Tests the LLM-based answer generation, verification, and citation building.
"""

import pytest

from api.state import Chunk
from tools.generator import (
    build_citations,
    generate_answer,
    verify_answer,
    _generate_fallback_answer,
    _clean_snippet,
)
from tests.conftest import FakeLLM, CapturingLLM


@pytest.mark.unit
def test_generate_answer_builds_citations():
    """Answer generation should include proper citations."""
    chunk = Chunk(
        id="1",
        text="context text",
        metadata={"title": "Doc", "url": "http://example.com"},
    )
    llm = FakeLLM()
    answer, citations = generate_answer(llm, "question", [chunk])
    assert isinstance(answer, str) and len(answer) > 0
    assert citations and citations[0]["marker"] == "[1]"


@pytest.mark.unit
def test_verify_answer_pass_through():
    """Verification should pass through valid answers."""
    chunk = Chunk(
        id="1",
        text="context text",
        metadata={"title": "Doc", "url": "http://example.com"},
    )
    llm = FakeLLM()
    verified = verify_answer(llm, "Draft", "question", [chunk])
    assert verified


@pytest.mark.unit
def test_generate_answer_receives_conversation_context():
    """Generator should include conversation context in prompt."""
    chunk = Chunk(
        id="1",
        text="context text",
        metadata={"title": "Doc", "url": "http://example.com"},
    )
    llm = CapturingLLM("Answer: ok [1]")
    context = "Turn 1: Q: Solar roofs\nA: Details"
    generate_answer(llm, "What about it?", [chunk], conversation_context=context)
    assert "Prior conversation summary" in llm.last_prompt


@pytest.mark.unit
def test_verify_answer_receives_conversation_context():
    """Verifier should include conversation context in prompt."""
    chunk = Chunk(
        id="1",
        text="context text",
        metadata={"title": "Doc", "url": "http://example.com"},
    )
    llm = CapturingLLM("Verified answer: ok [1]")
    context = "Turn 1: Q: Solar roofs\nA: Details"
    verify_answer(
        llm,
        "Draft answer",
        "What about it?",
        [chunk],
        conversation_context=context,
    )
    assert "Prior conversation summary" in llm.last_prompt


@pytest.mark.unit
def test_generate_answer_with_empty_chunks():
    """Generator should handle empty chunk list gracefully."""
    llm = FakeLLM()
    answer, citations = generate_answer(llm, "question", [])
    # Should still produce some answer
    assert isinstance(answer, str)
    assert isinstance(citations, list)


@pytest.mark.unit
def test_generate_answer_multiple_chunks():
    """Generator should handle multiple chunks with proper citations."""
    chunks = [
        Chunk(
            id=str(i),
            text=f"context text {i}",
            metadata={"title": f"Doc {i}", "url": f"http://example{i}.com"},
        )
        for i in range(3)
    ]
    llm = FakeLLM()
    answer, citations = generate_answer(llm, "question about multiple sources", chunks)
    assert isinstance(answer, str)
    assert len(citations) >= 1


@pytest.mark.unit
def test_build_citations_deduplicates_urls():
    """Citation builder should deduplicate citations with same URL."""
    chunks = [
        Chunk(
            id="1",
            text="first chunk",
            metadata={"title": "Doc", "url": "http://example.com"},
        ),
        Chunk(
            id="2",
            text="second chunk",
            metadata={"title": "Doc", "url": "http://example.com"},
        ),
    ]
    # build_citations returns (answer_str, citations_list)
    _, citations = build_citations(chunks)
    urls = [c["url"] for c in citations]
    # Should deduplicate - only one citation for same URL
    assert len(urls) == 1
    assert urls[0] == "http://example.com"
    assert all(isinstance(c, dict) for c in citations)


class TestFallbackAnswer:
    """Tests for the improved fallback answer generation."""

    @pytest.mark.unit
    def test_fallback_empty_chunks(self):
        """Fallback should handle empty chunks gracefully."""
        answer = _generate_fallback_answer("test query", [])
        assert "couldn't find" in answer.lower() or "insufficient" in answer.lower()

    @pytest.mark.unit
    def test_fallback_recommendation_query(self):
        """Fallback should detect recommendation queries and format nicely."""
        chunks = [
            Chunk(
                id="1",
                text="Solo Leveling is a great anime. The Apothecary Diaries is also popular.",
                metadata={
                    "title": "Best Anime 2025",
                    "url": "https://example.com/anime",
                },
            ),
        ]
        answer = _generate_fallback_answer("What anime should I watch?", chunks)
        assert "**" in answer  # Has bold formatting
        assert "Solo Leveling" in answer or "Apothecary" in answer

    @pytest.mark.unit
    def test_fallback_cleans_boilerplate(self):
        """Fallback should clean URLs and boilerplate from snippets."""
        chunks = [
            Chunk(
                id="1",
                text="Sign in to your account. https://example.com/link Check out this anime.",
                metadata={"title": "Test", "url": "https://example.com"},
            ),
        ]
        answer = _generate_fallback_answer("test", chunks)
        # Should not contain raw URLs in the snippet
        assert "https://example.com/link" not in answer

    @pytest.mark.unit
    def test_fallback_deduplicates_sources(self):
        """Fallback should not repeat the same source multiple times."""
        chunks = [
            Chunk(
                id="1",
                text="This is substantial content from the first chunk about anime recommendations for the winter season.",
                metadata={"title": "Same Source", "url": "https://example.com"},
            ),
            Chunk(
                id="2",
                text="This is different substantial content from the second chunk about more anime series to watch.",
                metadata={"title": "Same Source", "url": "https://example.com"},
            ),
        ]
        answer = _generate_fallback_answer("test", chunks)
        # Should only appear once since URL is duplicated
        assert answer.count("Same Source") == 1

    @pytest.mark.unit
    def test_clean_snippet_removes_noise(self):
        """Clean snippet should remove rating codes and boilerplate."""
        raw = "5.80 4.4K Add to My List StudioILCA Source Original Great anime content here."
        cleaned = _clean_snippet(raw)
        assert "5.80" not in cleaned
        assert "Add to My List" not in cleaned
        assert "content" in cleaned.lower()

    @pytest.mark.unit
    def test_clean_snippet_preserves_sentences(self):
        """Clean snippet should preserve complete sentences."""
        raw = "This is a complete sentence. Here is another one. And a third."
        cleaned = _clean_snippet(raw, max_len=50)
        # Should end at a sentence boundary, not mid-word
        assert cleaned.endswith(".") or cleaned.endswith("...")
