"""Tests for contextual chunking module."""

import pytest

from api.state import Chunk, Document
from tools.contextual_chunker import (
    CONTEXT_GENERATION_PROMPT,
    generate_chunk_context,
    contextualize_chunks,
)


class FakeLLM:
    """Fake LLM that returns predictable context."""

    def __init__(self, response: str = "This is from ACME Corp Q2 2023 filing."):
        self.response = response
        self.call_count = 0
        self.last_prompt = ""

    def __call__(self, prompt: str):
        self.call_count += 1
        self.last_prompt = prompt
        return [{"generated_text": self.response}]


@pytest.fixture
def sample_document():
    return Document(
        url="https://example.com/report",
        title="ACME Corp Q2 2023 Report",
        content="ACME Corporation reported strong growth in Q2 2023. "
        "Revenue increased by 15% compared to the previous quarter. "
        "The company's new product line contributed significantly to this growth.",
        media_type="html",
    )


@pytest.fixture
def sample_chunks(sample_document):
    return [
        Chunk(
            id="chunk-1",
            text="Revenue increased by 15% compared to the previous quarter.",
            metadata={"url": sample_document.url, "title": sample_document.title},
        ),
        Chunk(
            id="chunk-2",
            text="The company's new product line contributed significantly.",
            metadata={"url": sample_document.url, "title": sample_document.title},
        ),
    ]


def test_generate_chunk_context_success(sample_document, sample_chunks, monkeypatch):
    """Test successful context generation."""
    # Disable config check for test and force non-limited backend
    monkeypatch.setattr("tools.contextual_chunker.CONTEXTUAL_CHUNKS_ENABLED", True)
    monkeypatch.setattr("api.backend_utils.LLM_BACKEND", "gemini")

    llm = FakeLLM("This chunk is from ACME Corp's Q2 2023 financial report.")
    context = generate_chunk_context(
        sample_document, sample_chunks[0], llm=llm, run_id="test"
    )

    assert context == "This chunk is from ACME Corp's Q2 2023 financial report."
    assert llm.call_count == 1
    assert "ACME Corp" in llm.last_prompt


def test_generate_chunk_context_disabled(sample_document, sample_chunks, monkeypatch):
    """Test that context generation is skipped when disabled."""
    monkeypatch.setattr("tools.contextual_chunker.CONTEXTUAL_CHUNKS_ENABLED", False)

    llm = FakeLLM()
    context = generate_chunk_context(sample_document, sample_chunks[0], llm=llm)

    assert context == ""
    assert llm.call_count == 0


def test_generate_chunk_context_invalid_length(
    sample_document, sample_chunks, monkeypatch
):
    """Test that too short/long contexts are rejected."""
    monkeypatch.setattr("tools.contextual_chunker.CONTEXTUAL_CHUNKS_ENABLED", True)
    monkeypatch.setattr("api.backend_utils.LLM_BACKEND", "gemini")

    # Too short
    llm = FakeLLM("Hi")
    context = generate_chunk_context(sample_document, sample_chunks[0], llm=llm)
    assert context == ""

    # Too long (over 500 chars)
    llm = FakeLLM("x" * 600)
    context = generate_chunk_context(sample_document, sample_chunks[0], llm=llm)
    assert context == ""


def test_generate_chunk_context_llm_error(sample_document, sample_chunks, monkeypatch):
    """Test graceful handling of LLM errors."""
    monkeypatch.setattr("tools.contextual_chunker.CONTEXTUAL_CHUNKS_ENABLED", True)
    monkeypatch.setattr("api.backend_utils.LLM_BACKEND", "gemini")

    class FailingLLM:
        def __call__(self, prompt):
            raise RuntimeError("API error")

    context = generate_chunk_context(
        sample_document, sample_chunks[0], llm=FailingLLM()
    )
    assert context == ""


def test_contextualize_chunks_success(sample_document, sample_chunks, monkeypatch):
    """Test full chunk contextualization."""
    monkeypatch.setattr("tools.contextual_chunker.CONTEXTUAL_CHUNKS_ENABLED", True)
    monkeypatch.setattr("api.backend_utils.LLM_BACKEND", "gemini")

    llm = FakeLLM("Context: ACME Q2 2023 report section.")
    result = contextualize_chunks(sample_document, sample_chunks, llm=llm)

    assert len(result) == 2
    # Check context was prepended
    assert result[0].text.startswith("Context: ACME Q2 2023")
    assert "Revenue increased" in result[0].text
    # Check metadata updated
    assert result[0].metadata.get("has_context") is True
    original_length = result[0].metadata.get("original_text_length")
    assert original_length is not None and original_length > 0


def test_contextualize_chunks_disabled(sample_document, sample_chunks, monkeypatch):
    """Test that chunks pass through unchanged when disabled."""
    monkeypatch.setattr("tools.contextual_chunker.CONTEXTUAL_CHUNKS_ENABLED", False)

    llm = FakeLLM()
    result = contextualize_chunks(sample_document, sample_chunks, llm=llm)

    assert len(result) == 2
    assert result[0].text == sample_chunks[0].text
    assert llm.call_count == 0


def test_contextualize_chunks_empty_list(sample_document, monkeypatch):
    """Test handling of empty chunk list."""
    monkeypatch.setattr("tools.contextual_chunker.CONTEXTUAL_CHUNKS_ENABLED", True)
    monkeypatch.setattr("api.backend_utils.LLM_BACKEND", "gemini")

    result = contextualize_chunks(sample_document, [], llm=FakeLLM())
    assert result == []


def test_context_prompt_contains_placeholders():
    """Test that prompt template has required placeholders."""
    assert "{document_text}" in CONTEXT_GENERATION_PROMPT
    assert "{chunk_text}" in CONTEXT_GENERATION_PROMPT
