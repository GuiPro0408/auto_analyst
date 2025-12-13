"""Tests for document chunking.

Tests the TextChunker class for splitting documents into overlapping chunks.
"""

import pytest

from api.state import Document
from tools.chunker import TextChunker


@pytest.mark.unit
def test_chunker_creates_overlapping_chunks():
    """Chunker should create multiple overlapping chunks for long documents."""
    text = " ".join([f"word{i}" for i in range(1500)])
    doc = Document(url="http://example.com", title="Example", content=text)
    chunker = TextChunker(chunk_size=200, overlap=50)
    chunks = chunker.chunk_document(doc)
    assert chunks
    # Expect multiple overlapping chunks
    assert len(chunks) > 1
    assert all("chunk_index" in c.metadata for c in chunks)


@pytest.mark.unit
def test_chunker_empty_document():
    """Chunker should handle empty documents gracefully."""
    doc = Document(url="http://example.com", title="Empty", content="")
    chunker = TextChunker(chunk_size=200, overlap=50)
    chunks = chunker.chunk_document(doc)
    assert chunks == [] or len(chunks) == 1


@pytest.mark.unit
def test_chunker_short_document():
    """Chunker should return single chunk for short documents."""
    doc = Document(
        url="http://example.com", title="Short", content="This is a short document."
    )
    chunker = TextChunker(chunk_size=200, overlap=50)
    chunks = chunker.chunk_document(doc)
    assert len(chunks) == 1
    assert chunks[0].text == "This is a short document."


@pytest.mark.unit
def test_chunker_preserves_metadata():
    """Chunker should preserve document metadata in chunks."""
    doc = Document(
        url="http://example.com/article",
        title="Test Article",
        content=" ".join([f"word{i}" for i in range(500)]),
    )
    chunker = TextChunker(chunk_size=100, overlap=20)
    chunks = chunker.chunk_document(doc)

    for chunk in chunks:
        assert chunk.metadata.get("url") == "http://example.com/article"
        assert chunk.metadata.get("title") == "Test Article"


@pytest.mark.unit
def test_chunker_overlap_works():
    """Chunks should have overlapping content when overlap is set."""
    # Create a document with distinct words
    words = [f"word{i:04d}" for i in range(200)]
    doc = Document(url="http://example.com", title="Test", content=" ".join(words))
    chunker = TextChunker(chunk_size=50, overlap=10)
    chunks = chunker.chunk_document(doc)

    # If we have multiple chunks, check for overlap
    if len(chunks) > 1:
        # The end of chunk 0 should overlap with the start of chunk 1
        chunk0_words = set(chunks[0].text.split())
        chunk1_words = set(chunks[1].text.split())
        overlap = chunk0_words & chunk1_words
        # There should be some overlap
        assert len(overlap) > 0


@pytest.mark.unit
def test_chunker_special_characters():
    """Chunker should handle special characters in content."""
    content = "Hello! @#$%^&*() Special <characters> & entities ñ é ü"
    doc = Document(url="http://example.com", title="Special", content=content)
    chunker = TextChunker(chunk_size=200, overlap=50)
    chunks = chunker.chunk_document(doc)
    assert len(chunks) >= 1
    # Content should be preserved
    assert "Special" in chunks[0].text or "characters" in chunks[0].text


@pytest.mark.unit
def test_chunker_unicode_content():
    """Chunker should handle unicode content correctly."""
    content = "日本語テスト 中文测试 한국어 테스트 Emoji: 🎉🚀"
    doc = Document(url="http://example.com", title="Unicode", content=content)
    chunker = TextChunker(chunk_size=200, overlap=50)
    chunks = chunker.chunk_document(doc)
    assert len(chunks) >= 1


@pytest.mark.unit
def test_chunker_newlines_and_whitespace():
    """Chunker should handle various whitespace correctly."""
    content = "Line 1\n\nLine 2\n\n\n\nLine 3\t\tTabbed content"
    doc = Document(url="http://example.com", title="Whitespace", content=content)
    chunker = TextChunker(chunk_size=200, overlap=50)
    chunks = chunker.chunk_document(doc)
    assert len(chunks) >= 1


@pytest.mark.unit
@pytest.mark.parametrize(
    "chunk_size,overlap",
    [
        (100, 20),
        (200, 50),
        (500, 100),
        (1000, 200),
    ],
)
def test_chunker_various_sizes(chunk_size, overlap):
    """Chunker should work with various chunk sizes and overlaps."""
    text = " ".join([f"word{i}" for i in range(2000)])
    doc = Document(url="http://example.com", title="Test", content=text)
    chunker = TextChunker(chunk_size=chunk_size, overlap=overlap)
    chunks = chunker.chunk_document(doc)
    assert len(chunks) >= 1
    # All chunks should have proper IDs
    for i, chunk in enumerate(chunks):
        assert chunk.id is not None
        assert chunk.metadata.get("chunk_index") == i


@pytest.mark.unit
def test_chunker_unique_chunk_ids():
    """Each chunk should have a unique ID."""
    text = " ".join([f"word{i}" for i in range(1000)])
    doc = Document(url="http://example.com", title="Test", content=text)
    chunker = TextChunker(chunk_size=100, overlap=20)
    chunks = chunker.chunk_document(doc)

    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids)), "Chunk IDs should be unique"
