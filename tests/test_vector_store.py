"""Tests for vector store implementations.

Tests FAISS and ChromaDB vector stores with fake embedders.
"""

import pytest

from api.state import Chunk
from vector_store.faiss_store import FaissVectorStore
from tests.conftest import FakeEmbedder


@pytest.mark.unit
def test_faiss_store_round_trip(monkeypatch):
    """FAISS store should store and retrieve chunks correctly."""
    monkeypatch.setattr(
        "vector_store.faiss_store.load_embedding_model",
        lambda model_name: FakeEmbedder(),
    )
    store = FaissVectorStore(model_name="fake")
    chunk = Chunk(
        id="c1",
        text="hello world",
        metadata={"title": "Title", "url": "http://example.com"},
    )
    store.upsert([chunk])
    results = store.query("hello", top_k=1)
    assert results
    assert results[0].chunk.id == "c1"


@pytest.mark.unit
def test_faiss_store_multiple_chunks(monkeypatch):
    """FAISS store should handle multiple chunks."""
    monkeypatch.setattr(
        "vector_store.faiss_store.load_embedding_model",
        lambda model_name: FakeEmbedder(),
    )
    store = FaissVectorStore(model_name="fake")
    chunks = [
        Chunk(id=f"c{i}", text=f"content {i}", metadata={"title": f"Doc {i}"})
        for i in range(5)
    ]
    store.upsert(chunks)
    results = store.query("content", top_k=3)
    assert len(results) == 3


@pytest.mark.unit
def test_faiss_store_empty_query(monkeypatch):
    """FAISS store should handle empty store queries."""
    monkeypatch.setattr(
        "vector_store.faiss_store.load_embedding_model",
        lambda model_name: FakeEmbedder(),
    )
    store = FaissVectorStore(model_name="fake")
    results = store.query("anything", top_k=5)
    assert results == []


@pytest.mark.unit
def test_faiss_store_metadata_preserved(monkeypatch):
    """FAISS store should preserve chunk metadata."""
    monkeypatch.setattr(
        "vector_store.faiss_store.load_embedding_model",
        lambda model_name: FakeEmbedder(),
    )
    store = FaissVectorStore(model_name="fake")
    chunk = Chunk(
        id="c1",
        text="test content",
        metadata={
            "title": "Test Title",
            "url": "http://example.com/test",
            "custom_field": "custom_value",
        },
    )
    store.upsert([chunk])
    results = store.query("test", top_k=1)

    assert results[0].chunk.metadata["title"] == "Test Title"
    assert results[0].chunk.metadata["url"] == "http://example.com/test"
    assert results[0].chunk.metadata["custom_field"] == "custom_value"


@pytest.mark.unit
def test_faiss_store_duplicate_upsert(monkeypatch):
    """FAISS store should handle duplicate chunk IDs."""
    monkeypatch.setattr(
        "vector_store.faiss_store.load_embedding_model",
        lambda model_name: FakeEmbedder(),
    )
    store = FaissVectorStore(model_name="fake")
    chunk1 = Chunk(id="c1", text="original content", metadata={})
    chunk2 = Chunk(id="c1", text="updated content", metadata={})

    store.upsert([chunk1])
    store.upsert([chunk2])

    results = store.query("content", top_k=5)
    # Implementation may keep both or dedupe - just ensure no crash
    assert len(results) >= 1


@pytest.mark.unit
def test_faiss_store_clear(monkeypatch):
    """FAISS store clear should remove all chunks."""
    monkeypatch.setattr(
        "vector_store.faiss_store.load_embedding_model",
        lambda model_name: FakeEmbedder(),
    )
    store = FaissVectorStore(model_name="fake")
    chunks = [Chunk(id=f"c{i}", text=f"content {i}", metadata={}) for i in range(3)]
    store.upsert(chunks)

    # Verify chunks exist
    results = store.query("content", top_k=5)
    assert len(results) > 0

    # Clear and verify empty
    store.clear()
    results = store.query("content", top_k=5)
    assert len(results) == 0


@pytest.mark.unit
def test_faiss_store_top_k_limiting(monkeypatch):
    """FAISS store should respect top_k parameter."""
    monkeypatch.setattr(
        "vector_store.faiss_store.load_embedding_model",
        lambda model_name: FakeEmbedder(),
    )
    store = FaissVectorStore(model_name="fake")
    chunks = [Chunk(id=f"c{i}", text=f"content {i}", metadata={}) for i in range(10)]
    store.upsert(chunks)

    results = store.query("content", top_k=3)
    assert len(results) == 3

    results = store.query("content", top_k=1)
    assert len(results) == 1
