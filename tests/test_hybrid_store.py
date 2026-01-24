"""Tests for hybrid vector store (BM25 + embeddings)."""

import pytest

from api.state import Chunk
from vector_store.bm25_store import BM25Store, _tokenize
from vector_store.hybrid_store import HybridVectorStore, _reciprocal_rank_fusion
from vector_store.base import ScoredChunk


@pytest.fixture
def sample_chunks():
    return [
        Chunk(
            id="chunk-1",
            text="The Tesla Model 3 has excellent range and performance.",
            metadata={"title": "Tesla Review"},
        ),
        Chunk(
            id="chunk-2",
            text="Error code TS-999 indicates a battery fault in the system.",
            metadata={"title": "Error Manual"},
        ),
        Chunk(
            id="chunk-3",
            text="Electric vehicles are becoming more popular worldwide.",
            metadata={"title": "EV Trends"},
        ),
        Chunk(
            id="chunk-4",
            text="The Model 3 acceleration is impressive for its price point.",
            metadata={"title": "Car Review"},
        ),
    ]


class TestTokenize:
    def test_basic_tokenization(self):
        result = _tokenize("Hello World! This is a TEST.")
        assert result == ["hello", "world", "this", "is", "a", "test"]

    def test_numbers_preserved(self):
        result = _tokenize("Error code TS-999")
        assert "999" in result
        assert "ts" in result

    def test_empty_string(self):
        assert _tokenize("") == []

    def test_special_chars_removed(self):
        result = _tokenize("test@email.com #hashtag $100")
        assert "@" not in "".join(result)
        assert "#" not in "".join(result)


class TestBM25Store:
    def test_init(self):
        store = BM25Store(run_id="test")
        assert store._chunks == []
        assert store._bm25 is None

    def test_upsert_and_query(self, sample_chunks):
        store = BM25Store(run_id="test")
        store.upsert(sample_chunks)

        assert len(store._chunks) == 4
        assert store._bm25 is not None

        # Query for Tesla - should find Tesla-related chunks
        results = store.query("Tesla Model 3", top_k=2)
        assert len(results) >= 1
        # At least one result should mention Tesla or Model 3
        texts = [r.chunk.text for r in results]
        assert any("Tesla" in t or "Model 3" in t for t in texts)

    def test_query_exact_match(self, sample_chunks):
        """BM25 should excel at exact keyword matches."""
        store = BM25Store(run_id="test")
        store.upsert(sample_chunks)

        # Query for specific error code
        results = store.query("TS-999 error code", top_k=2)
        assert len(results) >= 1
        # Should find the error manual chunk
        assert any("TS-999" in r.chunk.text for r in results)

    def test_query_empty_store(self):
        store = BM25Store()
        results = store.query("test query")
        assert results == []

    def test_clear(self, sample_chunks):
        store = BM25Store()
        store.upsert(sample_chunks)
        assert len(store._chunks) == 4

        store.clear()
        assert len(store._chunks) == 0
        assert store._bm25 is None

    def test_upsert_no_duplicates(self, sample_chunks):
        store = BM25Store()
        store.upsert(sample_chunks)
        store.upsert(sample_chunks)  # Upsert same chunks again

        # Should not have duplicates
        assert len(store._chunks) == 4


class TestReciprocalRankFusion:
    def test_basic_fusion(self):
        chunk1 = Chunk(id="1", text="First")
        chunk2 = Chunk(id="2", text="Second")
        chunk3 = Chunk(id="3", text="Third")

        list1 = [
            ScoredChunk(chunk=chunk1, score=0.9),
            ScoredChunk(chunk=chunk2, score=0.7),
        ]
        list2 = [
            ScoredChunk(chunk=chunk2, score=0.95),
            ScoredChunk(chunk=chunk3, score=0.6),
        ]

        result = _reciprocal_rank_fusion([list1, list2], weights=[0.5, 0.5])

        # chunk2 appears in both lists, should be ranked higher
        assert len(result) == 3
        # All chunks should be present
        result_ids = [r.chunk.id for r in result]
        assert "1" in result_ids
        assert "2" in result_ids
        assert "3" in result_ids

    def test_empty_lists(self):
        result = _reciprocal_rank_fusion([[], []], weights=[0.5, 0.5])
        assert result == []

    def test_single_list(self):
        chunk1 = Chunk(id="1", text="Only")
        list1 = [ScoredChunk(chunk=chunk1, score=0.9)]

        result = _reciprocal_rank_fusion([list1], weights=[1.0])
        assert len(result) == 1
        assert result[0].chunk.id == "1"


class TestHybridVectorStore:
    def test_init_hybrid_enabled(self, monkeypatch):
        monkeypatch.setattr("vector_store.hybrid_store.HYBRID_SEARCH_ENABLED", True)

        store = HybridVectorStore(
            model_name="all-MiniLM-L6-v2",
            run_id="test",
        )
        assert store._bm25_store is not None
        assert store._embedding_store is not None

    def test_init_hybrid_disabled(self, monkeypatch):
        monkeypatch.setattr("vector_store.hybrid_store.HYBRID_SEARCH_ENABLED", False)

        store = HybridVectorStore(
            model_name="all-MiniLM-L6-v2",
            run_id="test",
        )
        assert store._bm25_store is None
        assert store._embedding_store is not None

    def test_upsert_both_stores(self, sample_chunks, monkeypatch):
        monkeypatch.setattr("vector_store.hybrid_store.HYBRID_SEARCH_ENABLED", True)

        store = HybridVectorStore(
            model_name="all-MiniLM-L6-v2",
            run_id="test",
        )
        store.upsert(sample_chunks)

        # Both stores should have chunks
        assert store._bm25_store is not None
        assert len(store._bm25_store._chunks) == 4

    def test_clear_both_stores(self, sample_chunks, monkeypatch):
        monkeypatch.setattr("vector_store.hybrid_store.HYBRID_SEARCH_ENABLED", True)

        store = HybridVectorStore(
            model_name="all-MiniLM-L6-v2",
            run_id="test",
        )
        store.upsert(sample_chunks)
        store.clear()

        assert store._bm25_store is not None
        assert len(store._bm25_store._chunks) == 0

    def test_query_combines_results(self, sample_chunks, monkeypatch):
        monkeypatch.setattr("vector_store.hybrid_store.HYBRID_SEARCH_ENABLED", True)

        store = HybridVectorStore(
            model_name="all-MiniLM-L6-v2",
            run_id="test",
            bm25_weight=0.3,
        )
        store.upsert(sample_chunks)

        results = store.query("Tesla electric vehicle", top_k=3)

        assert len(results) > 0
        # Results should be ScoredChunks
        assert all(isinstance(r, ScoredChunk) for r in results)

    def test_query_fallback_when_disabled(self, sample_chunks, monkeypatch):
        monkeypatch.setattr("vector_store.hybrid_store.HYBRID_SEARCH_ENABLED", False)

        store = HybridVectorStore(
            model_name="all-MiniLM-L6-v2",
            run_id="test",
        )
        store.upsert(sample_chunks)

        # Should work with embedding store only
        results = store.query("Tesla", top_k=2)
        assert len(results) > 0
