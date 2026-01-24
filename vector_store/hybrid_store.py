"""Hybrid vector store combining BM25 lexical search with embedding-based semantic search.

Uses Reciprocal Rank Fusion (RRF) to combine results from both retrieval methods.
Based on research showing hybrid search outperforms either method alone:
- Embeddings excel at semantic similarity
- BM25 excels at exact keyword matches (error codes, technical terms, names)
"""

from pathlib import Path
from typing import Dict, List, Optional

from api.config import BM25_WEIGHT, DATA_DIR, HYBRID_SEARCH_ENABLED
from api.logging_setup import get_logger
from api.state import Chunk
from vector_store.base import ScoredChunk, VectorStore
from vector_store.bm25_store import BM25Store
from vector_store.chroma_store import ChromaVectorStore

# RRF constant (standard value from literature)
RRF_K = 60
# Maximum number of results to fetch from each store for fusion
MAX_FETCH_K = 50


def _reciprocal_rank_fusion(
    ranked_lists: List[List[ScoredChunk]],
    weights: List[float],
    k: int = RRF_K,
) -> List[ScoredChunk]:
    """Combine multiple ranked lists using Reciprocal Rank Fusion.

    RRF score = sum(weight_i / (k + rank_i)) for each list i

    Args:
        ranked_lists: List of ranked result lists.
        weights: Weight for each list (should sum to 1.0).
        k: RRF constant (default 60).

    Returns:
        Combined and re-ranked list of ScoredChunks.
    """
    # Collect RRF scores by chunk ID
    rrf_scores: Dict[str, float] = {}
    chunk_map: Dict[str, Chunk] = {}

    for results, weight in zip(ranked_lists, weights):
        for rank, scored in enumerate(results, start=1):
            chunk_id = scored.chunk.id
            rrf_score = weight / (k + rank)

            if chunk_id in rrf_scores:
                rrf_scores[chunk_id] += rrf_score
            else:
                rrf_scores[chunk_id] = rrf_score
                chunk_map[chunk_id] = scored.chunk

    # Sort by RRF score descending
    sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

    # Normalize scores to [0, 1]
    max_score = max(rrf_scores.values()) if rrf_scores else 1.0
    if max_score <= 0:
        max_score = 1.0

    return [
        ScoredChunk(
            chunk=chunk_map[chunk_id],
            score=rrf_scores[chunk_id] / max_score,
        )
        for chunk_id in sorted_ids
    ]


class HybridVectorStore(VectorStore):
    """Hybrid store combining ChromaDB embeddings with BM25 lexical search.

    Uses Reciprocal Rank Fusion to combine results from both methods.
    Can be disabled via HYBRID_SEARCH_ENABLED config (falls back to ChromaDB only).
    """

    def __init__(
        self,
        model_name: str,
        persist_path: Optional[Path] = None,
        collection_name: str = "auto-analyst",
        run_id: Optional[str] = None,
        bm25_weight: float = BM25_WEIGHT,
    ) -> None:
        self._logger = get_logger(__name__, run_id=run_id)
        self._run_id = run_id
        self._bm25_weight = bm25_weight
        self._embedding_weight = 1.0 - bm25_weight

        # Initialize embedding store
        self._embedding_store = ChromaVectorStore(
            model_name=model_name,
            persist_path=persist_path,
            collection_name=collection_name,
            run_id=run_id,
        )

        # Initialize BM25 store (only if hybrid search enabled)
        self._bm25_store: Optional[BM25Store] = None
        if HYBRID_SEARCH_ENABLED:
            self._bm25_store = BM25Store(run_id=run_id)
            self._logger.info(
                "hybrid_store_init",
                extra={
                    "model_name": model_name,
                    "bm25_weight": bm25_weight,
                    "embedding_weight": self._embedding_weight,
                },
            )
        else:
            self._logger.info(
                "hybrid_store_init_embedding_only",
                extra={"model_name": model_name},
            )

    def clear(self) -> None:
        """Clear both embedding and BM25 stores."""
        self._logger.info("hybrid_store_clear")
        self._embedding_store.clear()
        if self._bm25_store:
            self._bm25_store.clear()

    def upsert(self, chunks: List[Chunk]) -> None:
        """Upsert chunks to both stores."""
        if not chunks:
            self._logger.debug("hybrid_upsert_empty")
            return

        self._logger.info("hybrid_upsert_start", extra={"chunk_count": len(chunks)})

        # Upsert to embedding store
        self._embedding_store.upsert(chunks)

        # Upsert to BM25 store (if enabled)
        if self._bm25_store:
            self._bm25_store.upsert(chunks)

        self._logger.info("hybrid_upsert_complete", extra={"chunk_count": len(chunks)})

    def query(
        self, text: str, top_k: int = 5, *, run_id: Optional[str] = None
    ) -> List[ScoredChunk]:
        """Query both stores and combine results using RRF.

        Args:
            text: Query text.
            top_k: Number of results to return.
            run_id: Optional run ID for logging.

        Returns:
            Combined and re-ranked list of ScoredChunks.
        """
        logger = get_logger(__name__, run_id=run_id or self._run_id)

        # If BM25 disabled, just use embedding store
        if not self._bm25_store or not HYBRID_SEARCH_ENABLED:
            return self._embedding_store.query(text, top_k=top_k, run_id=run_id)

        logger.debug(
            "hybrid_query_start",
            extra={"query_length": len(text), "top_k": top_k},
        )

        # Query both stores (fetch more than top_k for better fusion)
        fetch_k = min(top_k * 3, MAX_FETCH_K)

        embedding_results = self._embedding_store.query(
            text, top_k=fetch_k, run_id=run_id
        )
        bm25_results = self._bm25_store.query(text, top_k=fetch_k, run_id=run_id)

        logger.debug(
            "hybrid_query_individual_results",
            extra={
                "embedding_results": len(embedding_results),
                "bm25_results": len(bm25_results),
            },
        )

        # If one method returns nothing, fall back to the other
        if not embedding_results and not bm25_results:
            logger.warning("hybrid_query_no_results")
            return []
        if not bm25_results:
            return embedding_results[:top_k]
        if not embedding_results:
            return bm25_results[:top_k]

        # Combine using RRF
        combined = _reciprocal_rank_fusion(
            ranked_lists=[embedding_results, bm25_results],
            weights=[self._embedding_weight, self._bm25_weight],
        )

        # Return top_k results
        results = combined[:top_k]

        logger.info(
            "hybrid_query_complete",
            extra={
                "results": len(results),
                "top_score": results[0].score if results else 0,
            },
        )

        return results
