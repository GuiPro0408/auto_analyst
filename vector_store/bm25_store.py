"""BM25-backed lexical search store for keyword matching."""

import re
from typing import List, Optional

from rank_bm25 import BM25Okapi

from api.logging_setup import get_logger
from api.state import Chunk
from vector_store.base import ScoredChunk, VectorStore


def _tokenize(text: str) -> List[str]:
    """Simple tokenization: lowercase, alphanumeric tokens; non-alphanumerics (e.g. hyphens) are stripped, so 'TS-999' -> ['ts', '999']."""
    return re.findall(r"\b[a-z0-9]+\b", text.lower())


class BM25Store(VectorStore):
    """BM25-based lexical search store.

    Uses BM25Okapi algorithm for term frequency-based retrieval.
    Excellent for exact keyword matches that embedding models might miss.
    """

    def __init__(self, run_id: Optional[str] = None) -> None:
        self._logger = get_logger(__name__, run_id=run_id)
        self._chunks: List[Chunk] = []
        self._tokenized_corpus: List[List[str]] = []
        self._bm25: Optional[BM25Okapi] = None
        self._run_id = run_id
        self._logger.info("bm25_store_init")

    def clear(self) -> None:
        """Clear all stored chunks."""
        self._logger.info("bm25_store_clear", extra={"chunk_count": len(self._chunks)})
        self._chunks = []
        self._tokenized_corpus = []
        self._bm25 = None

    def upsert(self, chunks: List[Chunk]) -> None:
        """Add chunks to the BM25 index.

        Note: BM25 requires rebuilding the index on upsert.
        For large datasets, consider batching upserts.
        """
        if not chunks:
            self._logger.debug("bm25_upsert_empty")
            return

        self._logger.info("bm25_upsert_start", extra={"chunk_count": len(chunks)})

        # Add new chunks
        for chunk in chunks:
            # Check for duplicate IDs
            existing_ids = {c.id for c in self._chunks}
            if chunk.id not in existing_ids:
                self._chunks.append(chunk)
                self._tokenized_corpus.append(_tokenize(chunk.text or ""))

        # Rebuild BM25 index
        if self._tokenized_corpus:
            self._bm25 = BM25Okapi(self._tokenized_corpus)
            self._logger.info(
                "bm25_index_built",
                extra={"total_chunks": len(self._chunks)},
            )
        else:
            self._bm25 = None

    def query(
        self, text: str, top_k: int = 5, *, run_id: Optional[str] = None
    ) -> List[ScoredChunk]:
        """Query the BM25 index for relevant chunks.

        Args:
            text: Query text.
            top_k: Number of results to return.
            run_id: Optional run ID for logging.

        Returns:
            List of ScoredChunk ordered by BM25 score (descending).
        """
        logger = get_logger(__name__, run_id=run_id or self._run_id)

        if not self._bm25 or not self._chunks:
            logger.warning("bm25_query_empty_index")
            return []

        logger.debug(
            "bm25_query_start",
            extra={"query_length": len(text), "top_k": top_k},
        )

        tokenized_query = _tokenize(text)
        if not tokenized_query:
            logger.warning("bm25_query_empty_tokens")
            return []

        # Get BM25 scores for all documents
        scores = self._bm25.get_scores(tokenized_query)

        # Pair chunks with scores and sort
        chunk_scores = list(zip(self._chunks, scores))
        chunk_scores.sort(key=lambda x: x[1], reverse=True)

        # Take top_k and normalize scores
        top_results = chunk_scores[:top_k]

        # Normalize scores to [0, 1] range
        max_score = max(scores) if len(scores) > 0 else 1.0
        if max_score <= 0:
            max_score = 1.0

        scored_chunks: List[ScoredChunk] = []
        for chunk, score in top_results:
            if score > 0:  # Only include positive scores
                normalized_score = min(score / max_score, 1.0)
                scored_chunks.append(ScoredChunk(chunk=chunk, score=normalized_score))

        logger.info(
            "bm25_query_complete",
            extra={
                "results": len(scored_chunks),
                "top_score": scored_chunks[0].score if scored_chunks else 0,
            },
        )

        return scored_chunks
