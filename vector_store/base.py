"""Abstractions for vector storage backends.

This module defines the abstract interface for vector storage backends.
The system uses sentence-transformers for embeddings.

Design Note:
    The VectorStore abstract base class is implemented by ChromaDB (persistent)
    and FAISS (in-memory) backends. All stores should use load_embedding_model()
    from tools/models.py and preserve Chunk.metadata for citation info.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from api.state import Chunk


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float


class VectorStore(ABC):
    """Abstract interface for vector storage backends.

    All vector store implementations must implement upsert() and query() methods.
    The clear() method is optional but recommended for test isolation.
    """

    @abstractmethod
    def upsert(self, chunks: List[Chunk]) -> None:  # pragma: no cover - interface
        """Insert or update chunks in the store.

        Args:
            chunks: List of Chunk objects to store.
        """
        raise NotImplementedError

    @abstractmethod
    def query(
        self, text: str, top_k: int = 5, *, run_id: Optional[str] = None
    ) -> List[ScoredChunk]:  # pragma: no cover - interface
        """Return top_k most similar chunks with scores.

        Args:
            text: Query text to search for.
            top_k: Number of results to return.
            run_id: Optional run correlation ID for logging.

        Returns:
            List of ScoredChunk objects sorted by similarity (highest first).
        """
        raise NotImplementedError

    def clear(self) -> None:
        """Optional: Clear all stored chunks."""
        return None
