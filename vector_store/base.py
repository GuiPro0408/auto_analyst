"""Abstractions for vector storage backends."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

from api.state import Chunk


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float


class VectorStore(ABC):
    @abstractmethod
    def upsert(self, chunks: List[Chunk]) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    @abstractmethod
    def query(
        self, text: str, top_k: int = 5
    ) -> List[ScoredChunk]:  # pragma: no cover - interface
        raise NotImplementedError

    def clear(self) -> None:
        """Optional clear implementation."""
        return None
