"""FAISS-based in-memory vector store."""

from typing import List

import faiss
import numpy as np

from tools.models import load_embedding_model
from vector_store.base import ScoredChunk, VectorStore
from api.state import Chunk


def _normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10
    return vecs / norms


class FaissVectorStore(VectorStore):
    def __init__(self, model_name: str) -> None:
        self.embedder = load_embedding_model(model_name=model_name)
        sample = self.embedder.encode(
            ["sample"], convert_to_numpy=True, normalize_embeddings=True
        )
        dim = sample.shape[1] if len(sample.shape) > 1 else sample.shape[0]
        self.index = faiss.IndexFlatIP(dim)
        self.chunks: List[Chunk] = []
        self.embeddings = np.empty((0, dim), dtype="float32")

    def clear(self) -> None:
        self.index.reset()
        self.chunks = []
        self.embeddings = np.empty(
            (0, self.embeddings.shape[1] if self.embeddings.size else 0),
            dtype="float32",
        )

    def upsert(self, chunks: List[Chunk]) -> None:
        if not chunks:
            return
        texts = [c.text for c in chunks]
        new_vectors = self.embedder.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True
        )
        new_vectors = _normalize(new_vectors.astype("float32"))
        if new_vectors.ndim == 1:
            new_vectors = new_vectors.reshape(1, -1)
        self.index.add(new_vectors.astype("float32"))
        self.chunks.extend(chunks)
        if self.embeddings.size == 0:
            self.embeddings = new_vectors.astype("float32")
        else:
            self.embeddings = np.vstack(
                [self.embeddings, new_vectors.astype("float32")]
            )

    def query(self, text: str, top_k: int = 5) -> List[ScoredChunk]:
        query_vec = self.embedder.encode(
            [text], convert_to_numpy=True, normalize_embeddings=True
        )
        query_vec = _normalize(query_vec.astype("float32"))
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)
        scores, indices = self.index.search(query_vec.astype("float32"), top_k)
        scored: List[ScoredChunk] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or idx >= len(self.chunks):
                continue
            scored.append(ScoredChunk(chunk=self.chunks[idx], score=float(score)))
        return scored
