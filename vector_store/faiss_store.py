"""FAISS-based in-memory vector store."""

from typing import List

import faiss
import numpy as np

from api.logging_setup import get_logger
from tools.models import load_embedding_model
from vector_store.base import ScoredChunk, VectorStore
from api.state import Chunk


def _normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10
    return vecs / norms


class FaissVectorStore(VectorStore):
    def __init__(self, model_name: str) -> None:
        logger = get_logger(__name__)
        logger.info("faiss_store_init", extra={"model_name": model_name})
        self.embedder = load_embedding_model(model_name=model_name)
        sample = self.embedder.encode(
            ["sample"], convert_to_numpy=True, normalize_embeddings=True
        )
        dim = sample.shape[1] if len(sample.shape) > 1 else sample.shape[0]
        self.index = faiss.IndexFlatIP(dim)
        self.chunks: List[Chunk] = []
        self.embeddings = np.empty((0, dim), dtype="float32")
        logger.debug("faiss_store_ready", extra={"embedding_dim": dim})

    def clear(self) -> None:
        logger = get_logger(__name__)
        logger.info("faiss_store_clear", extra={"previous_chunks": len(self.chunks)})
        self.index.reset()
        self.chunks = []
        self.embeddings = np.empty(
            (0, self.embeddings.shape[1] if self.embeddings.size else 0),
            dtype="float32",
        )

    def upsert(self, chunks: List[Chunk]) -> None:
        logger = get_logger(__name__)
        if not chunks:
            logger.debug("faiss_upsert_empty")
            return
        logger.info("faiss_upsert_start", extra={"chunk_count": len(chunks)})
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
        logger.info(
            "faiss_upsert_complete",
            extra={"chunk_count": len(chunks), "total_chunks": len(self.chunks)},
        )

    def query(self, text: str, top_k: int = 5) -> List[ScoredChunk]:
        logger = get_logger(__name__)
        logger.debug(
            "faiss_query_start", extra={"query_length": len(text), "top_k": top_k}
        )
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
        logger.info(
            "faiss_query_complete",
            extra={
                "results": len(scored),
                "top_score": scored[0].score if scored else 0,
            },
        )
        return scored
