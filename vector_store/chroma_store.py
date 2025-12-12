"""ChromaDB-backed vector store."""

from pathlib import Path
from typing import List, Optional, cast

import chromadb
from chromadb.api.types import EmbeddingFunction, Embeddable
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from api.config import DATA_DIR
from api.logging_setup import get_logger
from api.state import Chunk
from vector_store.base import ScoredChunk, VectorStore


class ChromaVectorStore(VectorStore):
    """ChromaDB-backed vector store with persistent storage.

    Uses run-isolated collections when run_id is provided to prevent
    cross-contamination between concurrent runs.
    """

    def __init__(
        self,
        model_name: str,
        persist_path: Optional[Path] = None,
        collection_name: str = "auto-analyst",
        run_id: Optional[str] = None,
    ) -> None:
        logger = get_logger(__name__, run_id=run_id)
        persist_path = persist_path or (DATA_DIR / "chromadb")
        persist_path.mkdir(parents=True, exist_ok=True)
        self.collection_name = (
            f"{collection_name}-{run_id}" if run_id else collection_name
        )
        self.run_id = run_id
        logger.info(
            "chroma_store_init",
            extra={
                "persist_path": str(persist_path),
                "model_name": model_name,
                "collection_name": self.collection_name,
                "run_id": run_id,
            },
        )

        settings = Settings(is_persistent=True, persist_directory=str(persist_path))
        self.client = chromadb.Client(settings)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name
        )
        self.collection = self._get_or_create_collection()
        logger.debug("chroma_store_ready")

    def _get_or_create_collection(self):
        logger = get_logger(__name__, run_id=self.run_id)
        collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=cast(EmbeddingFunction[Embeddable], self.embedding_fn),
            metadata={"hnsw:space": "cosine"},
        )
        logger.debug(
            "chroma_collection_available", extra={"collection": self.collection_name}
        )
        return collection

    def clear(self) -> None:
        logger = get_logger(__name__, run_id=self.run_id)
        logger.info("chroma_store_clear", extra={"collection": self.collection_name})
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            logger.debug(
                "chroma_clear_delete_failed", extra={"collection": self.collection_name}
            )
        self.collection = self._get_or_create_collection()

    def upsert(self, chunks: List[Chunk]) -> None:
        logger = get_logger(__name__)
        if not chunks:
            logger.debug("chroma_upsert_empty")
            return
        logger.info("chroma_upsert_start", extra={"chunk_count": len(chunks)})
        self.collection.upsert(
            ids=[c.id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[c.metadata for c in chunks],
        )
        logger.info("chroma_upsert_complete", extra={"chunk_count": len(chunks)})

    def query(
        self, text: str, top_k: int = 5, *, run_id: Optional[str] = None
    ) -> List[ScoredChunk]:
        logger = get_logger(__name__, run_id=run_id or self.run_id)
        logger.debug(
            "chroma_query_start", extra={"query_length": len(text), "top_k": top_k}
        )
        results = self.collection.query(
            query_texts=[text],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        scored: List[ScoredChunk] = []
        if not results or not results.get("documents"):
            logger.warning("chroma_query_no_results")
            return scored
        documents_list = results.get("documents") or [[]]
        for idx, document in enumerate(documents_list[0]):
            metadatas = results.get("metadatas", [[]])
            distances = results.get("distances", [[]])
            metadata = metadatas[0][idx] if metadatas and metadatas[0] else {}
            distance = distances[0][idx] if distances and distances[0] else 0.0
            score = 1.0 - distance
            chunk_id = metadata.get("chunk_id") or f"chroma-{idx}"
            chunk = Chunk(
                id=str(chunk_id),
                text=document,
                metadata=dict(metadata) if metadata else {},
            )
            scored.append(ScoredChunk(chunk=chunk, score=score))
        logger.info(
            "chroma_query_complete",
            extra={
                "results": len(scored),
                "top_score": scored[0].score if scored else 0,
            },
        )
        return scored
