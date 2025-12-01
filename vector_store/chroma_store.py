"""ChromaDB-backed vector store."""

from pathlib import Path
from typing import List, Optional

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from api.config import DATA_DIR
from api.logging_setup import get_logger
from api.state import Chunk
from vector_store.base import ScoredChunk, VectorStore


class ChromaVectorStore(VectorStore):
    def __init__(
        self,
        model_name: str,
        persist_path: Optional[Path] = None,
        collection_name: str = "auto-analyst",
    ) -> None:
        logger = get_logger(__name__)
        persist_path = persist_path or (DATA_DIR / "chromadb")
        persist_path.mkdir(parents=True, exist_ok=True)
        logger.info(
            "chroma_store_init",
            extra={
                "persist_path": str(persist_path),
                "model_name": model_name,
                "collection_name": collection_name,
            },
        )

        settings = Settings(is_persistent=True, persist_directory=str(persist_path))
        self.client = chromadb.Client(settings)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name
        )
        self.collection_name = collection_name
        self._create_collection()
        logger.debug("chroma_store_ready")

    def _create_collection(self):
        logger = get_logger(__name__)
        try:
            self.client.delete_collection(self.collection_name)
            logger.debug(
                "chroma_collection_deleted", extra={"collection": self.collection_name}
            )
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
        logger.debug(
            "chroma_collection_created", extra={"collection": self.collection_name}
        )

    def clear(self) -> None:
        logger = get_logger(__name__)
        logger.info("chroma_store_clear")
        self._create_collection()

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

    def query(self, text: str, top_k: int = 5) -> List[ScoredChunk]:
        logger = get_logger(__name__)
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
        for idx, document in enumerate(results.get("documents", [[]])[0]):
            metadatas = results.get("metadatas", [[]])
            distances = results.get("distances", [[]])
            metadata = metadatas[0][idx] if metadatas and metadatas[0] else {}
            distance = distances[0][idx] if distances and distances[0] else 0.0
            score = 1.0 - distance
            chunk_id = metadata.get("chunk_id") or f"chroma-{idx}"
            chunk = Chunk(id=str(chunk_id), text=document, metadata=metadata or {})
            scored.append(ScoredChunk(chunk=chunk, score=score))
        logger.info(
            "chroma_query_complete",
            extra={
                "results": len(scored),
                "top_score": scored[0].score if scored else 0,
            },
        )
        return scored
