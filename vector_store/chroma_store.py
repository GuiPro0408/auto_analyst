"""ChromaDB-backed vector store."""

from pathlib import Path
from typing import List, Optional

import chromadb
from chromadb.config import Settings

from api.config import DATA_DIR
from api.state import Chunk
from tools.models import load_embedding_model
from vector_store.base import ScoredChunk, VectorStore


class SentenceTransformerEmbeddingFunction:
    """Adapter so ChromaDB can call a sentence-transformers model."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.model = load_embedding_model(model_name=model_name)

    def __call__(self, input: List[str]) -> List[List[float]]:  # noqa: A002 - match Chroma interface
        return self.model.encode(input, show_progress_bar=False).tolist()

    # Chroma expects a name to detect embedding function changes across restarts
    def name(self) -> str:  # pragma: no cover - trivial passthrough
        return self.model_name

    # Chroma legacy API compatibility flag
    def is_legacy(self) -> bool:  # pragma: no cover - trivial passthrough
        return False


class ChromaVectorStore(VectorStore):
    def __init__(
        self,
        model_name: str,
        persist_path: Optional[Path] = None,
        collection_name: str = "auto-analyst",
    ) -> None:
        persist_path = persist_path or (DATA_DIR / "chromadb")
        persist_path.mkdir(parents=True, exist_ok=True)

        settings = Settings(is_persistent=True, persist_directory=str(persist_path))
        self.client = chromadb.Client(settings)
        self.embedding_fn = SentenceTransformerEmbeddingFunction(model_name=model_name)
        self.collection_name = collection_name
        self._create_collection()

    def _create_collection(self):
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def clear(self) -> None:
        # Drop and recreate collection to avoid invalid empty "where" filter errors.
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            # If deletion fails, fall back to no-op to avoid crashing pipeline.
            pass
        self._create_collection()

    def upsert(self, chunks: List[Chunk]) -> None:
        if not chunks:
            return
        self.collection.upsert(
            ids=[c.id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[c.metadata for c in chunks],
        )

    def query(self, text: str, top_k: int = 5) -> List[ScoredChunk]:
        results = self.collection.query(
            query_texts=[text],
            n_results=top_k,
            include=["documents", "metadatas", "distances", "ids"],
        )
        scored: List[ScoredChunk] = []
        if not results or not results.get("ids"):
            return scored
        for idx, chunk_id in enumerate(results["ids"][0]):
            metadata = results["metadatas"][0][idx]
            document = results["documents"][0][idx]
            distance = results["distances"][0][idx] if results.get("distances") else 0.0
            score = 1.0 - distance
            chunk = Chunk(id=chunk_id, text=document, metadata=metadata)
            scored.append(ScoredChunk(chunk=chunk, score=score))
        return scored
