"""Chunking, indexing, and retrieval helpers."""

from typing import List, Optional

from api.config import DEFAULT_EMBED_MODEL, TOP_K_RESULTS, VECTOR_STORE_BACKEND
from api.logging_setup import get_logger
from api.state import Chunk, Document
from tools.chunker import TextChunker
from vector_store.base import ScoredChunk, VectorStore
from vector_store.chroma_store import ChromaVectorStore
from vector_store.faiss_store import FaissVectorStore


def build_vector_store(model_name: str = DEFAULT_EMBED_MODEL) -> VectorStore:
    backend = VECTOR_STORE_BACKEND.lower()
    if backend == "faiss":
        return FaissVectorStore(model_name=model_name)
    return ChromaVectorStore(model_name=model_name)


def chunk_documents(
    documents: List[Document], chunker: TextChunker, run_id: Optional[str] = None
) -> List[Chunk]:
    logger = get_logger(__name__, run_id=run_id)
    chunks: List[Chunk] = []
    for doc in documents:
        chunks.extend(chunker.chunk_document(doc))
    logger.info(
        "chunking_complete",
        extra={
            "documents": len(documents),
            "chunks": len(chunks),
            "chunk_size": chunker.chunk_size,
        },
    )
    return chunks


def index_chunks(store: VectorStore, chunks: List[Chunk]) -> None:
    store.upsert(chunks)


def retrieve_chunks(
    store: VectorStore,
    query: str,
    top_k: int = TOP_K_RESULTS,
    run_id: Optional[str] = None,
) -> List[ScoredChunk]:
    logger = get_logger(__name__, run_id=run_id)
    results = store.query(query, top_k=top_k)
    logger.info(
        "retrieval_complete",
        extra={"query": query, "results": len(results), "top_k": top_k},
    )
    return results
