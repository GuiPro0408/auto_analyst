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
    logger = get_logger(__name__)
    backend = VECTOR_STORE_BACKEND.lower()
    logger.info(
        "build_vector_store",
        extra={"backend": backend, "model_name": model_name},
    )
    if backend == "faiss":
        logger.debug("vector_store_using_faiss")
        return FaissVectorStore(model_name=model_name)
    logger.debug("vector_store_using_chroma")
    return ChromaVectorStore(model_name=model_name)


def chunk_documents(
    documents: List[Document], chunker: TextChunker, run_id: Optional[str] = None
) -> List[Chunk]:
    logger = get_logger(__name__, run_id=run_id)
    logger.info(
        "chunk_documents_start",
        extra={
            "documents": len(documents),
            "chunk_size": chunker.chunk_size,
            "overlap": chunker.overlap,
        },
    )
    chunks: List[Chunk] = []
    for idx, doc in enumerate(documents):
        doc_chunks = chunker.chunk_document(doc)
        logger.debug(
            "document_chunked",
            extra={
                "doc_index": idx,
                "url": doc.url,
                "content_length": len(doc.content),
                "chunks_created": len(doc_chunks),
            },
        )
        chunks.extend(doc_chunks)
    logger.info(
        "chunking_complete",
        extra={
            "documents": len(documents),
            "chunks": len(chunks),
            "chunk_size": chunker.chunk_size,
            "avg_chunks_per_doc": len(chunks) / len(documents) if documents else 0,
        },
    )
    return chunks


def index_chunks(
    store: VectorStore, chunks: List[Chunk], run_id: Optional[str] = None
) -> None:
    logger = get_logger(__name__, run_id=run_id)
    logger.info(
        "index_chunks_start",
        extra={"chunk_count": len(chunks)},
    )
    store.upsert(chunks)
    logger.info(
        "index_chunks_complete",
        extra={"chunk_count": len(chunks)},
    )


def retrieve_chunks(
    store: VectorStore,
    query: str,
    top_k: int = TOP_K_RESULTS,
    run_id: Optional[str] = None,
) -> List[ScoredChunk]:
    logger = get_logger(__name__, run_id=run_id)
    logger.info(
        "retrieve_chunks_start",
        extra={"query": query, "top_k": top_k},
    )
    results = store.query(query, top_k=top_k)
    logger.info(
        "retrieval_complete",
        extra={
            "query": query,
            "results": len(results),
            "top_k": top_k,
            "scores": [r.score for r in results] if results else [],
        },
    )
    return results
