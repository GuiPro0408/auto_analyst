"""Contextual chunking with LLM-generated context for improved retrieval.

Based on Anthropic's Contextual Retrieval approach:
https://www.anthropic.com/news/contextual-retrieval

Prepends chunk-specific context to each chunk before embedding to preserve
document-level information that would otherwise be lost during chunking.
"""

from time import time
from typing import List, Optional

from api.backend_utils import is_limited_backend
from api.config import (
    CONTEXTUAL_CHUNK_CHAR_LIMIT,
    CONTEXTUAL_CHUNKS_ENABLED,
    CONTEXTUAL_DOCUMENT_CHAR_LIMIT,
    CONTEXTUAL_MAX_CHUNKS_PER_DOC,
)
from api.logging_setup import get_logger
from api.state import Chunk, Document

# Prompt template for generating chunk context
CONTEXT_GENERATION_PROMPT = """<document>
{document_text}
</document>

Here is a chunk from the document that we want to situate within the overall document context:
<chunk>
{chunk_text}
</chunk>

Please provide a brief context (2-3 sentences) to situate this chunk within the overall document. 
Focus on:
- What entity/topic the document is about (company name, person, product, etc.)
- The time period or date if relevant
- The section or aspect of the topic this chunk addresses

Answer only with the succinct context, nothing else."""

# Maximum document length to include in prompt (to avoid token limits)
MAX_DOCUMENT_CHARS = CONTEXTUAL_DOCUMENT_CHAR_LIMIT
# Maximum chunk length for context generation
MAX_CHUNK_CHARS = CONTEXTUAL_CHUNK_CHAR_LIMIT
# Context length validation bounds
MIN_CONTEXT_LENGTH = 10
MAX_CONTEXT_LENGTH = 500
# Circuit breaker settings for repeated failures (rate limits, quota exhaustion)
CONTEXTUAL_MAX_FAILURES = 3
CONTEXTUAL_FAILURE_COOLDOWN_SECONDS = 120

_consecutive_failures = 0
_cooldown_until = 0.0


def generate_chunk_context(
    document: Document,
    chunk: Chunk,
    llm=None,
    run_id: Optional[str] = None,
) -> str:
    """Generate contextual prefix for a chunk using LLM.

    Args:
        document: The source document for context.
        chunk: The chunk to contextualize.
        llm: Optional LLM instance (loads default if not provided).
        run_id: Run correlation ID for logging.

    Returns:
        Contextual prefix string (empty string if generation fails).
    """
    logger = get_logger(__name__, run_id=run_id)

    global _consecutive_failures, _cooldown_until

    if not CONTEXTUAL_CHUNKS_ENABLED:
        logger.debug("contextual_chunking_disabled")
        return ""

    if is_limited_backend():
        logger.info("contextual_chunking_skipped_limited_backend")
        return ""

    now = time()
    if _cooldown_until > now:
        logger.debug(
            "contextual_chunking_cooldown",
            extra={"cooldown_remaining_s": round(_cooldown_until - now, 2)},
        )
        return ""

    if llm is None:
        from tools.models import load_llm

        llm = load_llm()

    # Truncate document if too long
    doc_text = document.content[:MAX_DOCUMENT_CHARS]
    if len(document.content) > MAX_DOCUMENT_CHARS:
        doc_text += "\n... [document truncated for context generation]"

    # Truncate chunk if too long
    chunk_text = chunk.text[:MAX_CHUNK_CHARS]

    prompt = CONTEXT_GENERATION_PROMPT.format(
        document_text=doc_text,
        chunk_text=chunk_text,
    )

    try:
        result = llm(prompt)
        context = result[0]["generated_text"].strip()

        # Basic validation - context should be reasonable length
        if len(context) < MIN_CONTEXT_LENGTH or len(context) > MAX_CONTEXT_LENGTH:
            logger.warning(
                "contextual_chunk_context_invalid_length",
                extra={"context_length": len(context), "chunk_id": chunk.id},
            )
            return ""

        logger.debug(
            "contextual_chunk_context_generated",
            extra={
                "chunk_id": chunk.id,
                "context_length": len(context),
            },
        )
        _consecutive_failures = 0
        _cooldown_until = 0.0
        return context

    except Exception as exc:
        _consecutive_failures += 1
        if _consecutive_failures >= CONTEXTUAL_MAX_FAILURES:
            _cooldown_until = time() + CONTEXTUAL_FAILURE_COOLDOWN_SECONDS
        logger.warning(
            "contextual_chunk_context_failed",
            extra={
                "chunk_id": chunk.id,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "consecutive_failures": _consecutive_failures,
                "cooldown_seconds": (
                    CONTEXTUAL_FAILURE_COOLDOWN_SECONDS if _cooldown_until > 0 else 0
                ),
            },
        )
        return ""


def contextualize_chunks(
    document: Document,
    chunks: List[Chunk],
    llm=None,
    run_id: Optional[str] = None,
) -> List[Chunk]:
    """Add contextual prefixes to all chunks from a document.

    Args:
        document: The source document.
        chunks: List of chunks from this document.
        llm: Optional LLM instance.
        run_id: Run correlation ID for logging.

    Returns:
        New list of chunks with contextualized text.
    """
    logger = get_logger(__name__, run_id=run_id)

    if not CONTEXTUAL_CHUNKS_ENABLED:
        logger.debug("contextual_chunking_disabled_skipping")
        return chunks

    if is_limited_backend():
        logger.info("contextual_chunking_skipped_limited_backend")
        return chunks

    if not chunks:
        return chunks

    logger.info(
        "contextualize_chunks_start",
        extra={
            "document_url": document.url,
            "chunk_count": len(chunks),
        },
    )

    if llm is None:
        from tools.models import load_llm

        llm = load_llm()

    contextualized_chunks: List[Chunk] = []
    success_count = 0

    max_chunks = max(0, CONTEXTUAL_MAX_CHUNKS_PER_DOC)
    target_chunks = chunks[:max_chunks] if max_chunks else []
    skipped_count = max(len(chunks) - len(target_chunks), 0)

    if skipped_count:
        logger.info(
            "contextualize_chunks_limited",
            extra={
                "document_url": document.url,
                "requested_chunks": len(chunks),
                "contextualized_limit": len(target_chunks),
            },
        )

    for chunk in target_chunks:
        context = generate_chunk_context(document, chunk, llm=llm, run_id=run_id)

        if context:
            # Prepend context to chunk text
            contextualized_text = f"{context}\n\n{chunk.text}"
            success_count += 1
        else:
            # Fall back to original text
            contextualized_text = chunk.text

        # Create new chunk with contextualized text
        new_chunk = Chunk(
            id=chunk.id,
            text=contextualized_text,
            metadata={
                **chunk.metadata,
                "has_context": bool(context),
                "original_text_length": len(chunk.text),
            },
        )
        contextualized_chunks.append(new_chunk)

    if skipped_count:
        for chunk in chunks[len(target_chunks) :]:
            contextualized_chunks.append(
                Chunk(
                    id=chunk.id,
                    text=chunk.text,
                    metadata={
                        **chunk.metadata,
                        "has_context": False,
                        "original_text_length": len(chunk.text),
                    },
                )
            )

    logger.info(
        "contextualize_chunks_complete",
        extra={
            "document_url": document.url,
            "total_chunks": len(chunks),
            "contextualized_count": success_count,
            "skipped_count": skipped_count,
        },
    )

    return contextualized_chunks


def contextualize_document_chunks(
    documents: List[Document],
    chunks_by_url: dict,
    llm=None,
    run_id: Optional[str] = None,
) -> List[Chunk]:
    """Contextualize chunks for multiple documents.

    Args:
        documents: List of source documents.
        chunks_by_url: Dict mapping document URL to list of chunks.
        llm: Optional LLM instance.
        run_id: Run correlation ID for logging.

    Returns:
        Flat list of all contextualized chunks.
    """
    logger = get_logger(__name__, run_id=run_id)

    if not CONTEXTUAL_CHUNKS_ENABLED:
        # Return flattened chunks without modification
        all_chunks = []
        for url_chunks in chunks_by_url.values():
            all_chunks.extend(url_chunks)
        return all_chunks

    if llm is None:
        from tools.models import load_llm

        llm = load_llm()

    all_contextualized: List[Chunk] = []
    doc_by_url = {doc.url: doc for doc in documents}

    for url, chunks in chunks_by_url.items():
        doc = doc_by_url.get(url)
        if doc:
            contextualized = contextualize_chunks(doc, chunks, llm=llm, run_id=run_id)
            all_contextualized.extend(contextualized)
        else:
            # No document found, use chunks as-is
            logger.warning(
                "contextualize_document_not_found",
                extra={"url": url, "chunk_count": len(chunks)},
            )
            all_contextualized.extend(chunks)

    logger.info(
        "contextualize_all_documents_complete",
        extra={
            "total_documents": len(documents),
            "total_chunks": len(all_contextualized),
        },
    )

    return all_contextualized
