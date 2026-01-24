"""Split long documents into overlapping token-aware chunks."""

import uuid
from typing import Any, Dict, List, Optional

try:
    import tiktoken
except (ImportError, ModuleNotFoundError):  # pragma: no cover
    tiktoken = None

from api.config import CHUNK_OVERLAP, CHUNK_SIZE, CONTEXTUAL_CHUNKS_ENABLED
from api.logging_setup import get_logger
from api.state import Chunk, Document


def _get_encoding(model: str = "gpt2"):
    if not tiktoken:
        return None
    try:
        return tiktoken.get_encoding(model)
    except (KeyError, ValueError):
        return tiktoken.get_encoding("gpt2")


def _count_tokens(text: str, encoding=None) -> int:
    if encoding:
        return len(encoding.encode(text))
    return len(text.split())


class TextChunker:
    """Split documents into overlapping chunks for embedding."""

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        overlap: int = CHUNK_OVERLAP,
        encoding_model: str = "gpt2",
    ) -> None:
        """Initialize the chunker.

        Args:
            chunk_size: Target size of each chunk in tokens.
            overlap: Number of tokens to overlap between chunks.
            encoding_model: Tokenizer model name.

        Raises:
            ValueError: If chunk_size <= overlap (would cause infinite loop).
        """
        if chunk_size <= overlap:
            raise ValueError(
                f"chunk_size ({chunk_size}) must be greater than overlap ({overlap})"
            )

        self.chunk_size = chunk_size
        self.overlap = overlap
        self.encoding = _get_encoding(encoding_model)
        logger = get_logger(__name__)
        logger.debug(
            "chunker_initialized",
            extra={
                "chunk_size": chunk_size,
                "overlap": overlap,
                "encoding_model": encoding_model,
                "has_tiktoken": self.encoding is not None,
            },
        )

    def _split(self, text: str) -> List[str]:
        """Split text into chunks based on token count with overlap."""
        logger = get_logger(__name__)
        if not text:
            logger.debug("chunker_split_empty_text")
            return []
        words = text.split()
        if not self.encoding:
            # Approximate chunking using words if encoding unavailable.
            logger.debug(
                "chunker_using_word_split",
                extra={"word_count": len(words)},
            )
            step = max(self.chunk_size - self.overlap, 1)
            return [
                " ".join(words[i : i + self.chunk_size])
                for i in range(0, len(words), step)
            ]

        tokens = self.encoding.encode(text)
        logger.debug(
            "chunker_tokenized",
            extra={"token_count": len(tokens), "chunk_size": self.chunk_size},
        )
        chunks = []
        start = 0
        step = max(self.chunk_size - self.overlap, 1)
        while start < len(tokens):
            end = start + self.chunk_size
            chunk_tokens = tokens[start:end]
            chunk_text = self.encoding.decode(chunk_tokens)
            chunks.append(chunk_text)
            start += step
        logger.debug(
            "chunker_split_complete",
            extra={"input_tokens": len(tokens), "output_chunks": len(chunks)},
        )
        return chunks

    def chunk_document(
        self, doc: Document, llm=None, run_id: Optional[str] = None
    ) -> List[Chunk]:
        """Chunk a Document and attach metadata.

        Args:
            doc: The document to chunk.
            llm: Optional LLM for contextual chunking.
            run_id: Run correlation ID for logging.

        Returns:
            List of chunks with metadata.
        """
        logger = get_logger(__name__)
        logger.debug(
            "chunk_document_start",
            extra={
                "url": doc.url,
                "content_length": len(doc.content),
                "title": doc.title[:50] if doc.title else "no_title",
            },
        )
        raw_chunks = self._split(doc.content)
        chunks: List[Chunk] = []
        for idx, text in enumerate(raw_chunks):
            metadata: Dict[str, Any] = {
                "url": doc.url,
                "title": doc.title,
                "media_type": doc.media_type,
                "chunk_index": idx,
            }
            chunk_id = f"{uuid.uuid4()}"
            metadata["chunk_id"] = chunk_id
            chunks.append(Chunk(id=chunk_id, text=text, metadata=metadata))

        # Apply contextual chunking if enabled
        if CONTEXTUAL_CHUNKS_ENABLED and chunks:
            try:
                from tools.contextual_chunker import contextualize_chunks

                chunks = contextualize_chunks(doc, chunks, llm=llm, run_id=run_id)
                logger.debug(
                    "chunk_document_contextualized",
                    extra={"url": doc.url, "chunks": len(chunks)},
                )
            except Exception as exc:
                logger.warning(
                    "chunk_document_contextualization_failed",
                    extra={"url": doc.url, "error": str(exc)[:200]},
                )

        logger.debug(
            "chunk_document_complete",
            extra={"url": doc.url, "chunks_created": len(chunks)},
        )
        return chunks
