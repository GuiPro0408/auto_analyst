"""Split long documents into overlapping token-aware chunks."""

import math
import uuid
from typing import Any, Dict, List, Optional

try:
    import tiktoken
except Exception:  # pragma: no cover
    tiktoken = None

from api.config import CHUNK_OVERLAP, CHUNK_SIZE
from api.state import Chunk, Document


def _get_encoding(model: str = "gpt2"):
    if not tiktoken:
        return None
    try:
        return tiktoken.get_encoding(model)
    except Exception:
        return tiktoken.get_encoding("gpt2")


def _count_tokens(text: str, encoding=None) -> int:
    if encoding:
        return len(encoding.encode(text))
    return len(text.split())


class TextChunker:
    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        overlap: int = CHUNK_OVERLAP,
        encoding_model: str = "gpt2",
    ) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.encoding = _get_encoding(encoding_model)

    def _split(self, text: str) -> List[str]:
        """Split text into chunks based on token count with overlap."""
        if not text:
            return []
        words = text.split()
        if not self.encoding:
            # Approximate chunking using words if encoding unavailable.
            step = max(self.chunk_size - self.overlap, 1)
            return [
                " ".join(words[i : i + self.chunk_size])
                for i in range(0, len(words), step)
            ]

        tokens = self.encoding.encode(text)
        chunks = []
        start = 0
        step = max(self.chunk_size - self.overlap, 1)
        while start < len(tokens):
            end = start + self.chunk_size
            chunk_tokens = tokens[start:end]
            chunk_text = self.encoding.decode(chunk_tokens)
            chunks.append(chunk_text)
            start += step
        return chunks

    def chunk_document(self, doc: Document) -> List[Chunk]:
        """Chunk a Document and attach metadata."""
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
        return chunks
