---
applyTo: "vector_store/**"
---

# Vector Store Guidelines

## Abstract Interface

All vector store backends must implement `vector_store/base.py:VectorStore`:

```python
from abc import ABC, abstractmethod
from typing import List
from api.state import Chunk

@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float

class VectorStore(ABC):
    @abstractmethod
    def upsert(self, chunks: List[Chunk]) -> None:
        """Insert or update chunks in the store."""
        raise NotImplementedError

    @abstractmethod
    def query(self, text: str, top_k: int = 5) -> List[ScoredChunk]:
        """Return top_k most similar chunks with scores."""
        raise NotImplementedError

    def clear(self) -> None:
        """Optional: Clear all stored chunks."""
        return None
```

## Existing Implementations

| Backend             | File              | Use Case                             |
| ------------------- | ----------------- | ------------------------------------ |
| `ChromaVectorStore` | `chroma_store.py` | Default, persistent local storage    |
| `FaissVectorStore`  | `faiss_store.py`  | In-memory, faster for small datasets |

## Backend Selection

Controlled via environment variable:

```python
# api/config.py
VECTOR_STORE_BACKEND = os.getenv("AUTO_ANALYST_VECTOR_STORE", "chroma")
```

Factory function in `tools/retriever.py`:

```python
def build_vector_store(model_name: str = DEFAULT_EMBED_MODEL) -> VectorStore:
    backend = VECTOR_STORE_BACKEND.lower()
    if backend == "faiss":
        return FaissVectorStore(model_name=model_name)
    return ChromaVectorStore(model_name=model_name)
```

## Implementing a New Backend

1. Create `vector_store/new_store.py`:

```python
from typing import List
from api.state import Chunk
from vector_store.base import ScoredChunk, VectorStore
from tools.models import load_embedding_model

class NewVectorStore(VectorStore):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.embedder = load_embedding_model(model_name=model_name)
        self._storage = {}  # Your storage mechanism

    def upsert(self, chunks: List[Chunk]) -> None:
        texts = [c.text for c in chunks]
        embeddings = self.embedder.encode(texts)
        for chunk, emb in zip(chunks, embeddings):
            self._storage[chunk.id] = (chunk, emb)

    def query(self, text: str, top_k: int = 5) -> List[ScoredChunk]:
        query_emb = self.embedder.encode([text])[0]
        # Compute similarities, sort, return top_k
        scored = [...]
        return scored[:top_k]

    def clear(self) -> None:
        self._storage = {}
```

2. Register in `tools/retriever.py:build_vector_store()`:

```python
from vector_store.new_store import NewVectorStore

def build_vector_store(model_name: str = DEFAULT_EMBED_MODEL) -> VectorStore:
    backend = VECTOR_STORE_BACKEND.lower()
    if backend == "new":
        return NewVectorStore(model_name=model_name)
    # ... existing backends
```

3. Update `api/config.py` documentation for new option.

## Embedding Model

All stores should use `tools/models.py:load_embedding_model()`:

```python
from tools.models import load_embedding_model

embedder = load_embedding_model(model_name="all-MiniLM-L6-v2")
vectors = embedder.encode(["text1", "text2"])  # Returns numpy array
```

## Chunk Metadata

`Chunk.metadata` carries citation info—preserve it through storage:

```python
{
    "url": "https://example.com/page",
    "title": "Page Title",
    "media_type": "html",
    "chunk_index": 0
}
```

## Testing New Backends

Add tests in `tests/test_vector_store.py` following existing patterns:

- Test `upsert()` stores chunks
- Test `query()` returns `ScoredChunk` list
- Test `clear()` removes all data
- Test metadata preservation
