---
applyTo: "vector_store/**"
---

# Vector Store Guidelines

## Overview

All vector store backends implement the abstract interface in `vector_store/base.py`. The system uses sentence-transformers for embeddings.

## Abstract Interface

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

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
    def query(self, text: str, top_k: int = 5, *, run_id: Optional[str] = None) -> List[ScoredChunk]:
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
| `BM25Store`         | `bm25_store.py`   | Lexical keyword search (exact match) |
| `HybridVectorStore` | `hybrid_store.py` | Hybrid BM25 + embeddings (RRF fusion)|

### ChromaVectorStore

- Persistent storage via SQLite
- Run-isolated collections (appends `run_id` to collection name)
- Uses ChromaDB's built-in sentence-transformer embedding
- Cosine similarity via `hnsw:space` metadata
- Converts Chroma cosine *distance* to similarity score in [0, 1]

```python
class ChromaVectorStore(VectorStore):
    def __init__(
        self,
        model_name: str,
        persist_path: Optional[Path] = None,
        collection_name: str = "auto-analyst",
        run_id: Optional[str] = None,
    ) -> None:
        # Collection name becomes f"{collection_name}-{run_id}" if run_id provided
```

### FaissVectorStore

- In-memory storage (no persistence)
- Uses FAISS IndexFlatIP for inner product similarity
- Normalizes vectors for cosine similarity behavior
### BM25Store

- In-memory lexical index using `rank_bm25.BM25Okapi`
- Rebuilds BM25 index on upsert (fine for the per-run, small-to-medium corpora)
- Normalizes BM25 scores to [0, 1]

### HybridVectorStore

- Combines `ChromaVectorStore` (semantic) + `BM25Store` (lexical)
- Uses Reciprocal Rank Fusion (RRF) to merge ranked lists
- Enabled by default via `AUTO_ANALYST_HYBRID_SEARCH=true`

```python
class FaissVectorStore(VectorStore):
    def __init__(self, model_name: str, run_id: Optional[str] = None) -> None:
        self.embedder = load_embedding_model(model_name=model_name)
        self.index = faiss.IndexFlatIP(dim)
        self.chunks: List[Chunk] = []
```

## Backend Selection

Controlled via environment variable:

```python
# api/config.py
VECTOR_STORE_BACKEND = os.getenv("AUTO_ANALYST_VECTOR_STORE", "chroma")
HYBRID_SEARCH_ENABLED = os.getenv("AUTO_ANALYST_HYBRID_SEARCH", "true").lower() == "true"
```

Factory function in `tools/retriever.py`:

```python
def build_vector_store(
    model_name: str = DEFAULT_EMBED_MODEL, run_id: Optional[str] = None
) -> VectorStore:
    backend = VECTOR_STORE_BACKEND.lower()
    # Hybrid takes precedence when enabled (except when explicitly using faiss)
    if HYBRID_SEARCH_ENABLED and backend not in ("faiss",):
        return HybridVectorStore(model_name=model_name, run_id=run_id)
    if backend == "faiss":
        return FaissVectorStore(model_name=model_name, run_id=run_id)
    return ChromaVectorStore(model_name=model_name, run_id=run_id)
```

## Implementing a New Backend

1. Create `vector_store/new_store.py`:

```python
from typing import List, Optional
from api.state import Chunk
from api.logging_setup import get_logger
from vector_store.base import ScoredChunk, VectorStore
from tools.models import load_embedding_model

class NewVectorStore(VectorStore):
    def __init__(self, model_name: str, run_id: Optional[str] = None) -> None:
        self.logger = get_logger(__name__, run_id=run_id)
        self.embedder = load_embedding_model(model_name=model_name)
        self._storage = {}  # Your storage mechanism
        self.logger.info("new_store_init", extra={"model_name": model_name})

    def upsert(self, chunks: List[Chunk]) -> None:
        if not chunks:
            self.logger.debug("new_store_upsert_empty")
            return
        self.logger.info("new_store_upsert_start", extra={"chunk_count": len(chunks)})
        texts = [c.text for c in chunks]
        embeddings = self.embedder.encode(texts, convert_to_numpy=True)
        for chunk, emb in zip(chunks, embeddings):
            self._storage[chunk.id] = (chunk, emb)
        self.logger.info("new_store_upsert_complete", extra={"chunk_count": len(chunks)})

    def query(self, text: str, top_k: int = 5, *, run_id: Optional[str] = None) -> List[ScoredChunk]:
        self.logger.debug("new_store_query_start", extra={"query_length": len(text)})
        query_emb = self.embedder.encode([text], convert_to_numpy=True)[0]
        # Compute similarities, sort, return top_k
        scored = [...]
        self.logger.info("new_store_query_complete", extra={"results": len(scored)})
        return scored[:top_k]

    def clear(self) -> None:
        self.logger.info("new_store_clear")
        self._storage = {}
```

2. Register in `tools/retriever.py:build_vector_store()`:

```python
from vector_store.new_store import NewVectorStore

def build_vector_store(model_name: str = DEFAULT_EMBED_MODEL, run_id: Optional[str] = None) -> VectorStore:
    backend = VECTOR_STORE_BACKEND.lower()
    if backend == "new":
        return NewVectorStore(model_name=model_name, run_id=run_id)
    # ... existing backends
```

3. Update `api/config.py` documentation for new option.

## Embedding Model

All stores should use `tools/models.py:load_embedding_model()`:

```python
from tools.models import load_embedding_model

embedder = load_embedding_model(model_name="all-MiniLM-L6-v2")
vectors = embedder.encode(["text1", "text2"], convert_to_numpy=True)
```

Default model: `all-MiniLM-L6-v2` (384 dimensions, fast, good quality)

## Chunk Metadata

`Chunk.metadata` carries citation info — preserve it through storage:

```python
{
    "url": "https://example.com/page",
    "title": "Page Title",
    "media_type": "html",
    "chunk_index": 0,
    "source": "tavily",  # or "gemini_grounding"
}
```

## Run Isolation

ChromaDB collections are isolated per run:

```python
# In ChromaVectorStore.__init__
self.collection_name = f"{collection_name}-{run_id}" if run_id else collection_name
```

This prevents cross-contamination between concurrent runs.

Hybrid and BM25 stores are also run-scoped in practice (they are created per `run_id` in `build_vector_store()`).

## Scoring

- **ChromaDB**: Returns cosine *distances*; converted to similarity in [0, 1] as `1.0 - (distance / 2.0)`
- **FAISS**: Returns inner product scores (vectors are normalized for cosine-like behavior)
- **BM25**: Scores are normalized to [0, 1] by dividing by the max score for the query
- **Hybrid**: RRF scores are normalized to [0, 1]

All backends return `ScoredChunk` with `score` in range [0, 1] where higher is better.

## Environment Variables

| Variable | Default | Description |
| --- | --- | --- |
| `AUTO_ANALYST_VECTOR_STORE` | `chroma` | Vector store backend (`chroma`, `faiss`) |
| `AUTO_ANALYST_HYBRID_SEARCH` | `true` | Enable Hybrid retrieval (Chroma + BM25) when backend supports it |
| `AUTO_ANALYST_BM25_WEIGHT` | `0.3` | Weight of BM25 list in RRF fusion (0.0–1.0) |

## Testing New Backends

Add tests in `tests/test_vector_store.py` following existing patterns:

```python
def test_new_store_upsert_and_query():
    store = NewVectorStore(model_name="all-MiniLM-L6-v2", run_id="test")
    chunks = [
        Chunk(id="1", text="First chunk", metadata={"title": "Doc1"}),
        Chunk(id="2", text="Second chunk", metadata={"title": "Doc2"}),
    ]
    
    # Test upsert
    store.upsert(chunks)
    
    # Test query returns ScoredChunk list
    results = store.query("First", top_k=1)
    assert len(results) == 1
    assert isinstance(results[0], ScoredChunk)
    assert results[0].chunk.id == "1"
    assert 0 <= results[0].score <= 1
    
    # Test clear removes all data
    store.clear()
    results = store.query("First", top_k=1)
    assert len(results) == 0

def test_new_store_metadata_preserved():
    store = NewVectorStore(model_name="all-MiniLM-L6-v2")
    chunk = Chunk(id="1", text="Test", metadata={"url": "http://example.com", "title": "Test"})
    store.upsert([chunk])
    results = store.query("Test", top_k=1)
    assert results[0].chunk.metadata["url"] == "http://example.com"
```

## Best Practices

1. **Log all operations** — Include chunk counts, query times, scores
2. **Handle empty inputs** — Return early for empty chunk lists
3. **Preserve metadata** — Don't lose citation info during storage
4. **Normalize scores** — Ensure 0-1 range, higher is better
5. **Support run isolation** — Accept `run_id` parameter
6. **Use embedding model from tools/models.py** — Don't load separately
