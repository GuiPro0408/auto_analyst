---
applyTo: "tests/**"
---

# Testing Guidelines

## Mocking External Dependencies

All external I/O must be mocked. Use these established patterns from `tests/test_end_to_end.py`:

### FakeLLM Pattern

```python
class FakeLLM:
    def __call__(self, prompt):
        if "Verified answer:" in prompt:
            return [{"generated_text": "Verified answer: validated [1]"}]
        return [{"generated_text": "Answer: drafted [1]"}]
```

Key: Return `[{"generated_text": ...}]` format matching HuggingFace pipeline.

### FakeVectorStore Pattern

```python
from vector_store.base import ScoredChunk, VectorStore

class FakeVectorStore(VectorStore):
    def __init__(self):
        self.chunks = []

    def clear(self):
        self.chunks = []

    def upsert(self, chunks):
        self.chunks.extend(chunks)

    def query(self, text, top_k: int = 5):
        return [ScoredChunk(chunk=c, score=1.0) for c in self.chunks[:top_k]]
```

### Monkeypatching I/O Functions

```python
def test_something(monkeypatch):
    def fake_search(tasks, max_results=5, run_id=None):
        return [SearchResult(url="http://example.com", title="Test", snippet="...", source="example")]

    def fake_fetch(result: SearchResult):
        return Document(url=result.url, title=result.title, content="test content", media_type="html")

    monkeypatch.setattr("api.graph.run_search_tasks", fake_search)
    monkeypatch.setattr("api.graph.fetch_url", fake_fetch)
```

## Test File Organization

| Test File                    | Purpose                                  |
| ---------------------------- | ---------------------------------------- |
| `test_end_to_end.py`         | Full pipeline integration with all mocks |
| `test_planner.py`            | Planner logic and heuristic fallback     |
| `test_generator.py`          | Answer generation and verification       |
| `test_chunker.py`            | Token-aware chunking                     |
| `test_vector_store.py`       | VectorStore implementations              |
| `test_parser.py`             | HTML/PDF parsing                         |
| `test_evaluation_metrics.py` | RAG metrics calculations                 |

## Required Imports

```python
from api.state import Document, SearchResult, Chunk, SearchQuery
from vector_store.base import ScoredChunk, VectorStore
```

## Assertions

- Always assert on `result.plan`, `result.search_results`, `result.documents`, `result.retrieved`, `result.citations`
- Check `result.verified_answer` for final output
- Use `assert result.errors == []` to catch silent failures

## Running Tests

```bash
pytest                           # All tests
pytest tests/test_end_to_end.py  # Integration only
pytest -k "planner"              # Filter by name
pytest -v                        # Verbose output
```
