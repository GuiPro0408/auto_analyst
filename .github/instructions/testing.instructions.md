---
applyTo: "**"
---

# Testing Guidelines

## Overview

All tests are in the `tests/` directory using pytest. External I/O must be mocked to ensure fast, deterministic tests.

## Mocking External Dependencies

### FakeLLM Pattern

```python
class FakeLLM:
    def __call__(self, prompt):
        if "Verified answer:" in prompt:
            return [{"generated_text": "Verified answer: validated [1]"}]
        return [{"generated_text": "Answer: drafted [1]"}]
```

Key: Return `[{"generated_text": ...}]` format matching HuggingFace pipeline and Gemini wrapper.

### CapturingLLM Pattern

For testing prompt construction:

```python
class CapturingLLM:
    def __init__(self, response: str):
        self.response = response
        self.last_prompt = ""

    def __call__(self, prompt):
        self.last_prompt = prompt
        return [{"generated_text": self.response}]

# Usage
llm = CapturingLLM("Answer: ok [1]")
generate_answer(llm, "question", [chunk], conversation_context="context")
assert "Prior conversation summary" in llm.last_prompt
```

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
        return (
            [SearchResult(url="http://example.com", title="Test", snippet="...", source="tavily")],
            [],  # warnings
        )

    def fake_fetch(result: SearchResult, run_id=None):
        return (
            Document(url=result.url, title=result.title, content="test content", media_type="html"),
            None,  # warning
        )

    def fake_smart_search(query, max_results=5, run_id=None):
        return fake_search([], max_results, run_id)

    monkeypatch.setattr("api.graph.run_search_tasks", fake_search)
    monkeypatch.setattr("api.graph.fetch_url", fake_fetch)
    monkeypatch.setattr("api.graph.smart_search", fake_smart_search)
```

## Test File Organization

| Test File                    | Purpose                                  |
| ---------------------------- | ---------------------------------------- |
| `test_end_to_end.py`         | Full pipeline integration with all mocks |
| `test_planner.py`            | Planner heuristics and time detection    |
| `test_generator.py`          | Answer generation and verification       |
| `test_chunker.py`            | Token-aware text chunking                |
| `test_vector_store.py`       | VectorStore implementations              |
| `test_parser.py`             | HTML/PDF parsing                         |
| `test_fetcher.py`            | URL fetching and robots.txt              |
| `test_fetcher_robots.py`     | Robots.txt compliance                    |
| `test_evaluation_metrics.py` | RAG metrics calculations                 |
| `test_search_backends.py`    | Search backend implementations           |
| `test_gemini_grounding.py`   | Gemini grounding integration             |
| `test_smart_search.py`       | LLM-powered search pipeline              |
| `test_grounded_fastpath.py`  | Grounded answer fast path                |
| `test_reranker.py`           | Cross-encoder reranking                  |
| `test_memory.py`             | Conversation memory functions            |
| `test_cache.py`              | Query result caching                     |
| `test_models.py`             | LLM loading and wrappers                 |
| `test_refactored_modules.py` | State builder utilities                  |
| `test_chroma_store_isolation.py` | ChromaDB isolation testing          |

## Required Imports

```python
from api.state import Document, SearchResult, Chunk, SearchQuery, ConversationTurn
from vector_store.base import ScoredChunk, VectorStore
```

## End-to-End Test Pattern

```python
def test_end_to_end_pipeline(monkeypatch):
    def fake_search(tasks, max_results=5, run_id=None):
        return (
            [SearchResult(url="http://example.com", title="Example", snippet="...", source="tavily")],
            [],
        )

    def fake_fetch(result: SearchResult, run_id=None):
        return (
            Document(url=result.url, title=result.title, content="context", media_type="html"),
            None,
        )

    def fake_smart_search(query, max_results=5, run_id=None):
        return fake_search([], max_results, run_id)

    monkeypatch.setattr("api.graph.run_search_tasks", fake_search)
    monkeypatch.setattr("api.graph.fetch_url", fake_fetch)
    monkeypatch.setattr("api.graph.smart_search", fake_smart_search)

    fake_llm = FakeLLM()
    fake_store = FakeVectorStore()
    result = run_research(
        "Test question",
        llm=fake_llm,
        vector_store=fake_store,
        embed_model="fake",
        top_k=1,
    )

    # Assertions
    assert result.plan
    assert result.search_results
    assert result.documents
    assert result.retrieved
    assert result.citations
    assert result.verified_answer.startswith("validated")
    assert result.conversation_history
    assert result.conversation_history[-1].query == "Test question"
    assert result.errors == []
```

## Assertions

- Always assert on `result.plan`, `result.search_results`, `result.documents`, `result.retrieved`, `result.citations`
- Check `result.verified_answer` for final output
- Use `assert result.errors == []` to catch silent failures
- Check `result.warnings` for non-fatal issues
- Verify conversation history is updated

## Testing Planner

```python
def test_plan_query_heuristic():
    query = "Impacts of solar energy adoption on the grid?"
    tasks, is_time_sensitive = plan_query(query, max_tasks=3)
    assert tasks
    assert len(tasks) <= 3
    assert all(task.text for task in tasks)
    assert isinstance(is_time_sensitive, bool)

def test_plan_query_uses_conversation_context():
    query = "What about its battery warranty?"
    context = "Turn 1: Q: Tell me about Tesla Model 3\nA: Discussion about Tesla."
    tasks_with_context, _ = plan_query(query, max_tasks=3, conversation_context=context)
    tasks_without_context, _ = plan_query(query, max_tasks=3)
    assert tasks_with_context
```

## Testing Generator

```python
def test_generate_answer_builds_citations():
    chunk = Chunk(
        id="1",
        text="context text",
        metadata={"title": "Doc", "url": "http://example.com"},
    )
    llm = FakeLLM()
    answer, citations = generate_answer(llm, "question", [chunk])
    assert "citation" in answer.lower()
    assert citations and citations[0]["marker"] == "[1]"

def test_verify_answer_pass_through():
    chunk = Chunk(
        id="1",
        text="context text",
        metadata={"title": "Doc", "url": "http://example.com"},
    )
    llm = FakeLLM()
    verified = verify_answer(llm, "Draft", "question", [chunk])
    assert verified
```

## Testing Vector Store

```python
def test_vector_store_upsert_and_query():
    store = ChromaVectorStore(model_name="all-MiniLM-L6-v2", run_id="test")
    chunks = [
        Chunk(id="1", text="First chunk", metadata={}),
        Chunk(id="2", text="Second chunk", metadata={}),
    ]
    store.upsert(chunks)
    results = store.query("First", top_k=1)
    assert len(results) == 1
    assert results[0].chunk.id == "1"
    store.clear()
```

## Running Tests

```bash
pytest                           # All tests
pytest tests/test_end_to_end.py  # Integration only
pytest -k "planner"              # Filter by name
pytest -v                        # Verbose output
pytest --cov=api --cov=tools     # With coverage
```

## Best Practices

1. **Mock all external I/O** — Network calls, file system, APIs
2. **Use FakeLLM/FakeVectorStore** — Follow established patterns
3. **Return tuples where expected** — Search functions return `(results, warnings)`
4. **Test both success and failure paths** — Include error handling tests
5. **Use descriptive test names** — `test_generate_answer_builds_citations`
6. **Keep tests isolated** — No shared state between tests
7. **Check for empty errors** — `assert result.errors == []`
8. **Include run_id in mocks** — Match real function signatures
