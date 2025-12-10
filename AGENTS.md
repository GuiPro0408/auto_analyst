# AGENTS.md

Instructions for AI coding agents working on the Auto-Analyst codebase.

## Project Overview

Auto-Analyst is an autonomous research assistant powered by a LangGraph RAG pipeline. It plans queries, searches free sources, chunks and embeds content, retrieves context, generates cited answers, and verifies claims—all using free/open-source components.

## Tech Stack

- **Language:** Python 3.10+
- **Orchestration:** LangGraph (stateful graph-based pipeline)
- **LLM Backend:** Gemini 2.0 Flash (default), HuggingFace Inference API
- **Embeddings:** sentence-transformers (`all-MiniLM-L6-v2`)
- **Vector Store:** ChromaDB (default), FAISS (alternative)
- **Search:** DuckDuckGo, Wikipedia API, SearxNG, Gemini Grounding
- **UI:** Streamlit
- **Testing:** pytest
- **Parsing:** BeautifulSoup, pdfplumber, Playwright

## Project Structure

```
api/           # Orchestration, state management, config, logging
  graph.py     # LangGraph pipeline nodes and edges
  state.py     # TypedDict state definitions (Document, Chunk, SearchResult, etc.)
  config.py    # Central configuration (env vars, model selection)
  logging_setup.py  # Centralized logging with run correlation IDs

tools/         # Functional pipeline components
  planner.py   # Query decomposition into search tasks
  search.py    # Multi-backend web search
  fetcher.py   # URL fetching with robots.txt compliance
  parser.py    # HTML/PDF content extraction
  chunker.py   # Token-aware text splitting
  generator.py # LLM answer generation with citations
  reranker.py  # Result re-ranking
  retriever.py # Vector similarity search

vector_store/  # Storage abstractions
  base.py      # VectorStore abstract base class
  chroma_store.py  # ChromaDB implementation
  faiss_store.py   # FAISS implementation

evaluation/    # RAG evaluation metrics
  metrics.py   # Context relevance, answer correctness, hallucination detection

ui/            # Streamlit web interface
  app.py       # Main application entry point

tests/         # pytest test suite
```

## Commands

```bash
# Install dependencies
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
playwright install chromium

# Run the application
streamlit run ui/app.py

# Run all tests
pytest

# Run specific test file
pytest tests/test_end_to_end.py

# Run tests with coverage
pytest --cov=api --cov=tools --cov=vector_store

# Run tests matching a pattern
pytest -k "planner"

# Verbose test output
pytest -v

# Type checking (single file)
python -m py_compile path/to/file.py

# Format code
black path/to/file.py

# Lint code
ruff check path/to/file.py --fix
```

## Code Style

### Do

- Use type hints on all function signatures
- Use `TypedDict` for state objects (see `api/state.py`)
- Use `get_logger(__name__, run_id=state.get("run_id"))` for logging
- Use structured extras in log calls: `log.info("event", extra={"key": value})`
- Use `log.exception()` for errors (auto-captures stack trace)
- Keep modules small, composable, and testable
- Use dependency injection (pass models/stores into orchestrators)
- Prefer free/open-source dependencies only
- Return errors in state rather than raising exceptions in pipeline nodes

### Don't

- Don't use `logging.getLogger()` directly—use `get_logger()` from `api/logging_setup`
- Don't leak UI concerns into core modules (`api/`, `tools/`, `vector_store/`)
- Don't hardcode API keys or secrets
- Don't add heavy proprietary dependencies without approval
- Don't bypass the centralized logging configuration

### Code Example: Pipeline Node Pattern

```python
from time import perf_counter
from api.logging_setup import get_logger

def my_node(state: dict):
    log = get_logger("api.graph.my_node", run_id=state.get("run_id"))
    start = perf_counter()

    try:
        result = do_work()
        log.info("my_node_complete", extra={
            "result_count": len(result),
            "duration_ms": (perf_counter() - start) * 1000
        })
        return {"output": result}
    except Exception as exc:
        log.exception("my_node_failed")
        errors = state.get("errors", [])
        errors.append(f"my_node_failed: {exc}")
        return {"output": [], "errors": errors}
```

### Code Example: State Types

```python
from api.state import Document, SearchResult, Chunk, SearchQuery
from vector_store.base import ScoredChunk, VectorStore
```

## Testing Instructions

### Required Mocking Patterns

All external I/O must be mocked in tests. Use these patterns:

```python
# FakeLLM - return HuggingFace pipeline format
class FakeLLM:
    def __call__(self, prompt):
        return [{"generated_text": "Answer: test [1]"}]

# FakeVectorStore - implement VectorStore interface
class FakeVectorStore(VectorStore):
    def __init__(self):
        self.chunks = []
    def upsert(self, chunks):
        self.chunks.extend(chunks)
    def query(self, text, top_k=5):
        return [ScoredChunk(chunk=c, score=1.0) for c in self.chunks[:top_k]]

# Monkeypatch I/O functions
monkeypatch.setattr("api.graph.run_search_tasks", fake_search)
monkeypatch.setattr("api.graph.fetch_url", fake_fetch)
```

### Test Assertions

- Assert on `result.plan`, `result.search_results`, `result.documents`, `result.retrieved`, `result.citations`
- Check `result.verified_answer` for final output
- Use `assert result.errors == []` to catch silent failures

## Environment Variables

| Variable                       | Default            | Description                             |
| ------------------------------ | ------------------ | --------------------------------------- |
| `AUTO_ANALYST_LLM`             | `gemini-2.0-flash` | LLM model identifier                    |
| `AUTO_ANALYST_LLM_BACKEND`     | `gemini`           | LLM backend (gemini, huggingface)       |
| `AUTO_ANALYST_EMBED`           | `all-MiniLM-L6-v2` | Embedding model                         |
| `AUTO_ANALYST_VECTOR_STORE`    | `chroma`           | Vector store backend                    |
| `AUTO_ANALYST_SEARCH_BACKENDS` | `gemini_grounding` | Comma-separated search backends         |
| `AUTO_ANALYST_LOG_LEVEL`       | `DEBUG`            | Log level (DEBUG, INFO, WARNING, ERROR) |
| `AUTO_ANALYST_LOG_FORMAT`      | `plain`            | Log format (plain, json)                |
| `AUTO_ANALYST_LOG_FILE`        | `auto_analyst.log` | Log file path                           |
| `GOOGLE_API_KEY`               | -                  | Gemini API key                          |
| `GOOGLE_API_KEYS`              | -                  | Comma-separated API keys for rotation   |

## Architecture Boundaries

| Directory       | Responsibility               | May Import From                 |
| --------------- | ---------------------------- | ------------------------------- |
| `api/`          | Orchestration, state, config | `tools/`, `vector_store/`       |
| `tools/`        | Pure functional units        | `api/state.py`, `api/config.py` |
| `vector_store/` | Storage abstractions         | `api/state.py`                  |
| `ui/`           | Streamlit interface          | All modules                     |
| `evaluation/`   | RAG metrics                  | `api/state.py`                  |

## Safety and Permissions

### Allowed without prompt

- Read files, list directories
- Run tests (`pytest`)
- Type check single files
- Format and lint code

### Ask first

- Adding new dependencies to `requirements.txt`
- Modifying configuration in `api/config.py`
- Changes to the LangGraph pipeline structure in `api/graph.py`
- Database schema changes

### Never do

- Commit API keys or secrets
- Remove failing tests without fixing the underlying issue
- Modify `data/chromadb/` directly
- Add proprietary/paid dependencies

## PR Checklist

- [ ] All tests pass: `pytest`
- [ ] Code follows existing patterns and style
- [ ] Logging uses `get_logger()` with run correlation
- [ ] External I/O is mocked in new tests
- [ ] No secrets or API keys in code
- [ ] Small, focused diff with clear commit message

## When Stuck

- Ask a clarifying question before making speculative changes
- Propose a short plan for complex refactors
- Reference existing patterns in `tests/test_end_to_end.py` for mocking examples
- Check `.github/instructions/*.md` for domain-specific guidelines
