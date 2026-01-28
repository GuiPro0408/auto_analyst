# AGENTS.md

Instructions for AI coding agents working on the Auto-Analyst codebase.

## Project Overview

Auto-Analyst is an autonomous research assistant powered by a LangGraph RAG pipeline. It plans queries, searches free sources, chunks and embeds content, retrieves context, generates cited answers, and verifies claims—all using free/open-source components.

## Tech Stack

- **Language:** Python 3.11+
- **Orchestration:** LangGraph (stateful graph-based pipeline)
- **LLM Backend:** Gemini 2.0 Flash (default), HuggingFace Inference API (fallback)
- **Embeddings:** sentence-transformers (`all-MiniLM-L6-v2`)
- **Vector Store:** ChromaDB (default), FAISS (alternative)
- **Search:** Tavily, Gemini Grounding (Google Search)
- **UI:** Streamlit
- **Testing:** pytest
- **Parsing:** BeautifulSoup, pdfplumber

## Project Structure

```
api/               # Orchestration, state management, config, logging
  graph.py         # LangGraph pipeline nodes and edges
  state.py         # Dataclasses and TypedDict definitions
  state_builder.py # State construction helpers
  config.py        # Central configuration (env vars, model selection)
  logging_setup.py # Centralized logging with run correlation IDs
  cache_manager.py # Query result caching
  key_rotator.py   # API key rotation for rate limits
  memory.py        # Conversation history management

tools/             # Functional pipeline components (pure functional units)
  planner.py       # Query decomposition into search tasks
  search.py        # Multi-backend web search
  fetcher.py       # URL fetching with robots.txt compliance
  parser.py        # HTML/PDF content extraction
  chunker.py       # Token-aware text splitting
  generator.py     # LLM answer generation with citations
  models.py        # LLM and embedding model loading
  reranker.py      # Result re-ranking
  retriever.py     # Vector similarity search
  gemini_grounding.py # Gemini web-augmented responses
  smart_search.py  # LLM-powered search pipeline
  quality_control.py # Runtime quality assessment

vector_store/      # Storage abstractions
  base.py          # VectorStore abstract base class + ScoredChunk
  chroma_store.py  # ChromaDB implementation (persistent)
  faiss_store.py   # FAISS implementation (in-memory)

evaluation/        # RAG evaluation metrics
  metrics.py       # Context relevance, answer correctness, hallucination detection
  run_evaluation.py # Evaluation runner

ui/                # Streamlit web interface
  app.py           # Main application entry point

tests/             # pytest test suite
```

## Commands

```bash
# Install dependencies
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

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

# Run evaluation
python evaluation/run_evaluation.py --dataset data/sample_eval.json --model all-MiniLM-L6-v2
```

---

## Clean Architecture

### Module Boundaries

| Directory       | Responsibility                                     | May Import From                                           |
| --------------- | -------------------------------------------------- | --------------------------------------------------------- |
| `api/`          | Orchestration, state types, config, logging, cache | `tools/`, `vector_store/`                                 |
| `tools/`        | Pure functional units (search, parse, generate)    | `api/state.py`, `api/config.py`, `api/logging_setup.py`   |
| `vector_store/` | Storage abstractions (VectorStore interface)       | `api/state.py`, `api/logging_setup.py`                    |
| `ui/`           | Streamlit interface                                | All modules                                               |
| `evaluation/`   | RAG metrics                                        | `api/state.py`, `tools/models.py`                         |

### Core Principles

1. **Single Responsibility** — Each module handles one concern
2. **Dependency Injection** — Pass dependencies explicitly rather than creating internally
3. **State Immutability** — Pipeline nodes return only modified keys
4. **Error Handling** — Use `errors` and `warnings` lists for recoverable issues
5. **No UI in Core Modules** — `tools/` and `api/` must never import from `ui/`

### Pipeline Structure

The LangGraph pipeline flows through these nodes:

```
START → plan → search → fetch → retrieve → adaptive → generate → verify → qc → END
```

---

## State Management

### Core State Types

```python
from api.state import Document, SearchResult, Chunk, SearchQuery, ConversationTurn, ResearchState
from api.state import GraphState  # TypedDict for LangGraph
from vector_store.base import ScoredChunk, VectorStore
```

### Adding New Data to Pipeline

When adding a new field:

1. Add to `ResearchState` dataclass with default
2. Add key to `GraphState` TypedDict
3. Initialize in `create_initial_state()` in `api/state_builder.py`
4. Map in `build_research_state()` in `api/state_builder.py`

### Node Function Pattern

```python
def my_node(state: GraphState) -> GraphState:
    log = get_logger("api.graph.my_node", run_id=state.get("run_id"))
    
    # Read inputs with defaults
    input_data = state.get("some_key", default_value)
    warnings = state.get("warnings", [])

    # Process
    output = process(input_data)

    # Return ONLY the keys this node modifies
    return {"output_key": output, "warnings": warnings}
```

**Rules:**
- Never mutate `state` directly
- Return a new dict with only changed keys
- Use `.get()` with defaults for optional keys
- Preserve `warnings`/`errors` — append, don't replace

---

## Logging

### Getting a Logger

Always use `get_logger()` from `api/logging_setup`:

```python
from api.logging_setup import get_logger

# Without run correlation (module-level)
logger = get_logger(__name__)

# With run correlation (inside pipeline nodes)
log = get_logger("api.graph.plan", run_id=state.get("run_id"))
```

**Never use `logging.getLogger()` directly.**

### Logging Pattern in Pipeline Nodes

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

### Standard Log Event Names

| Event Pattern   | Usage                         |
| --------------- | ----------------------------- |
| `*_start`       | Beginning of an operation     |
| `*_complete`    | Successful completion         |
| `*_failed`      | Operation failure             |
| `*_skip`        | Operation skipped             |
| `*_retry`       | Retrying an operation         |
| `*_cache_hit`   | Retrieved from cache          |
| `*_rate_limited`| API rate limit encountered    |

### Environment Variables

| Variable                          | Default            | Options                              |
| --------------------------------- | ------------------ | ------------------------------------ |
| `AUTO_ANALYST_LOG_LEVEL`          | `DEBUG`            | `DEBUG`, `INFO`, `WARNING`, `ERROR`  |
| `AUTO_ANALYST_LOG_FORMAT`         | `plain`            | `plain`, `json`                      |
| `AUTO_ANALYST_LOG_FILE`           | `auto_analyst.log` | Path to log file                     |
| `AUTO_ANALYST_LOG_REDACT_QUERIES` | `false`            | `true`, `false`                      |

---

## LLM Integration

### LLM Backends

**Gemini (Default):**

```python
from tools.models import load_llm

llm = load_llm()  # Uses DEFAULT_LLM_MODEL (gemini-2.0-flash)
```

**HuggingFace Inference (Fallback):**

```python
from tools.models import load_huggingface_llm

hf_llm = load_huggingface_llm()
```

### LLM Call Contract

All LLM calls follow this exact pattern:

```python
result = llm(prompt)[0]["generated_text"]
```

Returns a list containing a dict with `generated_text` key (HuggingFace pipeline format).

### Response Parsing

Strip prompt echo when model repeats the prompt:

```python
output = llm(prompt)[0]["generated_text"]
answer = (
    output.split("Answer:", 1)[-1].strip()
    if "Answer:" in output
    else output.strip()
)
```

### Gemini Grounding

For web-augmented responses:

```python
from tools.gemini_grounding import query_with_grounding, GroundingResult

result: GroundingResult = query_with_grounding(query, run_id=run_id)
# result.answer, result.sources, result.success
```

### Environment Variables

| Variable                          | Default                          | Description                      |
| --------------------------------- | -------------------------------- | -------------------------------- |
| `AUTO_ANALYST_LLM`                | `gemini-2.0-flash`               | Default LLM model                |
| `AUTO_ANALYST_LLM_BACKEND`        | `gemini`                         | Backend (gemini/huggingface)     |
| `GOOGLE_API_KEY`                  | -                                | Single Gemini API key            |
| `GOOGLE_API_KEYS`                 | -                                | Comma-separated API keys         |
| `HUGGINGFACE_API_TOKEN`           | -                                | HuggingFace API token            |

---

## Vector Store

### Abstract Interface

```python
from abc import ABC, abstractmethod
from vector_store.base import ScoredChunk, VectorStore

class VectorStore(ABC):
    @abstractmethod
    def upsert(self, chunks: List[Chunk]) -> None: ...
    
    @abstractmethod
    def query(self, text: str, top_k: int = 5) -> List[ScoredChunk]: ...
    
    def clear(self) -> None: ...
```

### Implementations

| Backend             | File              | Use Case                             |
| ------------------- | ----------------- | ------------------------------------ |
| `ChromaVectorStore` | `chroma_store.py` | Default, persistent local storage    |
| `FaissVectorStore`  | `faiss_store.py`  | In-memory, faster for small datasets |

### Backend Selection

```python
# Controlled via environment variable
VECTOR_STORE_BACKEND = os.getenv("AUTO_ANALYST_VECTOR_STORE", "chroma")

# Factory function
from tools.retriever import build_vector_store
store = build_vector_store(model_name=embed_model, run_id=run_id)
```

### Embedding Model

```python
from tools.models import load_embedding_model

embedder = load_embedding_model(model_name="all-MiniLM-L6-v2")
vectors = embedder.encode(["text1", "text2"], convert_to_numpy=True)
```

### Best Practices

1. **Log all operations** — Include chunk counts, query times, scores
2. **Handle empty inputs** — Return early for empty chunk lists
3. **Preserve metadata** — Don't lose citation info during storage
4. **Normalize scores** — Ensure 0-1 range, higher is better
5. **Support run isolation** — Accept `run_id` parameter

---

## Evaluation

### RAG Metrics

| Metric                   | Function                 | Range | Interpretation                                      |
| ------------------------ | ------------------------ | ----- | --------------------------------------------------- |
| **Context Relevance**    | `context_relevance()`    | 0-1   | Avg cosine similarity between query and contexts    |
| **Context Sufficiency**  | `context_sufficiency()`  | 0-1   | Fraction of contexts above similarity threshold     |
| **Answer Relevance**     | `answer_relevance()`     | 0-1   | Similarity between answer and reference             |
| **Answer Correctness**   | `answer_correctness()`   | 0-1   | Direct similarity to ground truth                   |
| **Answer Hallucination** | `answer_hallucination()` | 0-1   | Fraction of unsupported sentences (lower is better) |

### Batch Evaluation

```python
from evaluation.metrics import evaluate_all

results = evaluate_all(
    question="What is X?",
    answer="X is ...",
    reference="X is the correct answer...",
    contexts=["context 1", "context 2"],
    model_name="all-MiniLM-L6-v2",
    thresholds=(0.5, 0.4)  # (sufficiency_threshold, hallucination_threshold)
)
```

### Interpretation Guidelines

- **Context Relevance < 0.5**: Retrieved contexts may be off-topic
- **Context Sufficiency < 0.5**: Insufficient evidence retrieved
- **Answer Relevance < 0.6**: Answer doesn't address the question
- **Answer Correctness < 0.6**: Factual issues likely
- **Answer Hallucination > 0.3**: Significant unsupported claims

---

## Testing

### Required Mocking Patterns

All external I/O must be mocked in tests:

**FakeLLM:**

```python
class FakeLLM:
    def __call__(self, prompt):
        if "Verified answer:" in prompt:
            return [{"generated_text": "Verified answer: validated [1]"}]
        return [{"generated_text": "Answer: drafted [1]"}]
```

**CapturingLLM (for testing prompts):**

```python
class CapturingLLM:
    def __init__(self, response: str):
        self.response = response
        self.last_prompt = ""

    def __call__(self, prompt):
        self.last_prompt = prompt
        return [{"generated_text": self.response}]
```

**FakeVectorStore:**

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

**Monkeypatching I/O:**

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

    monkeypatch.setattr("api.graph.run_search_tasks", fake_search)
    monkeypatch.setattr("api.graph.fetch_url", fake_fetch)
```

### Test Assertions

- Assert on `result.plan`, `result.search_results`, `result.documents`, `result.retrieved`, `result.citations`
- Check `result.verified_answer` for final output
- Use `assert result.errors == []` to catch silent failures
- Check `result.warnings` for non-fatal issues
- Verify conversation history is updated

### Running Tests

```bash
pytest                           # All tests
pytest tests/test_end_to_end.py  # Integration only
pytest -k "planner"              # Filter by name
pytest -v                        # Verbose output
pytest --cov=api --cov=tools     # With coverage
```

---

## Environment Variables Reference

| Variable                       | Default            | Description                             |
| ------------------------------ | ------------------ | --------------------------------------- |
| `AUTO_ANALYST_LLM`             | `gemini-2.0-flash` | LLM model identifier                    |
| `AUTO_ANALYST_LLM_BACKEND`     | `gemini`           | LLM backend (gemini, huggingface)       |
| `AUTO_ANALYST_EMBED`           | `all-MiniLM-L6-v2` | Embedding model                         |
| `AUTO_ANALYST_VECTOR_STORE`    | `chroma`           | Vector store backend                    |
| `AUTO_ANALYST_SEARCH_BACKENDS` | `tavily,gemini_grounding` | Comma-separated search backends  |
| `AUTO_ANALYST_GROQ_MODEL`      | `llama-3.3-70b-versatile` | Groq model name               |
| `GROQ_API_KEY`                 | -                  | Groq API key                            |
| `AUTO_ANALYST_LOG_LEVEL`       | `DEBUG`            | Log level                               |
| `AUTO_ANALYST_LOG_FORMAT`      | `plain`            | Log format (plain, json)                |
| `AUTO_ANALYST_LOG_FILE`        | `auto_analyst.log` | Log file path                           |
| `AUTO_ANALYST_LOG_REDACT_QUERIES` | `false`         | Redact queries in logs                  |
| `GOOGLE_API_KEY`               | -                  | Gemini API key                          |
| `GOOGLE_API_KEYS`              | -                  | Comma-separated API keys for rotation   |
| `HUGGINGFACE_API_TOKEN`        | -                  | HuggingFace API token                   |

---

## Code Style

### Do

- Use type hints on all function signatures
- Use `get_logger(__name__, run_id=state.get("run_id"))` for logging
- Use structured extras in log calls: `log.info("event", extra={"key": value})`
- Use `log.exception()` for errors (auto-captures stack trace)
- Keep modules small, composable, and testable
- Use dependency injection (pass models/stores into orchestrators)
- Prefer free/open-source dependencies only
- Return errors in state rather than raising exceptions in pipeline nodes

### Don't

- Don't use `logging.getLogger()` directly—use `get_logger()`
- Don't leak UI concerns into core modules (`api/`, `tools/`, `vector_store/`)
- Don't hardcode API keys or secrets
- Don't add heavy proprietary dependencies without approval
- Don't bypass the centralized logging configuration
- Don't mutate state directly in pipeline nodes

---

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

---

## PR Checklist

- [ ] All tests pass: `pytest`
- [ ] Code follows existing patterns and style
- [ ] Logging uses `get_logger()` with run correlation
- [ ] External I/O is mocked in new tests
- [ ] No secrets or API keys in code
- [ ] Small, focused diff with clear commit message

---

## When Stuck

- Ask a clarifying question before making speculative changes
- Propose a short plan for complex refactors
- Reference existing patterns in `tests/test_end_to_end.py` for mocking examples
- Check `.github/instructions/*.md` for domain-specific guidelines:
  - `clean-architecture.instructions.md` — Module boundaries and dependency rules
  - `state-management.instructions.md` — State types and node patterns
  - `logging.instructions.md` — Logging configuration and patterns
  - `llm-integration.instructions.md` — LLM backends and prompts
  - `vector-store.instructions.md` — Vector store implementations
  - `evaluation.instructions.md` — RAG metrics and quality assessment
  - `testing.instructions.md` — Test patterns and mocking
