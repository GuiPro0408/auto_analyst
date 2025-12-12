---
applyTo: "**"
---

# Clean Architecture Guidelines

## Overview

Auto-Analyst follows a clean architecture with strict module boundaries. Each layer has clear responsibilities and import rules.

## Module Boundaries

| Directory       | Responsibility                                     | May Import From                       |
| --------------- | -------------------------------------------------- | ------------------------------------- |
| `api/`          | Orchestration, state types, config, logging, cache | `tools/`, `vector_store/`             |
| `tools/`        | Pure functional units (search, parse, generate)    | `api/state.py`, `api/config.py`, `api/logging_setup.py` |
| `vector_store/` | Storage abstractions (VectorStore interface)       | `api/state.py`, `api/logging_setup.py` |
| `ui/`           | Streamlit interface                                | All modules                           |
| `evaluation/`   | RAG metrics                                        | `api/state.py`, `tools/models.py`     |

## Core Principles

### 1. Single Responsibility

Each module handles one concern:

- `api/graph.py` — LangGraph pipeline orchestration
- `api/state.py` — Dataclasses and TypedDict definitions
- `api/config.py` — Environment variable configuration
- `api/logging_setup.py` — Centralized logging with run correlation
- `api/cache_manager.py` — Query result caching
- `api/key_rotator.py` — API key rotation for rate limits
- `api/memory.py` — Conversation history management
- `api/state_builder.py` — State construction helpers

### 2. Dependency Injection

Pass dependencies explicitly rather than creating them internally:

```python
# Good: Accept dependencies as parameters
def run_research(
    query: str,
    llm=None,  # Injectable
    vector_store: Optional[VectorStore] = None,  # Injectable
    embed_model: str = DEFAULT_EMBED_MODEL,
    top_k: int = TOP_K_RESULTS,
) -> ResearchState:
    llm = llm or load_llm()
    store = vector_store or build_vector_store(model_name=embed_model, run_id=run_id)

# Bad: Hard-coded dependencies
def run_research(query: str) -> ResearchState:
    llm = load_llm()  # Not injectable, hard to test
```

### 3. State Immutability in Pipeline Nodes

Pipeline nodes receive state and return only modified keys:

```python
def my_node(state: GraphState) -> GraphState:
    # Read inputs
    input_data = state.get("some_key", default_value)
    
    # Process
    output = process(input_data)
    
    # Return ONLY the keys this node modifies
    return {"output_key": output}
```

**Rules:**
- Never mutate `state` directly
- Return a new dict with only changed keys
- Use `.get()` with defaults for optional keys

### 4. Error Handling in Pipeline

Use the `errors` and `warnings` lists for recoverable issues:

```python
def node_with_errors(state: Dict) -> Dict:
    warnings = state.get("warnings", [])
    errors = state.get("errors", [])
    try:
        result = risky_operation()
    except Exception as e:
        errors.append(f"Node failed: {e}")
        result = fallback_value
    return {"output": result, "errors": errors, "warnings": warnings}
```

### 5. No UI in Core Modules

`tools/` and `api/` must never import from `ui/` or use Streamlit directly.

## Pipeline Structure

The LangGraph pipeline flows through these nodes:

```
START → plan → search → fetch → retrieve → adaptive → generate → verify → qc → END
```

Key nodes:
- **plan**: Decompose query into search tasks
- **search**: Execute searches via Tavily/Gemini grounding
- **fetch**: Download and parse documents
- **retrieve**: Vector similarity search with optional reranking
- **adaptive**: Context assessment and iterative refinement
- **generate**: LLM answer generation with citations
- **verify**: LLM fact-checking against context
- **qc**: Quality control assessment and improvement

## Search Backend Architecture

Search uses a backend abstraction pattern:

```python
class SearchBackend(ABC):
    name: str = "base"
    
    @abstractmethod
    def search(
        self, query: str, max_results: int = 5, run_id: Optional[str] = None
    ) -> Tuple[List[SearchResult], List[str]]:
        pass
```

Current implementations:
- `GeminiGroundingBackend` — Gemini 2.0+ with Google Search grounding
- `TavilyBackend` — Tavily API for RAG-optimized search

## Caching Architecture

Query results are cached via `CacheManager`:

- SQLite-backed persistent cache
- TTL-based expiration
- Skip caching for time-sensitive queries
- Skip caching for fallback/low-context results

## Conversation Memory

Multi-turn support via `api/memory.py`:

- `trim_history()` — Keep recent N turns
- `summarize_history()` — Create text summary for context
- `resolve_followup_query()` — Inject context for pronoun references
- `append_turn()` — Add new conversation turn

## Best Practices

1. **Keep modules small** — Single file should do one thing
2. **Use type hints** — All function signatures should be typed
3. **Prefer composition** — Build complex behavior from simple functions
4. **Test in isolation** — Each module should be independently testable
5. **Log with correlation** — Always pass `run_id` for tracing
