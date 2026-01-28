---
applyTo: "**"
---

# State Management Guidelines

## Overview

All pipeline data flows through types defined in `api/state.py`. The system uses dataclasses for domain entities and TypedDict for LangGraph state.

## Core State Types

### Domain Entities (Dataclasses)

```python
@dataclass
class SearchQuery:
    text: str
    rationale: str = ""
    topic: str = ""

@dataclass
class SearchResult:
    url: str
    title: str
    snippet: str = ""
    source: str = "web"
    content: str = ""  # Pre-fetched content (e.g., grounded search summaries)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Document:
    url: str
    title: str
    content: str
    media_type: str = "text"
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Chunk:
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ConversationTurn:
    query: str
    answer: str
    citations: List[Dict[str, str]] = field(default_factory=list)
    timestamp: float = field(default_factory=time)
    
    def to_dict(self) -> Dict[str, Any]: ...
    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ConversationTurn": ...
```

### Pipeline State (ResearchState)

```python
@dataclass
class ResearchState:
    query: str
    run_id: str = ""
    plan: List[SearchQuery] = field(default_factory=list)
    search_results: List[SearchResult] = field(default_factory=list)
    documents: List[Document] = field(default_factory=list)
    chunks: List[Chunk] = field(default_factory=list)
    retrieved: List[Chunk] = field(default_factory=list)
    retrieval_scores: List[float] = field(default_factory=list)
    draft_answer: str = ""
    verified_answer: str = ""
    citations: List[Dict[str, str]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    adaptive_iterations: int = 0
    qc_passes: int = 0
    qc_notes: List[str] = field(default_factory=list)
    time_sensitive: bool = False
    conversation_history: List[ConversationTurn] = field(default_factory=list)
    grounded_answer: str = ""
    grounded_sources: List[Chunk] = field(default_factory=list)
    query_type: str = "factual"  # factual, recommendation, or creative

    def add_error(self, message: str) -> None:
        self.errors.append(message)
```

### LangGraph State (GraphState TypedDict)

```python
class GraphState(TypedDict, total=False):
    query: str
    run_id: str
    plan: List[SearchQuery]
    search_results: List[SearchResult]
    documents: List[Document]
    chunks: List[Chunk]
    retrieved: List[Chunk]
    retrieval_scores: List[float]  # Similarity scores from vector store query
    draft_answer: str
    verified_answer: str
    citations: List[Dict[str, str]]
    errors: List[str]
    warnings: List[str]
    adaptive_iterations: int
    qc_passes: int
    qc_notes: List[str]
    time_sensitive: bool
    conversation_history: List[ConversationTurn]
    grounded_answer: str  # Direct answer from Gemini grounding
    grounded_sources: List[Chunk]  # Sources from grounding for citations
    query_type: str  # Query classification: factual, recommendation, creative
```

## Adding New Data to the Pipeline

**Checklist when adding a new field:**

1. Add field to `ResearchState` dataclass with default:

   ```python
   new_field: str = ""
   # or
   new_field: List[SomeType] = field(default_factory=list)
   ```

2. Add key to `GraphState` TypedDict:

   ```python
   class GraphState(TypedDict, total=False):
       ...
       new_field: str  # or appropriate type
   ```

3. Initialize in `create_initial_state()` in `api/state_builder.py`:

   ```python
   def create_initial_state(query: str, run_id: str, history: List, query_type: str = "factual") -> Dict:
       return {
           ...
           "new_field": "",  # Match the default
       }
   ```

4. Map in `build_research_state()` in `api/state_builder.py`:

   ```python
   def build_research_state(...) -> ResearchState:
       return ResearchState(
           ...
           new_field=result.get("new_field", ""),
       )
   ```

## Node Function Pattern

Every node reads from state dict and returns only modified keys:

```python
def my_node(state: GraphState) -> GraphState:
    log = get_logger("api.graph.my_node", run_id=state.get("run_id"))
    
    # Read inputs
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
- Type hints use `GraphState` for clarity

## Error Handling

Use the `errors` and `warnings` lists for recoverable issues:

```python
def node_with_errors(state: GraphState) -> GraphState:
    errors = state.get("errors", [])
    warnings = state.get("warnings", [])
    
    try:
        result = risky_operation()
    except Exception as e:
        errors.append(f"Node failed: {e}")
        result = fallback_value
        
    return {"output": result, "errors": errors, "warnings": warnings}
```

## State Builder Utilities

`api/state_builder.py` provides helpers for state construction:

```python
from api.state_builder import (
    build_research_state,      # Convert GraphState result to ResearchState
    create_initial_state,      # Create initial state dict for workflow
    normalize_conversation_history,  # Convert dicts to ConversationTurn
    extract_grounded_answer,   # Extract grounded answer from search results
)
```

### Creating Initial State

```python
def create_initial_state(
    query: str,
    run_id: str,
    conversation_history: List[ConversationTurn],
) -> Dict[str, Any]:
    """Create the initial state dictionary for the workflow."""
```

### Building Final ResearchState

```python
def build_research_state(
    query: str,
    run_id: str,
    result: Dict[str, Any],
    conversation_history: List[ConversationTurn],
) -> ResearchState:
    """Build a ResearchState from workflow result dict."""
```

## Grounded Answer Flow

When Gemini grounding provides a direct answer:

1. `search_node` extracts grounded answer and sources from search results
2. State carries `grounded_answer` and `grounded_sources` through nodes
3. `generate_node` checks for grounded answer and uses it directly
4. Citations are built from grounded sources

```python
# In search_node
if result.source == "gemini_grounding" and result.content:
    grounded_answer = result.content
    grounded_sources = [
        Chunk(id=f"grounding_{idx}", text=r.snippet, metadata={...})
        for idx, r in enumerate(results)
        if r.source == "gemini_grounding" and r.url
    ]
    
return {
    "search_results": results,
    "grounded_answer": grounded_answer,
    "grounded_sources": grounded_sources,
}
```

## Conversation History

Multi-turn support via `api/memory.py`:

```python
from api.memory import (
    trim_history,           # Keep recent N turns
    summarize_history,      # Create text summary
    resolve_followup_query, # Handle pronoun references
    append_turn,           # Add new turn
)

# In run_research()
normalized_history = normalize_conversation_history(conversation_history)
history_window = trim_history(normalized_history, max_turns=CONVERSATION_MEMORY_TURNS)

# After answer generation
updated_history = append_turn(
    history_window,
    query=query,
    answer=answer_text,
    citations=research_state.citations,
    max_turns=CONVERSATION_MEMORY_TURNS,
)
```

## Caching

Results are cached via `api/cache_manager.py`:

```python
from api.cache_manager import CacheManager

cache_manager = CacheManager(CACHE_DB_PATH, CACHE_TTL_SECONDS, run_id=run_id)

# Check cache
cached_result = cache_manager.get_cached_result(query)
if cached_result:
    return cached_result

# ... run pipeline ...

# Save to cache
cache_manager.save_result(query, research_state)
```

Cache skips:
- Time-sensitive queries
- Fallback/low-context results
- Grounded results (optional)

## Best Practices

1. **Always use defaults** — State keys may be missing
2. **Preserve warnings/errors** — Append to existing lists, don't replace
3. **Pass grounded state** — Preserve `grounded_answer`/`grounded_sources` through nodes
4. **Use state_builder** — Don't manually construct ResearchState
5. **Type annotate** — Use `List[SearchQuery]` not just `list`
