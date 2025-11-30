---
applyTo: "api/**"
---

# State Management Guidelines

## Core State Types

All pipeline data flows through types defined in `api/state.py`:

### Domain Entities (Dataclasses)

```python
@dataclass
class SearchQuery:
    text: str
    rationale: str = ""

@dataclass
class SearchResult:
    url: str
    title: str
    snippet: str = ""
    source: str = "web"

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
```

### Pipeline State

```python
@dataclass
class ResearchState:
    # Full result object returned by run_research()
    query: str
    plan: List[SearchQuery]
    search_results: List[SearchResult]
    documents: List[Document]
    chunks: List[Chunk]
    retrieved: List[Chunk]
    draft_answer: str
    verified_answer: str
    citations: List[Dict[str, str]]
    errors: List[str]

class GraphState(TypedDict, total=False):
    # TypedDict for LangGraph node inputs/outputs
    # Keys mirror ResearchState fields
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

3. Initialize in `run_research()` initial state dict in `api/graph.py`:

   ```python
   initial_state = {
       ...
       "new_field": "",  # Match the default
   }
   ```

4. Map in `run_research()` return statement:
   ```python
   return ResearchState(
       ...
       new_field=result.get("new_field", ""),
   )
   ```

## Node Function Pattern

Every node reads from state dict and returns only modified keys:

```python
def my_node(state: Dict) -> Dict:
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
- Type hints use `Dict` not `GraphState` (LangGraph requirement)

## Error Handling

Use the `errors` list for recoverable errors:

```python
def node_with_errors(state: Dict) -> Dict:
    errors = state.get("errors", [])
    try:
        result = risky_operation()
    except Exception as e:
        errors.append(f"Node failed: {e}")
        result = fallback_value
    return {"output": result, "errors": errors}
```
