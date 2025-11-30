---
applyTo: "tools/{planner,generator}.py"
---

# LLM Integration Guidelines

## LLM Loading

LLMs are loaded via `tools/models.py:load_llm()`:

```python
from tools.models import load_llm

llm = load_llm()  # Uses DEFAULT_LLM_MODEL from config
llm = load_llm(model_name="microsoft/Phi-3-mini-4k-instruct")
```

Models are cached with `@lru_cache` to prevent redundant loading.

## Call Contract

**All LLM calls follow this exact pattern:**

```python
result = llm(prompt)[0]["generated_text"]
```

This returns the HuggingFace text-generation pipeline format. Never assume OpenAI-style responses.

## Prompt Templates

### Planner Prompt

```python
prompt = (
    "You are a research planner. Break the user question into at most "
    f"{max_tasks} focused web search tasks. Use concise, self-contained queries. "
    "Return each task on its own line as '<query> -- <rationale>'.\n"
    f"User question: {query}"
)
```

### Generator Prompt

```python
prompt = (
    "You are an evidence-based research assistant. Using only the context provided, "
    "write a concise answer to the user question. Cite supporting evidence inline using [n] "
    "where n matches the numbered context entries. Do not fabricate details.\n\n"
    f"User question: {query}\n\nContext:\n{context_block}\n\nAnswer:"
)
```

### Verifier Prompt

```python
prompt = (
    "You are a fact-checking verifier. Review the draft answer against the provided context. "
    "Remove or correct any statements that are not directly supported. Preserve inline citations [n] "
    "only when the claim is supported by the corresponding context entry. Keep the answer concise.\n\n"
    f"User question: {query}\n\nContext:\n{context_block}\n\nDraft answer:\n{draft}\n\nVerified answer:"
)
```

## Response Parsing

Strip prompt echo when model repeats the prompt:

```python
output = llm(prompt)[0]["generated_text"]
answer = (
    output.split("Answer:", 1)[-1].strip()
    if "Answer:" in output
    else output.strip()
)
```

## Fallback Pattern

Always implement heuristic fallback when LLM fails:

```python
def plan_query(query: str, llm=None, max_tasks: int = 4) -> List[SearchQuery]:
    if not llm:
        return heuristic_plan(query, max_tasks=max_tasks)

    try:
        result = llm(prompt)[0]["generated_text"]
        tasks = parse_llm_output(result)
        if tasks:
            return tasks
    except Exception:
        pass  # Swallow and fall back

    return heuristic_plan(query, max_tasks=max_tasks)
```

## Context Formatting

Format retrieved chunks for LLM context:

```python
def _format_context(chunks: List[Chunk]) -> str:
    lines = []
    for idx, chunk in enumerate(chunks):
        meta = chunk.metadata or {}
        title = meta.get("title") or "Source"
        url = meta.get("url") or ""
        lines.append(f"[{idx + 1}] {title} ({url}) :: {chunk.text}")
    return "\n".join(lines)
```

## Citation Convention

- Generator produces `[n]` markers matching context indices (1-indexed)
- Verifier prunes citations that lack supporting evidence
- `build_citations()` creates citation metadata for UI display

## Configuration

Generation parameters in `api/config.py`:

```python
GENERATION_KWARGS = {
    "max_new_tokens": 512,
    "temperature": 0.4,
    "do_sample": True,
}
```
