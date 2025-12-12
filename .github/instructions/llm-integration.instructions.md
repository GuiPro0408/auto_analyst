---
applyTo: "**"
---

# LLM Integration Guidelines

## Overview

Auto-Analyst uses Gemini as the primary LLM backend via the `google-genai` SDK, with HuggingFace Inference API as an optional fallback. All LLM interactions go through `tools/models.py`.

## LLM Backends

### Gemini (Default)

```python
from tools.models import load_llm

llm = load_llm()  # Uses DEFAULT_LLM_MODEL (gemini-2.0-flash)
```

The `GeminiLLM` class handles:
- API key rotation via `APIKeyRotator` for rate limit handling
- Multiple API keys support (`GOOGLE_API_KEYS` env var)
- Automatic fallback to secondary LLM on exhaustion
- HuggingFace-compatible output format

### HuggingFace Inference (Fallback)

```python
from tools.models import load_huggingface_llm

hf_llm = load_huggingface_llm()  # Uses HUGGINGFACE_INFERENCE_MODEL
```

## Call Contract

**All LLM calls follow this exact pattern:**

```python
result = llm(prompt)[0]["generated_text"]
```

This returns a list containing a dict with `generated_text` key, matching HuggingFace pipeline format. Both Gemini and HuggingFace backends conform to this interface.

## Prompt Templates

### Generator Prompt

```python
prompt = (
    "You are an evidence-based research assistant. Using only the context provided, "
    "write a comprehensive, well-structured answer to the user question.\n\n"
    "Guidelines:\n"
    "- Organize your response with clear sections using **bold headers** when appropriate\n"
    "- Provide detailed explanations, not just brief summaries\n"
    "- Use bullet points or numbered lists to present multiple items clearly\n"
    "- Cite supporting evidence inline using [n] where n matches the numbered context entries\n"
    "- Include relevant details like dates, names, descriptions when available\n"
    "- Do not fabricate details - only use information from the provided context\n\n"
    f"User question: {query}\n\nContext:\n{context_block}\n\nAnswer:"
)
```

### Verifier Prompt

```python
prompt = (
    "You are a fact-checking verifier. Review the draft answer against the provided context. "
    "Remove or correct any statements that are not directly supported by the context. "
    "Preserve the structure, formatting (headers, bullet points, lists), and level of detail from the draft. "
    "Keep inline citations [n] only when the claim is supported by the corresponding context entry. "
    "Do not shorten or oversimplify the answer - maintain comprehensive coverage.\n\n"
    f"User question: {query}\n\nContext:\n{context_block}\n\nDraft answer:\n{draft}\n\nVerified answer:"
)
```

### Smart Search Query Analysis Prompt

```python
QUERY_ANALYSIS_PROMPT = """Analyze this search query and provide a strategy.

Query: {query}

Respond in JSON only (no markdown):
{{
  "intent": "news|factual|comparison|howto|opinion",
  "entities": ["entity1", "entity2"],
  "topic": "technology|science|sports|entertainment|news|finance|health|general",
  "time_sensitivity": "realtime|recent|any",
  "suggested_searches": [
    {{"query": "specific search 1", "rationale": "why"}}
  ],
  "authoritative_sources": ["domain1.com", "domain2.com"]
}}"""
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

## Coherence Checking

Generated text is validated for coherence before use:

```python
def _is_coherent(
    text: str,
    min_words: int = 20,
    max_repetition_ratio: float = 0.4,
    min_alnum_ratio: float = 0.65,
    max_word_repeat: int = 8,
) -> bool:
    """Check if generated text appears coherent and not gibberish."""
```

Checks include:
- Minimum word count
- Word repetition ratio
- Alphanumeric character ratio
- Pattern detection (repeated words/fragments)

## Fallback Pattern

Always implement fallback when LLM output is unusable:

```python
def generate_answer(llm, query, retrieved, conversation_context=None):
    # ... LLM call ...
    
    if not _is_coherent(answer):
        logger.warning("generate_incoherent_answer")
        answer = _generate_fallback_answer(query, relevant_chunks)
    
    return answer, citations

def _generate_fallback_answer(query: str, chunks: List[Chunk]) -> str:
    """Generate a structured extractive answer when LLM output is incoherent."""
    # Build structured summary from retrieved chunks
```

## Gemini Grounding

For web-augmented responses, use `tools/gemini_grounding.py`:

```python
from tools.gemini_grounding import query_with_grounding, GroundingResult

result: GroundingResult = query_with_grounding(query, run_id=run_id)
# result.answer - The grounded response
# result.sources - List of GroundingSource with url, title, snippet
# result.success - Whether the call succeeded
# result.web_search_queries - Queries used by Gemini for grounding
```

The grounding tool uses `google-genai` SDK with `GoogleSearch` tool:

```python
from google import genai
from google.genai import types

google_search_tool = types.Tool(google_search=types.GoogleSearch())
response = client.models.generate_content(
    model=target_model,
    contents=query,
    config=types.GenerateContentConfig(tools=[google_search_tool]),
)
```

## API Key Rotation

For rate limit handling across multiple API keys:

```python
from api.key_rotator import APIKeyRotator, get_default_rotator

rotator = get_default_rotator(GEMINI_API_KEYS)
current_key = rotator.current_key

# On rate limit error:
has_more = rotator.mark_rate_limited(current_key)
if has_more:
    # Retry with next key
    continue
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
- `build_citations()` creates citation metadata and remaps duplicate URLs

## Configuration

Generation parameters in `api/config.py`:

```python
GENERATION_KWARGS = {
    "max_new_tokens": 1024,
    "temperature": 0.4,
    "do_sample": True,
}
```

Coherence thresholds:

```python
COHERENCE_MIN_WORDS = 20
COHERENCE_MAX_REPETITION_RATIO = 0.4
COHERENCE_MIN_ALNUM_RATIO = 0.65
COHERENCE_MAX_WORD_REPEAT = 8
```

## Environment Variables

| Variable                      | Default                              | Description                    |
| ----------------------------- | ------------------------------------ | ------------------------------ |
| `AUTO_ANALYST_LLM`            | `gemini-2.0-flash`                   | Default LLM model              |
| `AUTO_ANALYST_LLM_BACKEND`    | `gemini`                             | LLM backend (gemini/huggingface) |
| `AUTO_ANALYST_GEMINI_MODEL`   | `gemini-2.0-flash`                   | Gemini model name              |
| `GOOGLE_API_KEY`              | -                                    | Single Gemini API key          |
| `GOOGLE_API_KEYS`             | -                                    | Comma-separated API keys       |
| `HUGGINGFACE_API_TOKEN`       | -                                    | HuggingFace API token          |
| `AUTO_ANALYST_HF_INFERENCE_MODEL` | `mistralai/Mixtral-8x7B-Instruct-v0.1` | HuggingFace model        |
