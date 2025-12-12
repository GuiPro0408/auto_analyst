---
applyTo: "evaluation/**"
---

# Evaluation Guidelines

## Overview

Auto-Analyst uses embedding-based RAG evaluation metrics implemented in `evaluation/metrics.py`. All metrics use cosine similarity via sentence-transformers embeddings.

## RAG Metrics

Five metrics are implemented:

| Metric                   | Function                 | Range | Interpretation                                      |
| ------------------------ | ------------------------ | ----- | --------------------------------------------------- |
| **Context Relevance**    | `context_relevance()`    | 0-1   | Avg cosine similarity between query and contexts    |
| **Context Sufficiency**  | `context_sufficiency()`  | 0-1   | Fraction of contexts above similarity threshold     |
| **Answer Relevance**     | `answer_relevance()`     | 0-1   | Similarity between answer and reference             |
| **Answer Correctness**   | `answer_correctness()`   | 0-1   | Direct similarity to ground truth                   |
| **Answer Hallucination** | `answer_hallucination()` | 0-1   | Fraction of unsupported sentences (lower is better) |

## Core Implementation

All metrics use batched embedding for efficiency:

```python
from tools.models import load_embedding_model

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-10
    return float(np.dot(a, b) / denom)

def _embed_texts(texts: Iterable[str], model_name: str) -> np.ndarray:
    embedder = load_embedding_model(model_name=model_name)
    return embedder.encode(list(texts), convert_to_numpy=True, show_progress_bar=False)
```

## Batch Evaluation

Use `evaluate_all()` for computing all metrics at once:

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
# Returns: {"context_relevance": 0.8, "context_sufficiency": 0.6, ...}
```

## Evaluation Dataset Format

JSON file with list of evaluation items:

```json
[
  {
    "question": "User query",
    "reference_answer": "Ground truth answer",
    "answer": "Model-generated answer",
    "contexts": ["Retrieved context 1", "Retrieved context 2"]
  }
]
```

## Running Evaluation

```bash
python evaluation/run_evaluation.py --dataset data/sample_eval.json --model all-MiniLM-L6-v2
```

## Default Thresholds

Thresholds in `evaluate_all()`:

- **Context sufficiency**: 0.5 (contexts with similarity >= 0.5 are "sufficient")
- **Hallucination detection**: 0.4 (sentences with max context similarity < 0.4 are "hallucinated")

Adjust based on your embedding model's similarity distribution.

## Hallucination Detection

The `answer_hallucination()` metric uses batched embedding for efficiency:

```python
def answer_hallucination(
    answer: str, contexts: List[str], model_name: str, threshold: float = 0.4
) -> float:
    """Fraction of sentences that are unsupported by any context (lower is better)."""
    sentences = [s.strip() for s in re.split(r"[.!?]", answer) if s.strip()]
    
    # Batch embed: contexts first, then all sentences
    all_texts = contexts + sentences
    all_embeddings = _embed_texts(all_texts, model_name=model_name)
    ctx_embeddings = all_embeddings[: len(contexts)]
    sent_embeddings = all_embeddings[len(contexts) :]
    
    hallucinatory = 0
    for sent_vec in sent_embeddings:
        sims = [_cosine(sent_vec, ctx_vec) for ctx_vec in ctx_embeddings]
        supported = max(sims) >= threshold
        if not supported:
            hallucinatory += 1
    return hallucinatory / len(sentences)
```

## Quality Control Integration

The pipeline uses `tools/quality_control.py` for runtime quality assessment:

```python
from tools.quality_control import assess_answer, improve_answer

# Heuristic-based assessment (no LLM call)
assessment = assess_answer(
    question=query,
    answer=answer,
    contexts=retrieved_chunks,
    retrieval_scores=scores,
    run_id=run_id,
)

# Checks performed:
# 1. Answer is not empty
# 2. Supporting contexts exist
# 3. Answer has citations [n]
# 4. No fallback phrases ("Unable to generate", etc.)
# 5. Retrieval scores above minimum threshold
# 6. List-style questions have structured format
```

## Adding New Metrics

1. Add function in `evaluation/metrics.py`:

```python
def new_metric(answer: str, contexts: List[str], model_name: str, **kwargs) -> float:
    # Implementation using _embed_texts() and _cosine()
    return score
```

2. Include in `evaluate_all()` return dict:

```python
def evaluate_all(...) -> Dict[str, float]:
    nm = new_metric(answer, contexts, model_name=model_name)
    return {..., "new_metric": nm}
```

3. Add test in `tests/test_evaluation_metrics.py`.

## Best Practices

1. **Use batched embedding** — Avoid embedding one text at a time
2. **Handle edge cases** — Empty answers, empty contexts
3. **Configure thresholds** — Adjust based on embedding model characteristics
4. **Log metric values** — Include in pipeline logging for debugging

## Interpretation Guidelines

- **Context Relevance < 0.5**: Retrieved contexts may be off-topic
- **Context Sufficiency < 0.5**: Insufficient evidence retrieved
- **Answer Relevance < 0.6**: Answer doesn't address the question
- **Answer Correctness < 0.6**: Factual issues likely
- **Answer Hallucination > 0.3**: Significant unsupported claims
