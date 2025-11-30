---
applyTo: "evaluation/**"
---

# Evaluation Guidelines

## RAG Metrics

Five metrics are implemented in `evaluation/metrics.py`:

| Metric                   | Function                 | Range | Interpretation                                      |
| ------------------------ | ------------------------ | ----- | --------------------------------------------------- |
| **Context Relevance**    | `context_relevance()`    | 0-1   | Avg cosine similarity between query and contexts    |
| **Context Sufficiency**  | `context_sufficiency()`  | 0-1   | Fraction of contexts above similarity threshold     |
| **Answer Relevance**     | `answer_relevance()`     | 0-1   | Similarity between answer and reference             |
| **Answer Correctness**   | `answer_correctness()`   | 0-1   | Direct similarity to ground truth                   |
| **Answer Hallucination** | `answer_hallucination()` | 0-1   | Fraction of unsupported sentences (lower is better) |

## Metric Implementations

All metrics use embedding-based similarity via `tools/models.py:load_embedding_model()`:

```python
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

JSON file with list of evaluation items (see `data/sample_eval.json`):

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

## Thresholds

Default thresholds in `evaluate_all()`:

- **Context sufficiency**: 0.5 (contexts with similarity >= 0.5 are "sufficient")
- **Hallucination detection**: 0.4 (sentences with max context similarity < 0.4 are "hallucinated")

Adjust based on your embedding model's similarity distribution.

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

## Interpretation Guidelines

- **Context Relevance < 0.5**: Retrieved contexts may be off-topic
- **Context Sufficiency < 0.5**: Insufficient evidence retrieved
- **Answer Relevance < 0.6**: Answer doesn't address the question
- **Answer Correctness < 0.6**: Factual issues likely
- **Answer Hallucination > 0.3**: Significant unsupported claims
