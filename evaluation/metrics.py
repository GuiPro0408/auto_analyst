"""Evaluation metrics for RAG systems following the project methodology."""

import re
from typing import Dict, Iterable, List, Tuple

import numpy as np

from tools.models import load_embedding_model


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-10
    return float(np.dot(a, b) / denom)


def _embed_texts(texts: Iterable[str], model_name: str) -> np.ndarray:
    embedder = load_embedding_model(model_name=model_name)
    return embedder.encode(list(texts), convert_to_numpy=True, show_progress_bar=False)


def context_relevance(query: str, contexts: List[str], model_name: str) -> float:
    """Average cosine similarity between query and each context snippet."""
    if not contexts:
        return 0.0
    embeddings = _embed_texts([query] + contexts, model_name=model_name)
    query_vec, ctx_vecs = embeddings[0], embeddings[1:]
    sims = [_cosine(query_vec, vec) for vec in ctx_vecs]
    return float(np.mean(sims))


def context_sufficiency(
    query: str, contexts: List[str], model_name: str, threshold: float = 0.5
) -> float:
    """Portion of contexts whose similarity to the query exceeds the threshold."""
    if not contexts:
        return 0.0
    embeddings = _embed_texts([query] + contexts, model_name=model_name)
    query_vec, ctx_vecs = embeddings[0], embeddings[1:]
    sims = [_cosine(query_vec, vec) for vec in ctx_vecs]
    return float(np.mean([sim >= threshold for sim in sims]))


def answer_relevance(
    question: str, answer: str, reference: str, model_name: str
) -> float:
    """Similarity between model answer and reference answer, weighted by alignment to the question."""
    embeddings = _embed_texts([question, answer, reference], model_name=model_name)
    _, ans_vec, ref_vec = embeddings
    return _cosine(ans_vec, ref_vec)


def answer_correctness(answer: str, reference: str, model_name: str) -> float:
    """Similarity between answer and trusted reference answer."""
    embeddings = _embed_texts([answer, reference], model_name=model_name)
    return _cosine(embeddings[0], embeddings[1])


def answer_hallucination(
    answer: str, contexts: List[str], model_name: str, threshold: float = 0.4
) -> float:
    """Fraction of sentences that are unsupported by any context (lower is better).

    Uses batched embedding for efficiency - embeds all sentences at once instead of
    one-by-one.
    """
    if not answer or not contexts:
        return 1.0 if answer else 0.0
    sentences = [s.strip() for s in re.split(r"[.!?]", answer) if s.strip()]
    if not sentences:
        return 0.0

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


def evaluate_all(
    question: str,
    answer: str,
    reference: str,
    contexts: List[str],
    model_name: str,
    thresholds: Tuple[float, float] = (0.5, 0.4),
) -> Dict[str, float]:
    """Compute all metrics in one call."""
    cr = context_relevance(question, contexts, model_name=model_name)
    cs = context_sufficiency(
        question, contexts, model_name=model_name, threshold=thresholds[0]
    )
    ar = answer_relevance(question, answer, reference, model_name=model_name)
    ac = answer_correctness(answer, reference, model_name=model_name)
    ah = answer_hallucination(
        answer, contexts, model_name=model_name, threshold=thresholds[1]
    )
    return {
        "context_relevance": cr,
        "context_sufficiency": cs,
        "answer_relevance": ar,
        "answer_correctness": ac,
        "answer_hallucination": ah,
    }
