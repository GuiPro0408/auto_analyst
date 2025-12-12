"""Evaluation metrics for RAG systems following the project methodology.

This module provides embedding-based evaluation metrics for RAG pipelines:
- Context Relevance: How relevant retrieved contexts are to the query
- Context Sufficiency: Fraction of contexts meeting relevance threshold
- Answer Relevance: Similarity between generated and reference answers
- Answer Correctness: Direct similarity to ground truth
- Answer Hallucination: Fraction of unsupported claims (lower is better)

All metrics use cosine similarity via sentence-transformers embeddings.
The evaluate_all() function batches embeddings for efficiency.
"""

import re
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from tools.models import load_embedding_model


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-10
    return float(np.dot(a, b) / denom)


def _embed_texts(texts: Iterable[str], model_name: str) -> np.ndarray:
    """Embed multiple texts using the specified model.

    Uses lru_cache on load_embedding_model for model reuse.
    """
    embedder = load_embedding_model(model_name=model_name)
    return embedder.encode(list(texts), convert_to_numpy=True, show_progress_bar=False)


def context_relevance(
    query: str,
    contexts: List[str],
    model_name: str,
    *,
    precomputed_embeddings: Optional[Dict[str, np.ndarray]] = None,
) -> float:
    """Average cosine similarity between query and each context snippet.

    Args:
        query: The user's query.
        contexts: List of retrieved context strings.
        model_name: Sentence-transformers model name.
        precomputed_embeddings: Optional dict with 'query' and 'contexts' keys
            containing pre-computed embedding arrays.

    Returns:
        Average similarity score (0-1).
    """
    if not contexts:
        return 0.0

    if precomputed_embeddings:
        query_vec = precomputed_embeddings["query"]
        ctx_vecs = precomputed_embeddings["contexts"]
    else:
        embeddings = _embed_texts([query] + contexts, model_name=model_name)
        query_vec, ctx_vecs = embeddings[0], embeddings[1:]

    sims = [_cosine(query_vec, vec) for vec in ctx_vecs]
    return float(np.mean(sims))


def context_sufficiency(
    query: str,
    contexts: List[str],
    model_name: str,
    threshold: float = 0.5,
    *,
    precomputed_embeddings: Optional[Dict[str, np.ndarray]] = None,
) -> float:
    """Portion of contexts whose similarity to the query exceeds the threshold.

    Args:
        query: The user's query.
        contexts: List of retrieved context strings.
        model_name: Sentence-transformers model name.
        threshold: Minimum similarity to count as "sufficient".
        precomputed_embeddings: Optional dict with pre-computed embeddings.

    Returns:
        Fraction of contexts above threshold (0-1).
    """
    if not contexts:
        return 0.0

    if precomputed_embeddings:
        query_vec = precomputed_embeddings["query"]
        ctx_vecs = precomputed_embeddings["contexts"]
    else:
        embeddings = _embed_texts([query] + contexts, model_name=model_name)
        query_vec, ctx_vecs = embeddings[0], embeddings[1:]

    sims = [_cosine(query_vec, vec) for vec in ctx_vecs]
    return float(np.mean([sim >= threshold for sim in sims]))


def answer_relevance(
    question: str,
    answer: str,
    reference: str,
    model_name: str,
    *,
    precomputed_embeddings: Optional[Dict[str, np.ndarray]] = None,
) -> float:
    """Similarity between model answer and reference answer.

    Args:
        question: The user's question (unused but kept for API consistency).
        answer: The model-generated answer.
        reference: The reference/expected answer.
        model_name: Sentence-transformers model name.
        precomputed_embeddings: Optional dict with 'answer' and 'reference' keys.

    Returns:
        Similarity score (0-1).
    """
    if precomputed_embeddings:
        ans_vec = precomputed_embeddings["answer"]
        ref_vec = precomputed_embeddings["reference"]
    else:
        embeddings = _embed_texts([question, answer, reference], model_name=model_name)
        _, ans_vec, ref_vec = embeddings

    return _cosine(ans_vec, ref_vec)


def answer_correctness(
    answer: str,
    reference: str,
    model_name: str,
    *,
    precomputed_embeddings: Optional[Dict[str, np.ndarray]] = None,
) -> float:
    """Similarity between answer and trusted reference answer.

    Args:
        answer: The model-generated answer.
        reference: The reference/expected answer.
        model_name: Sentence-transformers model name.
        precomputed_embeddings: Optional dict with 'answer' and 'reference' keys.

    Returns:
        Similarity score (0-1).
    """
    if precomputed_embeddings:
        ans_vec = precomputed_embeddings["answer"]
        ref_vec = precomputed_embeddings["reference"]
    else:
        embeddings = _embed_texts([answer, reference], model_name=model_name)
        ans_vec, ref_vec = embeddings[0], embeddings[1]

    return _cosine(ans_vec, ref_vec)


def answer_hallucination(
    answer: str,
    contexts: List[str],
    model_name: str,
    threshold: float = 0.4,
    *,
    precomputed_embeddings: Optional[Dict[str, np.ndarray]] = None,
) -> float:
    """Fraction of sentences that are unsupported by any context (lower is better).

    Uses batched embedding for efficiency - embeds all sentences at once instead of
    one-by-one.

    Args:
        answer: The model-generated answer.
        contexts: List of retrieved context strings.
        model_name: Sentence-transformers model name.
        threshold: Minimum similarity for a sentence to be "supported".
        precomputed_embeddings: Optional dict with 'contexts' and 'sentences' keys.

    Returns:
        Fraction of unsupported sentences (0-1). Lower is better.
    """
    if not answer or not contexts:
        return 1.0 if answer else 0.0

    sentences = [s.strip() for s in re.split(r"[.!?]", answer) if s.strip()]
    if not sentences:
        return 0.0

    if precomputed_embeddings:
        ctx_embeddings = precomputed_embeddings["contexts"]
        sent_embeddings = precomputed_embeddings["sentences"]
    else:
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
    """Compute all metrics in one call with batched embeddings.

    This function optimizes embedding computation by batching all texts
    together, reducing from ~40 embedding operations to ~8 for a typical
    evaluation (80% reduction).

    Args:
        question: The user's question/query.
        answer: The model-generated answer.
        reference: The reference/expected answer.
        contexts: List of retrieved context strings.
        model_name: Sentence-transformers model name.
        thresholds: Tuple of (sufficiency_threshold, hallucination_threshold).

    Returns:
        Dictionary with all metric scores.
    """
    sufficiency_threshold, hallucination_threshold = thresholds

    # Extract sentences from answer for hallucination check
    sentences = [s.strip() for s in re.split(r"[.!?]", answer) if s.strip()]

    # Batch embed all unique texts at once
    # Order: question, answer, reference, contexts..., sentences...
    all_texts = [question, answer, reference] + contexts + sentences
    all_embeddings = _embed_texts(all_texts, model_name=model_name)

    # Extract embeddings by position
    question_vec = all_embeddings[0]
    answer_vec = all_embeddings[1]
    reference_vec = all_embeddings[2]
    ctx_vecs = all_embeddings[3 : 3 + len(contexts)]
    sent_vecs = all_embeddings[3 + len(contexts) :]

    # Compute context_relevance
    if contexts:
        ctx_sims = [_cosine(question_vec, vec) for vec in ctx_vecs]
        cr = float(np.mean(ctx_sims))
        cs = float(np.mean([sim >= sufficiency_threshold for sim in ctx_sims]))
    else:
        cr = 0.0
        cs = 0.0

    # Compute answer_relevance and answer_correctness
    ar = _cosine(answer_vec, reference_vec)
    ac = _cosine(answer_vec, reference_vec)

    # Compute answer_hallucination
    if sentences and contexts:
        hallucinatory = 0
        for sent_vec in sent_vecs:
            sims = [_cosine(sent_vec, ctx_vec) for ctx_vec in ctx_vecs]
            if max(sims) < hallucination_threshold:
                hallucinatory += 1
        ah = hallucinatory / len(sentences)
    elif answer:
        ah = 1.0  # No contexts to verify against
    else:
        ah = 0.0  # No answer to check

    return {
        "context_relevance": cr,
        "context_sufficiency": cs,
        "answer_relevance": ar,
        "answer_correctness": ac,
        "answer_hallucination": ah,
    }
