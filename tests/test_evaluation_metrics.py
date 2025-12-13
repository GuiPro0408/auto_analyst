"""Tests for RAG evaluation metrics.

Tests context relevance, hallucination detection, and aggregate metrics.
"""

import pytest

from evaluation.metrics import answer_hallucination, context_relevance, evaluate_all
from tests.conftest import FakeEmbedder


@pytest.mark.unit
def test_context_relevance(monkeypatch):
    """Context relevance should return a score between 0 and 1."""
    monkeypatch.setattr(
        "evaluation.metrics.load_embedding_model", lambda model_name: FakeEmbedder()
    )
    score = context_relevance("question", ["context"], model_name="fake")
    assert 0.0 <= score <= 1.0


@pytest.mark.unit
def test_hallucination(monkeypatch):
    """Hallucination score should return a value between 0 and 1."""
    monkeypatch.setattr(
        "evaluation.metrics.load_embedding_model", lambda model_name: FakeEmbedder()
    )
    score = answer_hallucination("This is a sentence.", ["context"], model_name="fake")
    assert 0.0 <= score <= 1.0


@pytest.mark.unit
def test_evaluate_all(monkeypatch):
    """Evaluate all should return complete metrics dictionary."""
    monkeypatch.setattr(
        "evaluation.metrics.load_embedding_model", lambda model_name: FakeEmbedder()
    )
    metrics = evaluate_all("q", "a", "ref", ["ctx"], model_name="fake")
    assert set(metrics.keys()) == {
        "context_relevance",
        "context_sufficiency",
        "answer_relevance",
        "answer_correctness",
        "answer_hallucination",
    }


@pytest.mark.unit
def test_context_relevance_empty_contexts(monkeypatch):
    """Context relevance should handle empty context list."""
    monkeypatch.setattr(
        "evaluation.metrics.load_embedding_model", lambda model_name: FakeEmbedder()
    )
    score = context_relevance("question", [], model_name="fake")
    assert 0.0 <= score <= 1.0


@pytest.mark.unit
def test_hallucination_multiple_sentences(monkeypatch):
    """Hallucination should handle multi-sentence answers."""
    monkeypatch.setattr(
        "evaluation.metrics.load_embedding_model", lambda model_name: FakeEmbedder()
    )
    answer = "First sentence. Second sentence. Third sentence."
    score = answer_hallucination(answer, ["context"], model_name="fake")
    assert 0.0 <= score <= 1.0


@pytest.mark.unit
def test_evaluate_all_with_multiple_contexts(monkeypatch):
    """Evaluate all should work with multiple context strings."""
    monkeypatch.setattr(
        "evaluation.metrics.load_embedding_model", lambda model_name: FakeEmbedder()
    )
    contexts = ["context 1", "context 2", "context 3"]
    metrics = evaluate_all(
        "question", "answer", "reference", contexts, model_name="fake"
    )

    for key, value in metrics.items():
        assert 0.0 <= value <= 1.0, f"{key} should be between 0 and 1"
