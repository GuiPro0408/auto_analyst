import numpy as np

from evaluation.metrics import answer_hallucination, context_relevance, evaluate_all


class FakeEmbedder:
    def encode(self, texts, convert_to_numpy=True, show_progress_bar=False):
        # Deterministic embeddings for testing
        return np.ones((len(texts), 3))


def test_context_relevance(monkeypatch):
    monkeypatch.setattr(
        "evaluation.metrics.load_embedding_model", lambda model_name: FakeEmbedder()
    )
    score = context_relevance("question", ["context"], model_name="fake")
    assert 0.0 <= score <= 1.0


def test_hallucination(monkeypatch):
    monkeypatch.setattr(
        "evaluation.metrics.load_embedding_model", lambda model_name: FakeEmbedder()
    )
    score = answer_hallucination("This is a sentence.", ["context"], model_name="fake")
    assert 0.0 <= score <= 1.0


def test_evaluate_all(monkeypatch):
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
