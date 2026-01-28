from api.state import Chunk
from tools import reranker


def test_rerank_chunks_orders_by_model(monkeypatch):
    chunks = [
        Chunk(id="1", text="alpha beta"),
        Chunk(id="2", text="alpha"),
    ]

    class DummyModel:
        def predict(self, pairs):
            return [len(text) for _, text in pairs]

    reranked, scores = reranker.rerank_chunks(
        "query", chunks, model=DummyModel(), top_k=None
    )

    assert reranked[0].id == "1"
    assert scores[0] > scores[1]
