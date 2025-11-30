import numpy as np

from api.state import Chunk
from vector_store.faiss_store import FaissVectorStore


class FakeEmbedder:
    def encode(
        self,
        texts,
        convert_to_numpy=False,
        normalize_embeddings=False,
        show_progress_bar=False,
    ):
        arr = np.array(
            [[float(i + 1) for i in range(4)] for _ in texts], dtype="float32"
        )
        if convert_to_numpy:
            return arr
        return arr.tolist()


def test_faiss_store_round_trip(monkeypatch):
    monkeypatch.setattr(
        "vector_store.faiss_store.load_embedding_model",
        lambda model_name: FakeEmbedder(),
    )
    store = FaissVectorStore(model_name="fake")
    chunk = Chunk(
        id="c1",
        text="hello world",
        metadata={"title": "Title", "url": "http://example.com"},
    )
    store.upsert([chunk])
    results = store.query("hello", top_k=1)
    assert results
    assert results[0].chunk.id == "c1"
