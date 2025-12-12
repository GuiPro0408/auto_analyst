from types import SimpleNamespace

from api.state import Chunk
from vector_store import chroma_store
from vector_store.chroma_store import ChromaVectorStore


class DummyEmbeddingFunction:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def __call__(self, texts):  # pragma: no cover - not used in fake
        return [[0.0] * len(texts)]


class FakeCollection:
    def __init__(self, name: str):
        self.name = name
        self.docs = []

    def upsert(self, ids, documents, metadatas):
        for doc_id, doc, meta in zip(ids, documents, metadatas):
            self.docs.append((doc_id, doc, meta))

    def query(self, query_texts, n_results, include, where=None):
        selected = self.docs[:n_results]
        return {
            "documents": [[doc for _, doc, _ in selected]],
            "metadatas": [[meta for _, _, meta in selected]],
            "distances": [[0.0 for _ in selected]],
        }


class FakeClient:
    def __init__(self):
        self.collections = {}
        self.deleted = []

    def get_or_create_collection(self, name, embedding_function=None, metadata=None):
        if name not in self.collections:
            self.collections[name] = FakeCollection(name)
        return self.collections[name]

    def delete_collection(self, name):
        self.deleted.append(name)
        self.collections.pop(name, None)


def test_chroma_clear_is_run_scoped(monkeypatch, tmp_path):
    fake_client = FakeClient()
    monkeypatch.setattr(
        chroma_store, "chromadb", SimpleNamespace(Client=lambda settings: fake_client)
    )
    monkeypatch.setattr(
        chroma_store, "Settings", lambda is_persistent, persist_directory: None
    )
    monkeypatch.setattr(
        chroma_store.embedding_functions,
        "SentenceTransformerEmbeddingFunction",
        lambda model_name: DummyEmbeddingFunction(model_name),
    )

    store_a = ChromaVectorStore(
        model_name="fake", persist_path=tmp_path, collection_name="coll", run_id="runA"
    )
    store_b = ChromaVectorStore(
        model_name="fake", persist_path=tmp_path, collection_name="coll", run_id="runB"
    )

    store_a.upsert([Chunk(id="a1", text="a", metadata={"title": "A"})])
    store_b.upsert([Chunk(id="b1", text="b", metadata={"title": "B"})])

    assert "coll-runA" in fake_client.collections
    assert "coll-runB" in fake_client.collections

    store_a.clear()

    # runB collection should remain intact
    assert "coll-runB" in fake_client.collections
    results_b = store_b.query("q", top_k=5)
    assert [sc.chunk.text for sc in results_b] == ["b"]

    # runA collection recreated empty after clear
    assert "coll-runA" in fake_client.collections
    results_a = store_a.query("q", top_k=5)
    assert results_a == []

    # Ensure delete_collection targeted only runA
    assert fake_client.deleted == ["coll-runA"]
