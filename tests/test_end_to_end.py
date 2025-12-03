from api.graph import run_research
from api.state import Document, SearchResult
from vector_store.base import ScoredChunk, VectorStore


class FakeLLM:
    def __call__(self, prompt):
        if "Verified answer:" in prompt:
            return [{"generated_text": "Verified answer: validated [1]"}]
        return [{"generated_text": "Answer: drafted [1]"}]


class FakeVectorStore(VectorStore):
    def __init__(self):
        self.chunks = []

    def clear(self):
        self.chunks = []

    def upsert(self, chunks):
        self.chunks.extend(chunks)

    def query(self, text, top_k: int = 5):
        return [ScoredChunk(chunk=c, score=1.0) for c in self.chunks[:top_k]]


def test_end_to_end_pipeline(monkeypatch):
    def fake_search(tasks, max_results=5, searx_host=None, run_id=None):
        return (
            [
                SearchResult(
                    url="http://example.com",
                    title="Example",
                    snippet="example snippet",
                    source="duckduckgo",
                )
            ],
            [],
        )

    def fake_fetch(result: SearchResult, run_id=None):
        return (
            Document(
                url=result.url,
                title=result.title,
                content="context for testing",
                media_type="html",
            ),
            None,
        )

    monkeypatch.setattr("api.graph.run_search_tasks", fake_search)
    monkeypatch.setattr("api.graph.fetch_url", fake_fetch)

    fake_llm = FakeLLM()
    fake_store = FakeVectorStore()
    result = run_research(
        "Test question",
        llm=fake_llm,
        vector_store=fake_store,
        embed_model="fake",
        top_k=1,
    )

    assert result.plan
    assert result.search_results
    assert result.documents
    assert result.retrieved
    assert result.citations
    assert result.verified_answer.startswith("validated")
    assert result.conversation_history
    assert result.conversation_history[-1].query == "Test question"
