from api.graph import run_research
from api.state import Chunk, SearchResult
from vector_store.base import ScoredChunk, VectorStore


class FakeLLM:
    def __call__(self, prompt):
        # For verify stage
        if "Verified answer:" in prompt:
            return [{"generated_text": "Verified answer: verified"}]
        return [{"generated_text": "Answer: drafted"}]


class FakeVectorStore(VectorStore):
    def __init__(self):
        self.chunks = []

    def clear(self):
        self.chunks = []

    def upsert(self, chunks):
        self.chunks.extend(chunks)

    def query(self, text, top_k: int = 5):
        return [ScoredChunk(chunk=c, score=1.0) for c in self.chunks[:top_k]]


def test_grounded_answer_fastpath(monkeypatch):
    grounded_answer = "Grounded response body"

    grounded_result = SearchResult(
        url="http://example.com",
        title="Example",
        snippet="snippet",
        source="gemini_grounding",
        content=grounded_answer,
    )

    def fake_search(tasks, max_results=5, run_id=None):
        return ([grounded_result], [])

    def fake_smart_search(query, max_results=5, run_id=None):
        return ([grounded_result], [])

    def fake_fetch(result: SearchResult, run_id=None):
        # Should not be required for grounded path, but provide a safe fallback
        return None, None

    monkeypatch.setattr("api.graph.run_search_tasks", fake_search)
    monkeypatch.setattr("api.graph.smart_search", fake_smart_search)
    monkeypatch.setattr("api.graph.fetch_url", fake_fetch)

    result = run_research(
        "Test grounded question",
        llm=FakeLLM(),
        vector_store=FakeVectorStore(),
        embed_model="fake",
        top_k=1,
    )
    assert result.search_results, "expected search results"
    assert result.search_results[0].source == "gemini_grounding"
    assert result.search_results[0].content == grounded_answer
    assert result.draft_answer.startswith("Grounded response"), result.draft_answer
    assert result.citations
    assert result.retrieved
    assert result.verified_answer.startswith("Verified answer")
