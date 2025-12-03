from api.state import Document, SearchResult
from tools import fetcher


def test_fetch_documents_parallel(monkeypatch):
    calls = []

    def fake_fetch_url(result, timeout=15, run_id=None):
        calls.append(result.url)
        return Document(url=result.url, title=result.title, content="ok"), None

    monkeypatch.setattr(fetcher, "fetch_url", fake_fetch_url)

    results = [
        SearchResult(url=f"https://example.com/{idx}", title=f"Source {idx}")
        for idx in range(3)
    ]

    documents, warnings = fetcher.fetch_documents_parallel(results, max_workers=2)

    assert len(documents) == 3
    assert not warnings
    assert set(calls) == {r.url for r in results}
