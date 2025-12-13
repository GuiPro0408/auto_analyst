"""Tests for document fetching.

Tests parallel document fetching, error handling, and timeout behavior.
"""

import pytest

from api.state import Document, SearchResult
from tools import fetcher


@pytest.mark.unit
def test_fetch_documents_parallel(monkeypatch):
    """Parallel fetcher should fetch all documents."""
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


@pytest.mark.unit
def test_fetch_documents_parallel_handles_failures(monkeypatch):
    """Parallel fetcher should handle failed fetches gracefully."""

    def fake_fetch_url(result, timeout=15, run_id=None):
        if "fail" in result.url:
            return None, f"Failed to fetch {result.url}"
        return Document(url=result.url, title=result.title, content="ok"), None

    monkeypatch.setattr(fetcher, "fetch_url", fake_fetch_url)

    results = [
        SearchResult(url="https://example.com/good1", title="Good 1"),
        SearchResult(url="https://fail.com/bad", title="Bad"),
        SearchResult(url="https://example.com/good2", title="Good 2"),
    ]

    documents, warnings = fetcher.fetch_documents_parallel(results, max_workers=2)

    assert len(documents) == 2
    assert len(warnings) == 1
    assert "fail.com" in warnings[0]


@pytest.mark.unit
def test_fetch_documents_parallel_empty_input(monkeypatch):
    """Parallel fetcher should handle empty input."""

    def fake_fetch_url(result, timeout=15, run_id=None):
        return Document(url=result.url, title=result.title, content="ok"), None

    monkeypatch.setattr(fetcher, "fetch_url", fake_fetch_url)

    documents, warnings = fetcher.fetch_documents_parallel([], max_workers=2)

    assert documents == []
    assert warnings == []


@pytest.mark.unit
def test_fetch_documents_parallel_all_fail(monkeypatch):
    """Parallel fetcher should handle all failures."""

    def fake_fetch_url(result, timeout=15, run_id=None):
        return None, f"Failed: {result.url}"

    monkeypatch.setattr(fetcher, "fetch_url", fake_fetch_url)

    results = [
        SearchResult(url=f"https://example.com/{idx}", title=f"Source {idx}")
        for idx in range(3)
    ]

    documents, warnings = fetcher.fetch_documents_parallel(results, max_workers=2)

    assert len(documents) == 0
    assert len(warnings) == 3


@pytest.mark.unit
def test_fetch_documents_parallel_returns_all_documents(monkeypatch):
    """Parallel fetcher should return all documents (order may vary due to as_completed)."""

    def fake_fetch_url(result, timeout=15, run_id=None):
        return Document(url=result.url, title=result.title, content="ok"), None

    monkeypatch.setattr(fetcher, "fetch_url", fake_fetch_url)

    results = [
        SearchResult(url=f"https://example.com/{idx}", title=f"Source {idx}")
        for idx in range(5)
    ]

    documents, _ = fetcher.fetch_documents_parallel(results, max_workers=2)

    # All documents should be returned (order may vary with parallel execution)
    fetched_urls = {d.url for d in documents}
    expected_urls = {r.url for r in results}
    assert fetched_urls == expected_urls
    assert len(documents) == len(results)


@pytest.mark.unit
@pytest.mark.parametrize("max_workers", [1, 2, 4, 8])
def test_fetch_documents_parallel_various_worker_counts(monkeypatch, max_workers):
    """Parallel fetcher should work with various worker counts."""

    def fake_fetch_url(result, timeout=15, run_id=None):
        return Document(url=result.url, title=result.title, content="ok"), None

    monkeypatch.setattr(fetcher, "fetch_url", fake_fetch_url)

    results = [
        SearchResult(url=f"https://example.com/{idx}", title=f"Source {idx}")
        for idx in range(10)
    ]

    documents, warnings = fetcher.fetch_documents_parallel(
        results, max_workers=max_workers
    )

    assert len(documents) == 10
    assert warnings == []


@pytest.mark.unit
def test_fetch_documents_parallel_with_run_id(monkeypatch):
    """Parallel fetcher should pass run_id to fetch function."""
    captured_run_ids = []

    def fake_fetch_url(result, timeout=15, run_id=None):
        captured_run_ids.append(run_id)
        return Document(url=result.url, title=result.title, content="ok"), None

    monkeypatch.setattr(fetcher, "fetch_url", fake_fetch_url)

    results = [SearchResult(url="https://example.com/1", title="Test")]
    fetcher.fetch_documents_parallel(results, max_workers=1, run_id="test-run-123")

    assert all(rid == "test-run-123" for rid in captured_run_ids)


@pytest.mark.unit
def test_fetch_documents_parallel_mixed_results(monkeypatch):
    """Parallel fetcher should handle mixed success/failure results."""

    def fake_fetch_url(result, timeout=15, run_id=None):
        idx = int(result.url.split("/")[-1])
        if idx % 2 == 0:
            return Document(url=result.url, title=result.title, content="ok"), None
        return None, f"Failed odd URL: {result.url}"

    monkeypatch.setattr(fetcher, "fetch_url", fake_fetch_url)

    results = [
        SearchResult(url=f"https://example.com/{idx}", title=f"Source {idx}")
        for idx in range(6)
    ]

    documents, warnings = fetcher.fetch_documents_parallel(results, max_workers=2)

    # Even indices (0, 2, 4) should succeed
    assert len(documents) == 3
    # Odd indices (1, 3, 5) should fail
    assert len(warnings) == 3
