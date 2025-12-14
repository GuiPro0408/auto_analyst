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


class TestContentLengthFilter:
    """Tests for minimum content length filtering."""

    @pytest.mark.unit
    def test_fetch_filters_short_content(self, monkeypatch):
        """Should reject pages with content shorter than MIN_CONTENT_LENGTH."""
        import requests

        class FakeResponse:
            status_code = 200
            ok = True
            content = b"Short"
            text = "<html><body>Short</body></html>"
            headers = {"content-type": "text/html"}

            def raise_for_status(self):
                pass

        def fake_get(*args, **kwargs):
            return FakeResponse()

        monkeypatch.setattr(requests, "get", fake_get)
        monkeypatch.setattr(fetcher, "is_allowed", lambda url, run_id=None: True)

        result = SearchResult(url="https://example.com/short", title="Short Page")
        doc, warning = fetcher.fetch_url(result)

        assert doc is None
        assert warning is not None
        assert "too short" in warning.lower()

    @pytest.mark.unit
    def test_fetch_accepts_long_content(self, monkeypatch):
        """Should accept pages with content >= MIN_CONTENT_LENGTH."""
        import requests

        long_content = "This is substantial content. " * 20  # ~600 chars

        class FakeResponse:
            status_code = 200
            ok = True
            content = long_content.encode()
            text = f"<html><body><main>{long_content}</main></body></html>"
            headers = {"content-type": "text/html"}

            def raise_for_status(self):
                pass

        def fake_get(*args, **kwargs):
            return FakeResponse()

        monkeypatch.setattr(requests, "get", fake_get)
        monkeypatch.setattr(fetcher, "is_allowed", lambda url, run_id=None: True)

        result = SearchResult(url="https://example.com/long", title="Long Page")
        doc, warning = fetcher.fetch_url(result)

        assert doc is not None
        assert warning is None
        assert len(doc.content) >= 200


class TestPrefetchedContent:
    """Tests for using pre-fetched content from search results."""

    @pytest.mark.unit
    def test_uses_prefetched_content_when_available(self, monkeypatch):
        """Should use pre-fetched content without making HTTP request."""
        fetch_calls = []

        def fake_fetch_url(result, timeout=15, run_id=None):
            fetch_calls.append(result.url)
            return Document(url=result.url, title=result.title, content="fetched"), None

        monkeypatch.setattr(fetcher, "fetch_url", fake_fetch_url)

        prefetched_content = (
            "This is pre-fetched content from Tavily that is long enough. " * 10
        )
        results = [
            SearchResult(
                url="https://crunchyroll.com/anime",
                title="Anime Site",
                content=prefetched_content,  # Pre-fetched content
                source="tavily",
            ),
        ]

        documents, warnings = fetcher.fetch_documents_parallel(results, max_workers=2)

        # Should have document without calling fetch_url
        assert len(documents) == 1
        assert len(fetch_calls) == 0  # No HTTP fetch made
        assert documents[0].content == prefetched_content
        assert documents[0].url == "https://crunchyroll.com/anime"

    @pytest.mark.unit
    def test_fetches_when_prefetched_content_too_short(self, monkeypatch):
        """Should fetch URL when pre-fetched content is below minimum length."""
        fetch_calls = []

        def fake_fetch_url(result, timeout=15, run_id=None):
            fetch_calls.append(result.url)
            return (
                Document(url=result.url, title=result.title, content="fetched content"),
                None,
            )

        monkeypatch.setattr(fetcher, "fetch_url", fake_fetch_url)

        results = [
            SearchResult(
                url="https://example.com/page",
                title="Example",
                content="Short",  # Too short
                source="tavily",
            ),
        ]

        documents, warnings = fetcher.fetch_documents_parallel(results, max_workers=2)

        # Should have called fetch_url since content was too short
        assert len(fetch_calls) == 1
        assert "example.com" in fetch_calls[0]

    @pytest.mark.unit
    def test_mixed_prefetched_and_regular_fetch(self, monkeypatch):
        """Should handle mix of pre-fetched and regular fetches."""
        fetch_calls = []

        def fake_fetch_url(result, timeout=15, run_id=None):
            fetch_calls.append(result.url)
            return Document(url=result.url, title=result.title, content="fetched"), None

        monkeypatch.setattr(fetcher, "fetch_url", fake_fetch_url)

        prefetched_content = "Long pre-fetched content. " * 20
        results = [
            SearchResult(
                url="https://site1.com",
                title="Site 1",
                content=prefetched_content,  # Has pre-fetched
                source="tavily",
            ),
            SearchResult(
                url="https://site2.com",
                title="Site 2",
                content="",  # No content, needs fetch
                source="gemini_grounding",
            ),
            SearchResult(
                url="https://site3.com",
                title="Site 3",
                content="Short",  # Too short, needs fetch
                source="tavily",
            ),
        ]

        documents, warnings = fetcher.fetch_documents_parallel(results, max_workers=2)

        assert len(documents) == 3
        # Only 2 fetches made (site2 and site3)
        assert len(fetch_calls) == 2
        assert "site1.com" not in str(fetch_calls)
        assert "site2.com" in str(fetch_calls)
        assert "site3.com" in str(fetch_calls)
