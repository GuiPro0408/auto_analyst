import types
from unittest.mock import MagicMock, patch

import requests

from tools import fetcher


class MockResponse:
    def __init__(self, status_code=200, text=""):
        self.status_code = status_code
        self.text = text


def test_is_allowed_denies_on_robots_failure(monkeypatch):
    """Test that fetch failure defaults to block when ROBOTS_ON_ERROR=block."""
    # Clear cache
    fetcher._robots_cache = {}

    # Mock requests.get to raise an exception
    def mock_get(*args, **kwargs):
        raise requests.RequestException("Connection failed")

    monkeypatch.setattr(requests, "get", mock_get)
    monkeypatch.setattr(fetcher, "ROBOTS_ON_ERROR", "block")

    allowed = fetcher.is_allowed("https://example.com/page")

    assert allowed is False
    # cached denial should be stored
    assert fetcher._robots_cache.get("example.com")[0] is False


def test_is_allowed_uses_cache(monkeypatch):
    """Test that robots.txt result is cached."""
    fetcher._robots_cache = {}
    call_count = 0

    def mock_get(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return MockResponse(200, "User-agent: *\nAllow: /\n")

    monkeypatch.setattr(requests, "get", mock_get)

    first = fetcher.is_allowed("https://example.com/page")
    second = fetcher.is_allowed("https://example.com/other")

    assert first is True
    assert second is True
    # requests.get called only once due to cache hit
    assert call_count == 1


def test_is_allowed_allows_on_404(monkeypatch):
    """Test that 404 response defaults to allowed."""
    fetcher._robots_cache = {}

    def mock_get(*args, **kwargs):
        return MockResponse(404, "")

    monkeypatch.setattr(requests, "get", mock_get)

    allowed = fetcher.is_allowed("https://example.com/page")

    assert allowed is True
    assert fetcher._robots_cache.get("example.com")[0] is True


def test_is_allowed_respects_disallow(monkeypatch):
    """Test that Disallow rules are respected."""
    fetcher._robots_cache = {}

    def mock_get(*args, **kwargs):
        return MockResponse(200, "User-agent: *\nDisallow: /\n")

    monkeypatch.setattr(requests, "get", mock_get)

    allowed = fetcher.is_allowed("https://example.com/page")

    assert allowed is False
    assert fetcher._robots_cache.get("example.com")[0] is False
