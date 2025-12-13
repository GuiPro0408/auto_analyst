"""Tests for tools/search_filters.py filtering and deduplication."""

import pytest

from api.state import SearchResult
from tools.search_filters import (
    BLOCKED_DOMAINS,
    KNOWN_ROBOTS_BLOCKED_DOMAINS,
    META_CONTENT_DOMAINS,
    dedupe_results,
    filter_results,
)


class TestDedupeResults:
    """Tests for dedupe_results function."""

    def test_empty_results(self):
        """Empty results should return empty list."""
        assert dedupe_results([]) == []

    def test_removes_duplicates(self):
        """Should remove duplicate URLs."""
        results = [
            SearchResult(url="http://example.com/page1", title="Page 1", snippet=""),
            SearchResult(url="http://example.com/page1", title="Page 1", snippet=""),
            SearchResult(url="http://example.com/page2", title="Page 2", snippet=""),
        ]
        deduped = dedupe_results(results)
        assert len(deduped) == 2
        urls = [r.url for r in deduped]
        assert "http://example.com/page1" in urls
        assert "http://example.com/page2" in urls

    def test_strips_url_fragments(self):
        """Should treat URLs with same base but different fragments as duplicates."""
        results = [
            SearchResult(url="http://example.com/page#section1", title="P1", snippet=""),
            SearchResult(url="http://example.com/page#section2", title="P2", snippet=""),
        ]
        deduped = dedupe_results(results)
        assert len(deduped) == 1

    def test_skips_empty_urls(self):
        """Should skip results with empty URLs."""
        results = [
            SearchResult(url="", title="Empty", snippet=""),
            SearchResult(url="http://example.com", title="Valid", snippet=""),
        ]
        deduped = dedupe_results(results)
        assert len(deduped) == 1
        assert deduped[0].url == "http://example.com"


class TestFilterResults:
    """Tests for filter_results function."""

    def test_empty_results(self):
        """Empty results should return empty list."""
        assert filter_results("test query", []) == []

    def test_filters_blocked_domains(self):
        """Should filter out blocked domains."""
        results = [
            SearchResult(url="http://reddit.com/user/test", title="Reddit", snippet=""),
            SearchResult(url="http://example.com", title="Valid", snippet="test"),
        ]
        filtered = filter_results("test", results)
        assert len(filtered) == 1
        assert "reddit.com" not in filtered[0].url

    def test_filters_robots_blocked_domains(self):
        """Should filter out known robots-blocked domains."""
        results = [
            SearchResult(url="http://anime-planet.com/page", title="Anime", snippet="anime"),
            SearchResult(url="http://example.com", title="Valid", snippet="anime"),
        ]
        filtered = filter_results("anime", results)
        assert len(filtered) == 1
        assert "anime-planet.com" not in filtered[0].url

    def test_filters_meta_content_domains(self):
        """Should filter meta-content domains when query is not about language."""
        results = [
            SearchResult(
                url="http://english.stackexchange.com", title="English", snippet="word"
            ),
            SearchResult(url="http://example.com", title="Valid", snippet="anime"),
        ]
        filtered = filter_results("anime releases", results)
        assert len(filtered) == 1
        assert "stackexchange.com" not in filtered[0].url

    def test_allows_meta_domains_for_language_queries(self):
        """Should allow meta-content domains for language-related queries."""
        results = [
            SearchResult(
                url="http://english.stackexchange.com",
                title="English Grammar",
                snippet="grammar rules",
            ),
        ]
        filtered = filter_results("grammar rules definition", results)
        assert len(filtered) == 1

    def test_filters_irrelevant_results(self):
        """Should filter results with no keyword overlap."""
        results = [
            SearchResult(
                url="http://example.com/cats",
                title="Cat Videos",
                snippet="Cute cats playing",
            ),
            SearchResult(
                url="http://example.com/anime",
                title="Anime News",
                snippet="Latest anime releases",
            ),
        ]
        filtered = filter_results("anime releases 2025", results)
        assert len(filtered) == 1
        assert "anime" in filtered[0].url

    def test_preserves_gemini_grounding_results(self):
        """Should preserve Gemini grounding results regardless of overlap."""
        results = [
            SearchResult(
                url="http://example.com",
                title="Unrelated",
                snippet="No overlap keywords",
                source="gemini_grounding",
            ),
        ]
        filtered = filter_results("specific query terms", results)
        assert len(filtered) == 1

    def test_handles_empty_query_keywords(self):
        """Should include all non-blocked results when query has no meaningful keywords."""
        results = [
            SearchResult(url="http://example.com", title="Page", snippet="content"),
        ]
        filtered = filter_results("the is a", results)
        assert len(filtered) == 1


class TestBlockedDomains:
    """Tests for blocked domain constants."""

    def test_blocked_domains_is_set(self):
        """BLOCKED_DOMAINS should be a non-empty set."""
        assert isinstance(BLOCKED_DOMAINS, set)
        assert len(BLOCKED_DOMAINS) > 0

    def test_contains_social_media(self):
        """Should contain major social media domains."""
        assert "facebook.com" in BLOCKED_DOMAINS
        assert "twitter.com" in BLOCKED_DOMAINS
        assert "instagram.com" in BLOCKED_DOMAINS

    def test_robots_blocked_is_set(self):
        """KNOWN_ROBOTS_BLOCKED_DOMAINS should be a non-empty set."""
        assert isinstance(KNOWN_ROBOTS_BLOCKED_DOMAINS, set)
        assert len(KNOWN_ROBOTS_BLOCKED_DOMAINS) > 0

    def test_meta_content_domains_is_set(self):
        """META_CONTENT_DOMAINS should be a non-empty set."""
        assert isinstance(META_CONTENT_DOMAINS, set)
        assert len(META_CONTENT_DOMAINS) > 0
        assert "stackexchange.com" in META_CONTENT_DOMAINS
