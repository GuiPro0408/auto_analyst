"""Tests for smart search module."""

from unittest.mock import patch

from api.state import SearchResult
from tools.smart_search import (
    analyze_query_with_llm,
    smart_search,
    validate_results_with_llm,
)


class FakeLLM:
    """Fake LLM returning a predefined response."""

    def __init__(self, response: str):
        self.response = response

    def __call__(self, _prompt: str):
        return [{"generated_text": self.response}]


def test_analyze_query_with_llm_valid_json():
    fake_response = """```json
    {
        "intent": "news",
        "entities": ["openai"],
        "topic": "technology",
        "time_sensitivity": "realtime",
        "suggested_searches": [
            {"query": "OpenAI news 2025", "rationale": "recent news"}
        ],
        "authoritative_sources": ["techcrunch.com"]
    }
    ```"""

    with patch("tools.smart_search.load_llm", return_value=FakeLLM(fake_response)):
        result = analyze_query_with_llm("latest openai news")

    assert result["topic"] == "technology"
    assert "openai" in result["entities"]
    assert len(result["suggested_searches"]) > 0


def test_analyze_query_with_llm_invalid_json():
    with patch("tools.smart_search.load_llm", return_value=FakeLLM("invalid json")):
        result = analyze_query_with_llm("test query")

    assert result["intent"] == "general"
    assert result["topic"] == "general"


def test_validate_results_filters_irrelevant():
    results = [
        SearchResult(url="https://techcrunch.com/openai", title="OpenAI News", snippet="..."),
        SearchResult(url="https://random.de/accreditation", title="German Body", snippet="..."),
        SearchResult(url="https://theverge.com/ai", title="AI Updates", snippet="..."),
    ]

    with patch("tools.smart_search.load_llm", return_value=FakeLLM("[1, 3]")):
        validated = validate_results_with_llm("openai news", results)

    assert len(validated) == 2
    assert validated[0].url == "https://techcrunch.com/openai"
    assert validated[1].url == "https://theverge.com/ai"


def test_smart_search_end_to_end():
    analysis_response = (
        "{"
        '"intent": "news", "entities": ["test"], "topic": "technology", '
        '"time_sensitivity": "any", "suggested_searches": '
        '[{"query": "test", "rationale": "test"}], "authoritative_sources": []'  # noqa: E501
        "}"
    )

    fake_results = [
        SearchResult(url="https://example.com", title="Test", snippet="Test content"),
    ]

    # Create a mock TavilyBackend that returns fake results
    class MockTavilyBackend:
        def search(self, **kwargs):
            return (fake_results, [])

    with patch("tools.smart_search.load_llm", return_value=FakeLLM(analysis_response)):
        # Patch TavilyBackend at the source (tools.search) since it's imported inside the function
        with patch("tools.search.TavilyBackend", MockTavilyBackend):
            with patch(
                "tools.smart_search.validate_results_with_llm",
                return_value=fake_results,
            ):
                results, warnings = smart_search("test query")

    assert len(results) == 1
    assert results[0].url == "https://example.com"
    assert warnings == []
