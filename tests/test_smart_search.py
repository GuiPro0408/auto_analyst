"""Tests for smart search module.

Tests LLM-powered query analysis, result validation, and the
end-to-end smart search pipeline.
"""

import pytest
from unittest.mock import patch

from api.state import SearchResult
from tools.smart_search import (
    analyze_query_with_llm,
    smart_search,
    validate_results_with_llm,
)
from tests.conftest import ConfigurableLLM


class FakeSmartSearchLLM:
    """Fake LLM for smart search tests with configurable response."""

    def __init__(self, response: str):
        self.response = response

    def __call__(self, _prompt: str):
        return [{"generated_text": self.response}]


@pytest.mark.unit
def test_analyze_query_with_llm_valid_json():
    """Query analysis should parse valid JSON responses."""
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

    with patch(
        "tools.smart_search.load_llm", return_value=FakeSmartSearchLLM(fake_response)
    ):
        result = analyze_query_with_llm("latest openai news")

    assert result["topic"] == "technology"
    assert "openai" in result["entities"]
    assert len(result["suggested_searches"]) > 0


@pytest.mark.unit
def test_analyze_query_with_llm_invalid_json():
    """Query analysis should fallback gracefully on invalid JSON."""
    with patch(
        "tools.smart_search.load_llm", return_value=FakeSmartSearchLLM("invalid json")
    ):
        result = analyze_query_with_llm("test query")

    assert result["intent"] == "general"
    assert result["topic"] == "general"


@pytest.mark.unit
def test_validate_results_filters_irrelevant():
    """Result validation should filter out irrelevant results."""
    results = [
        SearchResult(
            url="https://techcrunch.com/openai", title="OpenAI News", snippet="..."
        ),
        SearchResult(
            url="https://random.de/accreditation", title="German Body", snippet="..."
        ),
        SearchResult(url="https://theverge.com/ai", title="AI Updates", snippet="..."),
    ]

    with patch(
        "tools.smart_search.load_llm", return_value=FakeSmartSearchLLM("[1, 3]")
    ):
        validated = validate_results_with_llm("openai news", results)

    assert len(validated) == 2
    assert validated[0].url == "https://techcrunch.com/openai"
    assert validated[1].url == "https://theverge.com/ai"


@pytest.mark.unit
def test_smart_search_end_to_end():
    """Smart search should integrate analysis, search, and validation."""
    analysis_response = (
        "{"
        '"intent": "news", "entities": ["test"], "topic": "technology", '
        '"time_sensitivity": "any", "suggested_searches": '
        '[{"query": "test", "rationale": "test"}], "authoritative_sources": []'
        "}"
    )

    fake_results = [
        SearchResult(url="https://example.com", title="Test", snippet="Test content"),
    ]

    # Create a mock TavilyBackend that returns fake results
    class MockTavilyBackend:
        def search(self, **kwargs):
            return (fake_results, [])

    with patch(
        "tools.smart_search.load_llm",
        return_value=FakeSmartSearchLLM(analysis_response),
    ):
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


@pytest.mark.unit
def test_analyze_query_with_llm_extracts_entities():
    """Query analysis should extract named entities."""
    response = """{
        "intent": "factual",
        "entities": ["Python", "machine learning", "TensorFlow"],
        "topic": "technology",
        "time_sensitivity": "any",
        "suggested_searches": [],
        "authoritative_sources": []
    }"""

    with patch(
        "tools.smart_search.load_llm", return_value=FakeSmartSearchLLM(response)
    ):
        result = analyze_query_with_llm(
            "how to use TensorFlow for machine learning in Python"
        )

    assert "Python" in result["entities"]
    assert "TensorFlow" in result["entities"]


@pytest.mark.unit
def test_validate_results_preserves_all_when_relevant():
    """Validation should keep all results when all are relevant."""
    results = [
        SearchResult(url="https://site1.com", title="Relevant 1", snippet="..."),
        SearchResult(url="https://site2.com", title="Relevant 2", snippet="..."),
    ]

    with patch(
        "tools.smart_search.load_llm", return_value=FakeSmartSearchLLM("[1, 2]")
    ):
        validated = validate_results_with_llm("test query", results)

    assert len(validated) == 2


@pytest.mark.unit
def test_validate_results_handles_empty_list():
    """Validation should handle empty result list."""
    with patch("tools.smart_search.load_llm", return_value=FakeSmartSearchLLM("[]")):
        validated = validate_results_with_llm("test query", [])

    assert validated == []
