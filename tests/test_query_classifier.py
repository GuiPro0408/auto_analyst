"""Tests for query classification."""

import pytest

from tools.query_classifier import (
    classify_query,
    QUERY_TYPE_FACTUAL,
    QUERY_TYPE_RECOMMENDATION,
    QUERY_TYPE_CREATIVE,
    get_query_type_description,
)


class TestClassifyQuery:
    """Test query classification into different types."""

    # =============================================================================
    # RECOMMENDATION QUERIES
    # =============================================================================

    @pytest.mark.parametrize(
        "query",
        [
            "What anime should I watch?",
            "Recommend me some good horror movies",
            "Can you suggest some books like Harry Potter?",
            "Best anime of 2024",
            "Top 10 anime series to watch",
            "Anime similar to Attack on Titan",
            "What games should I play?",
            "Good shows like Breaking Bad",
            "Alternatives to Netflix",
            "What are some good programming podcasts?",
            "Is this anime worth watching?",
            "Your favorite anime?",
        ],
    )
    def test_classifies_recommendation_queries(self, query):
        """Should classify recommendation queries correctly."""
        result = classify_query(query)
        assert (
            result == QUERY_TYPE_RECOMMENDATION
        ), f"'{query}' should be recommendation"

    # =============================================================================
    # FACTUAL QUERIES
    # =============================================================================

    @pytest.mark.parametrize(
        "query",
        [
            "What happened in the 2024 election?",
            "When was Python released?",
            "Who won the World Cup 2022?",
            "How much does a Tesla Model 3 cost?",
            "Tesla Q3 2024 earnings report",
            "What is machine learning?",
            "Explain quantum computing",
            "What is the release date of the next iPhone?",
            "Python list vs tuple difference",
            "How does the stock market work?",
            "Statistics on climate change",
            "Is it true that the earth is round?",
            "Attack on Titan season 4 release date",
            "Who created Naruto?",
        ],
    )
    def test_classifies_factual_queries(self, query):
        """Should classify factual queries correctly."""
        result = classify_query(query)
        assert result == QUERY_TYPE_FACTUAL, f"'{query}' should be factual"

    # =============================================================================
    # CREATIVE QUERIES
    # =============================================================================

    @pytest.mark.parametrize(
        "query",
        [
            "Write me a poem about love",
            "Create a story about a dragon",
            "Generate some business name ideas",
            "Brainstorm topics for my blog",
            "Help me think of gift ideas",
            "Ideas for a birthday party",
        ],
    )
    def test_classifies_creative_queries(self, query):
        """Should classify creative queries correctly."""
        result = classify_query(query)
        assert result == QUERY_TYPE_CREATIVE, f"'{query}' should be creative"

    # =============================================================================
    # EDGE CASES
    # =============================================================================

    def test_defaults_to_factual(self):
        """Should default to factual for ambiguous queries."""
        result = classify_query("Python programming")
        assert result == QUERY_TYPE_FACTUAL

    def test_empty_query(self):
        """Should handle empty queries."""
        result = classify_query("")
        assert result == QUERY_TYPE_FACTUAL

    def test_case_insensitive(self):
        """Should be case insensitive."""
        result1 = classify_query("RECOMMEND me anime")
        result2 = classify_query("recommend me ANIME")
        assert result1 == QUERY_TYPE_RECOMMENDATION
        assert result2 == QUERY_TYPE_RECOMMENDATION


class TestQueryTypeDescription:
    """Test query type description helper."""

    def test_factual_description(self):
        desc = get_query_type_description(QUERY_TYPE_FACTUAL)
        assert "Research mode" in desc
        assert "citations" in desc

    def test_recommendation_description(self):
        desc = get_query_type_description(QUERY_TYPE_RECOMMENDATION)
        assert "Recommendation mode" in desc
        assert "suggestions" in desc

    def test_creative_description(self):
        desc = get_query_type_description(QUERY_TYPE_CREATIVE)
        assert "Creative mode" in desc

    def test_unknown_type(self):
        desc = get_query_type_description("unknown")
        assert "Unknown" in desc


class TestClassifierIntegration:
    """Integration tests for the classifier."""

    def test_anime_recommendation_flow(self):
        """Real-world anime query should be classified as recommendation."""
        queries = [
            "What anime should I watch this season?",
            "Best isekai anime recommendations",
            "Anime like Demon Slayer",
            "Good anime for beginners",
        ]
        for query in queries:
            result = classify_query(query)
            assert result == QUERY_TYPE_RECOMMENDATION, f"'{query}' failed"

    def test_anime_factual_flow(self):
        """Anime info queries should be factual."""
        queries = [
            "Attack on Titan final season air date",
            "Who is the author of One Piece?",
            "When does Jujutsu Kaisen season 3 come out?",
            "What is the plot of Naruto?",
        ]
        for query in queries:
            result = classify_query(query)
            assert result == QUERY_TYPE_FACTUAL, f"'{query}' failed"
