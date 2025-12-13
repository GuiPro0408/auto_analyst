"""Tests for tools/topic_utils.py."""

import pytest

from tools.topic_utils import TOPIC_KEYWORDS, detect_query_topic


class TestDetectQueryTopic:
    """Tests for detect_query_topic function."""

    @pytest.mark.parametrize(
        "query,expected_topic",
        [
            # Entertainment topic tests
            ("best anime of 2024", "entertainment"),
            ("upcoming movie releases", "entertainment"),
            ("best video game releases", "entertainment"),
            ("new album by Taylor Swift", "entertainment"),
            # Technology topic tests
            ("how to learn python programming", "technology"),
            ("best software for data analysis", "technology"),
            ("latest machine learning frameworks", "technology"),
            ("cybersecurity best practices", "technology"),
            # News topic tests
            ("latest breaking news today", "news"),
            ("headline news today", "news"),
            # Science topic tests
            ("scientific research on vaccines", "science"),
            ("new treatment for disease", "science"),
            ("biology of cell division", "science"),
            # Sports topic tests
            ("premier league table standings", "sports"),
            ("NBA playoff scores", "sports"),
            ("FIFA world cup fixtures", "sports"),
        ],
    )
    def test_detects_topic(self, query, expected_topic):
        """Should detect correct topic for query."""
        assert detect_query_topic(query) == expected_topic

    @pytest.mark.parametrize(
        "query",
        [
            "what is the meaning of life",
            "",
            "the quick brown fox",
        ],
    )
    def test_returns_none_for_generic_query(self, query):
        """Should return None for queries with no clear topic."""
        result = detect_query_topic(query)
        assert result is None

    @pytest.mark.parametrize(
        "query,expected_topic",
        [
            ("PYTHON programming", "technology"),
            ("ANIME recommendations", "entertainment"),
            ("NBA SCORES", "sports"),
        ],
    )
    def test_case_insensitive(self, query, expected_topic):
        """Topic detection should be case insensitive."""
        assert detect_query_topic(query) == expected_topic

    def test_partial_word_matching(self):
        """Should handle partial keyword matches in longer strings."""
        result = detect_query_topic("advanced programming techniques")
        assert result == "technology"

    def test_multiple_topic_keywords(self):
        """When multiple topics match, should return highest overlap."""
        query = "programming a game engine for action games"
        result = detect_query_topic(query)
        assert result in ["technology", "entertainment"]


class TestTopicKeywords:
    """Tests for TOPIC_KEYWORDS structure."""

    def test_all_topics_have_keywords(self):
        """Each topic should have a non-empty set of keywords."""
        for topic, keywords in TOPIC_KEYWORDS.items():
            assert isinstance(keywords, set)
            assert len(keywords) > 0, f"Topic {topic} has no keywords"

    @pytest.mark.parametrize(
        "topic",
        ["entertainment", "technology", "news", "science", "sports"],
    )
    def test_expected_topics_exist(self, topic):
        """Expected topic categories should exist."""
        assert topic in TOPIC_KEYWORDS

    def test_keywords_are_lowercase(self):
        """All keywords should be lowercase for case-insensitive matching."""
        for topic, keywords in TOPIC_KEYWORDS.items():
            for kw in keywords:
                assert kw == kw.lower(), f"Keyword '{kw}' in {topic} is not lowercase"


class TestTopicDetectionIntegration:
    """Integration tests for topic detection in realistic scenarios."""

    @pytest.mark.parametrize(
        "query",
        [
            "what anime should I watch this season",
            "when does the new season of my show premiere",
            "best horror movies of 2024",
        ],
    )
    def test_real_world_entertainment_queries(self, query):
        """Test detection with realistic entertainment queries."""
        result = detect_query_topic(query)
        assert result == "entertainment", f"Failed for: {query}"

    @pytest.mark.parametrize(
        "query",
        [
            "how to connect to a database in python",
            "best cloud hosting providers",
            "javascript framework comparison",
        ],
    )
    def test_real_world_tech_queries(self, query):
        """Test detection with realistic technology queries."""
        result = detect_query_topic(query)
        assert result == "technology", f"Failed for: {query}"

    @pytest.mark.parametrize(
        "query",
        [
            "champions league fixtures this week",
            "NFL standings after week 10",
            "premier league results today",
        ],
    )
    def test_real_world_sports_queries(self, query):
        """Test detection with realistic sports queries."""
        result = detect_query_topic(query)
        assert result == "sports", f"Failed for: {query}"
