"""Tests for tools/topic_utils.py."""

import pytest

from tools.topic_utils import TOPIC_KEYWORDS, detect_query_topic


class TestDetectQueryTopic:
    """Tests for detect_query_topic function."""

    # Entertainment topic tests
    def test_detects_anime_topic(self):
        """Should detect entertainment for anime-related queries."""
        assert detect_query_topic("best anime of 2024") == "entertainment"

    def test_detects_movie_topic(self):
        """Should detect entertainment for movie-related queries."""
        assert detect_query_topic("upcoming movie releases") == "entertainment"

    def test_detects_gaming_topic(self):
        """Should detect entertainment for gaming queries."""
        assert detect_query_topic("best video game releases") == "entertainment"

    def test_detects_music_topic(self):
        """Should detect entertainment for music queries."""
        assert detect_query_topic("new album by Taylor Swift") == "entertainment"

    # Technology topic tests
    def test_detects_programming_topic(self):
        """Should detect technology for programming queries."""
        assert detect_query_topic("how to learn python programming") == "technology"

    def test_detects_software_topic(self):
        """Should detect technology for software queries."""
        assert detect_query_topic("best software for data analysis") == "technology"

    def test_detects_ai_topic(self):
        """Should detect technology for AI queries."""
        assert detect_query_topic("latest machine learning frameworks") == "technology"

    def test_detects_cybersecurity_topic(self):
        """Should detect technology for cybersecurity queries."""
        assert detect_query_topic("cybersecurity best practices") == "technology"

    # News topic tests
    def test_detects_news_topic(self):
        """Should detect news for news-related queries."""
        assert detect_query_topic("latest breaking news today") == "news"

    def test_detects_headlines_topic(self):
        """Should detect news for headline queries."""
        # Note: "headlines" (plural) is not a keyword, but "headline" (singular) is
        assert detect_query_topic("headline news today") == "news"

    # Science topic tests
    def test_detects_research_topic(self):
        """Should detect science for research queries."""
        assert detect_query_topic("scientific research on vaccines") == "science"

    def test_detects_medicine_topic(self):
        """Should detect science for medical queries."""
        assert detect_query_topic("new treatment for disease") == "science"

    def test_detects_biology_topic(self):
        """Should detect science for biology queries."""
        assert detect_query_topic("biology of cell division") == "science"

    # Sports topic tests
    def test_detects_football_topic(self):
        """Should detect sports for football queries."""
        assert detect_query_topic("premier league table standings") == "sports"

    def test_detects_nba_topic(self):
        """Should detect sports for NBA queries."""
        # Use "nba" without "today" to avoid ambiguity with news keywords
        assert detect_query_topic("NBA playoff scores") == "sports"

    def test_detects_soccer_topic(self):
        """Should detect sports for soccer queries."""
        assert detect_query_topic("FIFA world cup fixtures") == "sports"

    # Edge cases
    def test_returns_none_for_generic_query(self):
        """Should return None for queries with no clear topic."""
        result = detect_query_topic("what is the meaning of life")
        assert result is None

    def test_returns_none_for_empty_query(self):
        """Should return None for empty query."""
        result = detect_query_topic("")
        assert result is None

    def test_case_insensitive(self):
        """Topic detection should be case insensitive."""
        assert detect_query_topic("PYTHON programming") == "technology"
        assert detect_query_topic("ANIME recommendations") == "entertainment"

    def test_partial_word_matching(self):
        """Should handle partial keyword matches in longer strings."""
        # 'programming' contains 'program' but we check for full words + substring
        result = detect_query_topic("advanced programming techniques")
        assert result == "technology"

    def test_multiple_topic_keywords(self):
        """When multiple topics match, should return highest overlap."""
        # This query has both entertainment and technology keywords
        query = "programming a game engine for action games"
        result = detect_query_topic(query)
        # Should pick one based on overlap score
        assert result in ["technology", "entertainment"]

    def test_minimum_overlap_threshold(self):
        """Should require minimum overlap to detect topic."""
        # Single short keyword shouldn't trigger detection
        result = detect_query_topic("the quick brown fox")
        assert result is None


class TestTopicKeywords:
    """Tests for TOPIC_KEYWORDS structure."""

    def test_all_topics_have_keywords(self):
        """Each topic should have a non-empty set of keywords."""
        for topic, keywords in TOPIC_KEYWORDS.items():
            assert isinstance(keywords, set)
            assert len(keywords) > 0, f"Topic {topic} has no keywords"

    def test_expected_topics_exist(self):
        """Expected topic categories should exist."""
        expected_topics = ["entertainment", "technology", "news", "science", "sports"]
        for topic in expected_topics:
            assert topic in TOPIC_KEYWORDS

    def test_keywords_are_lowercase(self):
        """All keywords should be lowercase for case-insensitive matching."""
        for topic, keywords in TOPIC_KEYWORDS.items():
            for kw in keywords:
                assert kw == kw.lower(), f"Keyword '{kw}' in {topic} is not lowercase"


class TestTopicDetectionIntegration:
    """Integration tests for topic detection in realistic scenarios."""

    def test_real_world_entertainment_queries(self):
        """Test detection with realistic entertainment queries."""
        queries = [
            "what anime should I watch this season",
            "when does the new season of my show premiere",
            "best horror movies of 2024",
        ]
        for query in queries:
            result = detect_query_topic(query)
            assert result == "entertainment", f"Failed for: {query}"

    def test_real_world_tech_queries(self):
        """Test detection with realistic technology queries."""
        queries = [
            "how to connect to a database in python",
            "best cloud hosting providers",
            "javascript framework comparison",
        ]
        for query in queries:
            result = detect_query_topic(query)
            assert result == "technology", f"Failed for: {query}"

    def test_real_world_sports_queries(self):
        """Test detection with realistic sports queries."""
        queries = [
            "champions league fixtures this week",
            "NFL standings after week 10",
            "premier league results today",
        ]
        for query in queries:
            result = detect_query_topic(query)
            assert result == "sports", f"Failed for: {query}"
