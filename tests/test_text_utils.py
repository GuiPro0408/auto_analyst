"""Tests for tools/text_utils.py keyword extraction and heuristics."""

import pytest

from tools.text_utils import (
    STOPWORDS,
    STRUCTURED_TRIGGERS,
    TEMPORAL_KEYWORDS,
    detect_time_sensitive,
    extract_keywords,
    requires_structured_list,
)


class TestExtractKeywords:
    """Tests for extract_keywords function."""

    def test_empty_text(self):
        """Empty text should return empty set."""
        assert extract_keywords("") == set()
        assert extract_keywords("   ") == set()

    def test_basic_extraction(self):
        """Should extract words of minimum length."""
        result = extract_keywords("hello world test")
        assert "hello" in result
        assert "world" in result
        assert "test" in result

    def test_respects_min_length(self):
        """Should filter words shorter than min_len."""
        result = extract_keywords("a an the hello", min_len=3)
        assert "a" not in result
        assert "an" not in result
        assert "the" in result
        assert "hello" in result

    def test_strips_punctuation(self):
        """Should strip punctuation when enabled."""
        result = extract_keywords("hello! world? test.", strip_punct=True)
        assert "hello" in result
        assert "world" in result
        assert "test" in result
        assert "hello!" not in result

    def test_preserves_punctuation_when_disabled(self):
        """Should preserve punctuation when disabled."""
        result = extract_keywords("hello! world?", strip_punct=False, min_len=1)
        # Punctuation is still stripped by regex
        assert "hello" in result
        assert "world" in result

    def test_filters_stopwords(self):
        """Should filter provided stopwords."""
        result = extract_keywords("the quick brown fox", stopwords={"the"})
        assert "the" not in result
        assert "quick" in result
        assert "brown" in result
        assert "fox" in result

    def test_lowercases_output(self):
        """Should return lowercase keywords."""
        result = extract_keywords("HELLO World TeSt")
        assert "hello" in result
        assert "world" in result
        assert "test" in result
        assert "HELLO" not in result

    def test_removes_special_characters(self):
        """Should remove non-alphanumeric characters."""
        result = extract_keywords("hello@world test#123")
        assert "helloworld" in result or "hello" in result
        assert "test123" in result or "test" in result


class TestDetectTimeSensitive:
    """Tests for detect_time_sensitive function."""

    def test_detects_temporal_keywords(self):
        """Should detect queries with temporal keywords."""
        is_sensitive, matched = detect_time_sensitive("upcoming anime releases")
        assert is_sensitive is True
        assert "upcoming" in matched

    def test_detects_year_keywords(self):
        """Should detect year keywords."""
        is_sensitive, matched = detect_time_sensitive("best movies 2025")
        assert is_sensitive is True
        assert "2025" in matched

    def test_detects_current_keywords(self):
        """Should detect 'current', 'latest', etc."""
        is_sensitive, matched = detect_time_sensitive("current NBA standings")
        assert is_sensitive is True
        assert "current" in matched

        is_sensitive, matched = detect_time_sensitive("latest news today")
        assert is_sensitive is True
        assert "latest" in matched
        assert "today" in matched

    def test_returns_all_matches(self):
        """Should return all matched temporal keywords."""
        is_sensitive, matched = detect_time_sensitive(
            "new releases this week 2025"
        )
        assert is_sensitive is True
        assert "new" in matched
        assert "releases" in matched
        assert "this week" in matched
        assert "2025" in matched

    def test_non_temporal_query(self):
        """Should return False for non-temporal queries."""
        is_sensitive, matched = detect_time_sensitive("what is machine learning")
        assert is_sensitive is False
        assert matched == []

    def test_case_insensitive(self):
        """Should be case insensitive."""
        is_sensitive, matched = detect_time_sensitive("UPCOMING RELEASES")
        assert is_sensitive is True
        assert "upcoming" in matched


class TestRequiresStructuredList:
    """Tests for requires_structured_list function."""

    def test_detects_list_queries(self):
        """Should detect queries expecting lists."""
        assert requires_structured_list("list of best anime") is True
        assert requires_structured_list("top 10 movies") is True
        assert requires_structured_list("ranking of teams") is True

    def test_detects_schedule_queries(self):
        """Should detect schedule-related queries."""
        assert requires_structured_list("NBA schedule today") is True
        assert requires_structured_list("release schedule") is True
        assert requires_structured_list("upcoming releases") is True

    def test_detects_standings_queries(self):
        """Should detect standings/table queries."""
        assert requires_structured_list("current standings") is True
        assert requires_structured_list("league table") is True

    def test_detects_season_queries(self):
        """Should detect seasonal queries."""
        assert requires_structured_list("fall anime lineup") is True
        assert requires_structured_list("spring releases") is True
        assert requires_structured_list("summer movies") is True

    def test_non_structured_queries(self):
        """Should return False for non-structured queries."""
        assert requires_structured_list("what is Python") is False
        assert requires_structured_list("explain machine learning") is False
        assert requires_structured_list("how does blockchain work") is False

    def test_case_insensitive(self):
        """Should be case insensitive."""
        assert requires_structured_list("TOP movies") is True
        assert requires_structured_list("UPCOMING RELEASES") is True


class TestConstants:
    """Tests for module constants."""

    def test_stopwords_is_set(self):
        """STOPWORDS should be a non-empty set."""
        assert isinstance(STOPWORDS, set)
        assert len(STOPWORDS) > 0
        assert "the" in STOPWORDS
        assert "is" in STOPWORDS
        assert "what" in STOPWORDS

    def test_temporal_keywords_is_set(self):
        """TEMPORAL_KEYWORDS should be a non-empty set."""
        assert isinstance(TEMPORAL_KEYWORDS, set)
        assert len(TEMPORAL_KEYWORDS) > 0
        assert "upcoming" in TEMPORAL_KEYWORDS
        assert "2025" in TEMPORAL_KEYWORDS

    def test_structured_triggers_is_set(self):
        """STRUCTURED_TRIGGERS should be a non-empty set."""
        assert isinstance(STRUCTURED_TRIGGERS, set)
        assert len(STRUCTURED_TRIGGERS) > 0
        assert "list" in STRUCTURED_TRIGGERS
        assert "top" in STRUCTURED_TRIGGERS
