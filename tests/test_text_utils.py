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

    @pytest.mark.parametrize("text", ["", "   "])
    def test_empty_text(self, text):
        """Empty text should return empty set."""
        assert extract_keywords(text) == set()

    def test_basic_extraction(self):
        """Should extract words of minimum length."""
        result = extract_keywords("hello world test")
        assert "hello" in result
        assert "world" in result
        assert "test" in result

    @pytest.mark.parametrize(
        "text,min_len,expected_in,expected_not_in",
        [
            ("a an the hello", 3, ["the", "hello"], ["a", "an"]),
            ("hi there everyone", 4, ["there", "everyone"], ["hi"]),
            ("x y z word", 2, ["word"], ["x", "y", "z"]),
        ],
    )
    def test_respects_min_length(self, text, min_len, expected_in, expected_not_in):
        """Should filter words shorter than min_len."""
        result = extract_keywords(text, min_len=min_len)
        for word in expected_in:
            assert word in result
        for word in expected_not_in:
            assert word not in result

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

    @pytest.mark.parametrize(
        "text,stopwords,expected_in,expected_not_in",
        [
            ("the quick brown fox", {"the"}, ["quick", "brown", "fox"], ["the"]),
            ("is a test example", {"is", "a"}, ["test", "example"], ["is", "a"]),
        ],
    )
    def test_filters_stopwords(self, text, stopwords, expected_in, expected_not_in):
        """Should filter provided stopwords."""
        result = extract_keywords(text, stopwords=stopwords)
        for word in expected_in:
            assert word in result
        for word in expected_not_in:
            assert word not in result

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

    @pytest.mark.parametrize(
        "query,expected_sensitive,expected_keywords",
        [
            ("upcoming anime releases", True, ["upcoming"]),
            ("best movies 2025", True, ["2025"]),
            ("current NBA standings", True, ["current"]),
            ("latest news today", True, ["latest", "today"]),
            ("what is machine learning", False, []),
            ("explain quantum physics", False, []),
        ],
    )
    def test_time_sensitive_detection(
        self, query, expected_sensitive, expected_keywords
    ):
        """Should detect temporal keywords in queries."""
        is_sensitive, matched = detect_time_sensitive(query)
        assert is_sensitive is expected_sensitive
        for kw in expected_keywords:
            assert kw in matched

    def test_returns_all_matches(self):
        """Should return all matched temporal keywords."""
        is_sensitive, matched = detect_time_sensitive("new releases this week 2025")
        assert is_sensitive is True
        assert "new" in matched
        assert "releases" in matched
        assert "this week" in matched
        assert "2025" in matched

    @pytest.mark.parametrize(
        "query",
        ["UPCOMING RELEASES", "Latest NEWS", "CURRENT standings"],
    )
    def test_case_insensitive(self, query):
        """Should be case insensitive."""
        is_sensitive, matched = detect_time_sensitive(query)
        assert is_sensitive is True
        assert len(matched) > 0


class TestRequiresStructuredList:
    """Tests for requires_structured_list function."""

    @pytest.mark.parametrize(
        "query,expected",
        [
            # List queries
            ("list of best anime", True),
            ("top 10 movies", True),
            ("ranking of teams", True),
            # Schedule queries
            ("NBA schedule today", True),
            ("release schedule", True),
            ("upcoming releases", True),
            # Standings queries
            ("current standings", True),
            ("league table", True),
            # Seasonal queries
            ("fall anime lineup", True),
            ("spring releases", True),
            ("summer movies", True),
            # Non-structured queries
            ("what is Python", False),
            ("explain machine learning", False),
            ("how does blockchain work", False),
        ],
    )
    def test_structured_list_detection(self, query, expected):
        """Should detect queries expecting structured lists."""
        assert requires_structured_list(query) is expected

    @pytest.mark.parametrize(
        "query",
        ["TOP movies", "UPCOMING RELEASES", "List OF items"],
    )
    def test_case_insensitive(self, query):
        """Should be case insensitive."""
        assert requires_structured_list(query) is True


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
