"""Tests for refactored modules: key_rotator, cache_manager, state_builder."""

import os
import tempfile

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from api.key_rotator import APIKeyRotator, get_default_rotator, reset_default_rotator
from api.cache_manager import CacheManager
from api.state_builder import (
    normalize_conversation_history,
    extract_grounded_answer,
    build_research_state,
    create_initial_state,
)
from api.state import ConversationTurn, Chunk, SearchQuery, SearchResult


class TestAPIKeyRotator:
    """Tests for the shared API key rotator."""

    @pytest.mark.parametrize(
        "keys,expected_key",
        [
            (["key1"], "key1"),
            ([], None),
            (None, None),
        ],
    )
    def test_key_rotation_basic(self, keys, expected_key):
        """Key rotator should return expected key for various inputs."""
        rotator = APIKeyRotator(keys)
        assert rotator.current_key == expected_key

    def test_single_key_rotation(self):
        """Single key should always return the same key."""
        rotator = APIKeyRotator(["key1"])
        assert rotator.current_key == "key1"
        assert rotator.current_key == "key1"
        assert rotator.current_key == "key1"

    def test_rate_limited_key_skipped(self):
        """Rate-limited keys should be skipped."""
        rotator = APIKeyRotator(["key1", "key2", "key3"])
        rotator.mark_rate_limited("key1")
        assert rotator.current_key == "key2"

    @pytest.mark.parametrize(
        "keys,expected_total",
        [
            (["key1", "key2", "key3"], 3),
            (["key1"], 1),
            ([], 0),
        ],
    )
    def test_total_keys(self, keys, expected_total):
        """total_keys should return number of configured keys."""
        rotator = APIKeyRotator(keys)
        assert rotator.total_keys == expected_total

    def test_available_keys(self):
        """available_keys should reflect non-rate-limited keys."""
        rotator = APIKeyRotator(["key1", "key2", "key3"])
        assert rotator.available_keys == 3
        rotator.mark_rate_limited("key1")
        assert rotator.available_keys == 2

    def test_reset_clears_rate_limits(self):
        """reset() should clear rate limit tracking."""
        rotator = APIKeyRotator(["key1", "key2"])
        rotator.mark_rate_limited("key1")
        assert rotator.available_keys == 1
        rotator.reset()
        assert rotator.available_keys == 2

    def test_default_rotator_singleton(self):
        """get_default_rotator should return consistent instance."""
        # Clear any cached instance
        reset_default_rotator()

        # Create first rotator and verify singleton pattern
        rotator1 = get_default_rotator(["test_key1", "test_key2"])
        rotator2 = get_default_rotator()
        assert rotator1 is rotator2

        # Clean up
        reset_default_rotator()

    @pytest.mark.parametrize(
        "num_keys,num_rate_limited,expected_available",
        [
            (5, 0, 5),
            (5, 2, 3),
            (5, 5, 0),
            (3, 1, 2),
        ],
    )
    def test_rate_limit_counts(self, num_keys, num_rate_limited, expected_available):
        """Rate limiting should correctly track available keys."""
        keys = [f"key{i}" for i in range(num_keys)]
        rotator = APIKeyRotator(keys)
        for i in range(num_rate_limited):
            rotator.mark_rate_limited(f"key{i}")
        assert rotator.available_keys == expected_available

    def test_is_exhausted_false_initially(self):
        """is_exhausted should be False when keys are available."""
        rotator = APIKeyRotator(["key1", "key2"])
        assert rotator.is_exhausted is False

    def test_is_exhausted_true_when_all_rate_limited(self):
        """is_exhausted should be True when all keys are rate-limited."""
        rotator = APIKeyRotator(["key1", "key2"])
        rotator.mark_rate_limited("key1")
        assert rotator.is_exhausted is False
        rotator.mark_rate_limited("key2")
        assert rotator.is_exhausted is True

    def test_is_exhausted_false_with_no_keys(self):
        """is_exhausted should be False when no keys configured."""
        rotator = APIKeyRotator([])
        assert rotator.is_exhausted is False

    def test_current_key_returns_none_when_exhausted(self):
        """current_key should return None when all keys are rate-limited."""
        rotator = APIKeyRotator(["key1", "key2"])
        assert rotator.current_key == "key1"
        rotator.mark_rate_limited("key1")
        assert rotator.current_key == "key2"
        rotator.mark_rate_limited("key2")
        assert rotator.current_key is None

    def test_reset_restores_exhausted_keys(self):
        """reset() should restore exhausted keys."""
        rotator = APIKeyRotator(["key1"])
        rotator.mark_rate_limited("key1")
        assert rotator.is_exhausted is True
        assert rotator.current_key is None
        rotator.reset()
        assert rotator.is_exhausted is False
        assert rotator.current_key == "key1"


class TestCacheManager:
    """Tests for the cache manager."""

    def test_cache_miss(self):
        """Cache miss should return None."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)

        try:
            manager = CacheManager(db_path, ttl_seconds=3600)
            result = manager.get_cached_result("unknown query")
            assert result is None
        finally:
            os.unlink(db_path)


class TestNormalizeConversationHistory:
    """Tests for conversation history normalization."""

    @pytest.mark.parametrize(
        "history,expected_len",
        [
            (None, 0),
            ([], 0),
        ],
    )
    def test_empty_history(self, history, expected_len):
        """Empty or None history should return empty list."""
        assert len(normalize_conversation_history(history)) == expected_len

    def test_conversation_turn_objects(self):
        """ConversationTurn objects should pass through."""
        turns = [
            ConversationTurn(query="q1", answer="a1"),
            ConversationTurn(query="q2", answer="a2"),
        ]
        result = normalize_conversation_history(turns)
        assert len(result) == 2
        assert result[0].query == "q1"
        assert result[1].query == "q2"

    def test_dict_conversion(self):
        """Dict history should be converted to ConversationTurn."""
        history = [
            {"query": "q1", "answer": "a1", "citations": []},
            {"query": "q2", "answer": "a2", "citations": []},
        ]
        result = normalize_conversation_history(history)
        assert len(result) == 2
        assert isinstance(result[0], ConversationTurn)
        assert result[0].query == "q1"


class TestExtractGroundedAnswer:
    """Tests for grounded answer extraction."""

    def test_extract_from_search_results(self):
        """Should extract from search_results if present."""
        sr = SearchResult(
            url="http://test.com",
            title="Test",
            snippet="snippet",
            source="gemini_grounding",
        )
        sr.content = "Grounded content"
        search_results = [sr]
        answer, sources = extract_grounded_answer(search_results)
        assert answer == "Grounded content"
        assert len(sources) == 1

    @pytest.mark.parametrize(
        "search_results",
        [
            [],
            [
                SearchResult(
                    url="http://test.com",
                    title="Test",
                    snippet="snippet",
                    source="tavily",
                )
            ],
        ],
    )
    def test_no_grounded_content(self, search_results):
        """Should return empty if no grounded content."""
        answer, sources = extract_grounded_answer(search_results)
        assert answer == ""
        assert sources == []


class TestCreateInitialState:
    """Tests for initial state creation."""

    def test_creates_valid_state(self):
        """Should create state with all required fields."""
        state = create_initial_state("test query", "run-123", [])
        assert state["query"] == "test query"
        assert state["run_id"] == "run-123"
        assert state["plan"] == []
        assert state["search_results"] == []
        assert state["documents"] == []
        assert state["chunks"] == []
        assert state["retrieved"] == []
        assert state["retrieval_scores"] == []
        assert state["draft_answer"] == ""
        assert state["verified_answer"] == ""
        assert state["citations"] == []
        assert state["errors"] == []
        assert state["warnings"] == []
        assert state["adaptive_iterations"] == 0
        assert state["qc_passes"] == 0
        assert state["qc_notes"] == []
        assert state["time_sensitive"] is False
        assert state["grounded_answer"] == ""
        assert state["grounded_sources"] == []

    def test_includes_conversation_history(self):
        """Should include provided conversation history."""
        history = [ConversationTurn(query="q", answer="a")]
        state = create_initial_state("test", "run-1", history)
        assert state["conversation_history"] == history


class TestBuildResearchState:
    """Tests for research state building."""

    def test_builds_complete_state(self):
        """Should build ResearchState from result dict."""
        result = {
            "plan": [SearchQuery(text="q1")],
            "search_results": [],
            "documents": [],
            "chunks": [],
            "retrieved": [],
            "draft_answer": "draft",
            "verified_answer": "verified",
            "citations": [],
            "errors": [],
            "warnings": ["warn1"],
            "adaptive_iterations": 1,
            "qc_passes": 2,
            "qc_notes": ["note1"],
            "time_sensitive": True,
            "grounded_answer": "",
            "grounded_sources": [],
        }
        history = [ConversationTurn(query="q", answer="a")]
        state = build_research_state(
            query="test",
            run_id="run-1",
            result=result,
            conversation_history=history,
        )
        assert state.query == "test"
        assert state.run_id == "run-1"
        assert state.draft_answer == "draft"
        assert state.verified_answer == "verified"
        assert state.warnings == ["warn1"]
        assert state.adaptive_iterations == 1
        assert state.qc_passes == 2
        assert state.time_sensitive is True
