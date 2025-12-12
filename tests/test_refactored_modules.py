"""Tests for refactored modules: key_rotator, cache_manager, state_builder."""

import pytest
from unittest.mock import MagicMock, patch
import os
import tempfile

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

    def test_single_key_rotation(self):
        """Single key should always return the same key."""
        rotator = APIKeyRotator(["key1"])
        assert rotator.current_key == "key1"
        assert rotator.current_key == "key1"
        assert rotator.current_key == "key1"

    def test_empty_keys(self):
        """Empty keys list should return None."""
        rotator = APIKeyRotator([])
        assert rotator.current_key is None

    def test_none_keys(self):
        """None keys should return None."""
        rotator = APIKeyRotator(None)
        assert rotator.current_key is None

    def test_rate_limited_key_skipped(self):
        """Rate-limited keys should be skipped."""
        rotator = APIKeyRotator(["key1", "key2", "key3"])
        rotator.mark_rate_limited("key1")
        assert rotator.current_key == "key2"

    def test_total_keys(self):
        """total_keys should return number of configured keys."""
        rotator = APIKeyRotator(["key1", "key2", "key3"])
        assert rotator.total_keys == 3

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


class TestCacheManager:
    """Tests for the cache manager."""

    def test_cache_miss(self):
        """Cache miss should return None."""
        from pathlib import Path

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

    def test_none_history(self):
        """None history should return empty list."""
        assert normalize_conversation_history(None) == []

    def test_empty_history(self):
        """Empty history should return empty list."""
        assert normalize_conversation_history([]) == []

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

    def test_no_grounded_content(self):
        """Should return empty if no grounded content."""
        search_results = []
        answer, sources = extract_grounded_answer(search_results)
        assert answer == ""
        assert sources == []

    def test_non_grounding_results_ignored(self):
        """Non-gemini_grounding results should not be extracted."""
        sr = SearchResult(
            url="http://test.com",
            title="Test",
            snippet="snippet",
            source="tavily",
        )
        search_results = [sr]
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
