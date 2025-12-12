"""Tests for tools/adaptive_research.py."""

import pytest

from api.state import Chunk, SearchQuery
from tools.adaptive_research import assess_context, refine_plan


class TestAssessContext:
    """Tests for assess_context function."""

    def test_no_chunks_needs_more_research(self):
        """Empty retrieved list should trigger adaptive search."""
        needs_more, warnings = assess_context(retrieved=[], min_chunks=1)
        assert needs_more is True
        assert any("No retrieved context" in w for w in warnings)

    def test_sufficient_chunks_no_need(self):
        """Sufficient chunks without score check should not need more."""
        chunks = [
            Chunk(id="1", text="chunk 1", metadata={}),
            Chunk(id="2", text="chunk 2", metadata={}),
        ]
        needs_more, warnings = assess_context(retrieved=chunks, min_chunks=1)
        assert needs_more is False
        assert warnings == []

    def test_insufficient_chunks_needs_more(self):
        """Fewer chunks than min_chunks should trigger adaptive search."""
        chunks = [Chunk(id="1", text="chunk 1", metadata={})]
        needs_more, warnings = assess_context(retrieved=chunks, min_chunks=3)
        assert needs_more is True
        assert any("Insufficient" in w for w in warnings)

    def test_low_relevance_scores_needs_more(self):
        """Low average relevance score should trigger adaptive search."""
        chunks = [
            Chunk(id="1", text="chunk 1", metadata={}),
            Chunk(id="2", text="chunk 2", metadata={}),
        ]
        # MIN_RELEVANCE_THRESHOLD is 0.3, so avg of 0.2 should trigger
        scores = [0.2, 0.2]
        needs_more, warnings = assess_context(
            retrieved=chunks, min_chunks=1, scores=scores
        )
        assert needs_more is True
        assert any("low relevance" in w.lower() for w in warnings)

    def test_high_relevance_scores_no_need(self):
        """High average relevance score should not trigger adaptive search."""
        chunks = [
            Chunk(id="1", text="chunk 1", metadata={}),
            Chunk(id="2", text="chunk 2", metadata={}),
        ]
        # Scores above threshold
        scores = [0.8, 0.7]
        needs_more, warnings = assess_context(
            retrieved=chunks, min_chunks=1, scores=scores
        )
        assert needs_more is False

    def test_accepts_run_id_and_query(self):
        """Function should accept optional run_id and query parameters."""
        chunks = [Chunk(id="1", text="chunk 1", metadata={})]
        needs_more, warnings = assess_context(
            retrieved=chunks,
            min_chunks=1,
            run_id="test-run-123",
            query="test query",
        )
        # Should complete without error
        assert isinstance(needs_more, bool)

    def test_empty_scores_list(self):
        """Empty scores list should not cause errors."""
        chunks = [Chunk(id="1", text="chunk 1", metadata={})]
        needs_more, warnings = assess_context(retrieved=chunks, min_chunks=1, scores=[])
        # With empty scores, only chunk count is checked
        assert needs_more is False


class TestRefinePlan:
    """Tests for refine_plan function."""

    def test_refine_with_existing_plan(self):
        """Refine should return new tasks when plan exists."""
        current_plan = [SearchQuery(text="original query", rationale="initial search")]
        new_tasks = refine_plan(
            query="test query about solar energy",
            current_plan=current_plan,
        )
        assert isinstance(new_tasks, list)
        assert all(isinstance(t, SearchQuery) for t in new_tasks)
        # Should return some new tasks (broadened scope strategy)
        assert len(new_tasks) > 0

    def test_refine_with_empty_plan(self):
        """Refine should create initial plan when none exists."""
        new_tasks = refine_plan(
            query="test query about solar energy",
            current_plan=[],
        )
        assert isinstance(new_tasks, list)
        assert len(new_tasks) > 0
        # Should use initial_plan strategy

    def test_refine_accepts_run_id(self):
        """Function should accept optional run_id parameter."""
        new_tasks = refine_plan(
            query="test query",
            current_plan=[],
            run_id="test-run-456",
        )
        assert isinstance(new_tasks, list)

    def test_refine_limits_tasks(self):
        """Refine should limit the number of new tasks."""
        new_tasks = refine_plan(
            query="very long query about many topics",
            current_plan=[SearchQuery(text="existing", rationale="")],
        )
        # refine_plan uses max_tasks=2
        assert len(new_tasks) <= 2

    def test_refine_generates_different_queries(self):
        """Refine should generate different queries from the original."""
        original_query = "solar energy impact"
        current_plan = [SearchQuery(text=original_query, rationale="")]
        new_tasks = refine_plan(
            query=original_query,
            current_plan=current_plan,
        )
        # New tasks should not exactly match the original
        new_texts = [t.text for t in new_tasks]
        assert not all(t == original_query for t in new_texts)


class TestIntegration:
    """Integration tests for adaptive research flow."""

    def test_assess_then_refine_flow(self):
        """Test typical flow: assess context, then refine if needed."""
        # Start with no context
        chunks = []
        needs_more, warnings = assess_context(retrieved=chunks, min_chunks=1)
        assert needs_more is True

        # Refine the plan
        original_plan = [SearchQuery(text="climate change effects", rationale="")]
        new_tasks = refine_plan(
            query="climate change effects on agriculture",
            current_plan=original_plan,
        )

        # Should have new tasks to try
        assert len(new_tasks) > 0

    def test_sufficient_context_no_refine(self):
        """When context is sufficient, no refinement needed."""
        chunks = [
            Chunk(id="1", text="relevant content about topic", metadata={}),
            Chunk(id="2", text="more relevant content", metadata={}),
        ]
        scores = [0.8, 0.75]

        needs_more, warnings = assess_context(
            retrieved=chunks, min_chunks=1, scores=scores
        )
        assert needs_more is False
        assert warnings == []
