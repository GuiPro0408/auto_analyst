"""Adaptive research agent utilities."""

from typing import List, Optional, Tuple

from api.logging_setup import get_logger
from api.state import Chunk, SearchQuery
from tools.planner import heuristic_plan

# Minimum average similarity score to consider context relevant
MIN_RELEVANCE_THRESHOLD = 0.3


def assess_context(
    retrieved: List[Chunk],
    min_chunks: int = 1,
    run_id: Optional[str] = None,
    query: Optional[str] = None,
    scores: Optional[List[float]] = None,
) -> Tuple[bool, List[str]]:
    """Determine if additional search is needed based on chunk count AND relevance.

    Args:
        retrieved: List of retrieved chunks
        min_chunks: Minimum number of chunks required
        run_id: Run correlation ID for logging
        query: Original query (used for logging context)
        scores: Similarity scores from vector store (if available)

    Returns:
        Tuple of (needs_more_research, warnings)
    """
    logger = get_logger(__name__, run_id=run_id)
    warnings: List[str] = []

    # Check 1: Do we have enough chunks?
    insufficient_chunks = len(retrieved) < min_chunks

    # Check 2: Are the chunks relevant enough? (if scores provided)
    low_relevance = False
    avg_score = 0.0
    if scores and len(scores) > 0:
        avg_score = sum(scores) / len(scores)
        low_relevance = avg_score < MIN_RELEVANCE_THRESHOLD

    needs_more = insufficient_chunks or low_relevance

    logger.info(
        "assess_context",
        extra={
            "retrieved_count": len(retrieved),
            "min_chunks": min_chunks,
            "avg_score": round(avg_score, 4) if scores else None,
            "relevance_threshold": MIN_RELEVANCE_THRESHOLD,
            "insufficient_chunks": insufficient_chunks,
            "low_relevance": low_relevance,
            "needs_more": needs_more,
            "query": query,
        },
    )

    if insufficient_chunks:
        warnings.append("Insufficient retrieved context; triggering adaptive search.")
        logger.warning(
            "assess_context_insufficient_chunks",
            extra={"retrieved": len(retrieved), "required": min_chunks},
        )

    if low_relevance:
        warnings.append(
            f"Retrieved context has low relevance (avg score: {avg_score:.3f} < {MIN_RELEVANCE_THRESHOLD}); "
            "triggering adaptive search."
        )
        logger.warning(
            "assess_context_low_relevance",
            extra={
                "avg_score": round(avg_score, 4),
                "threshold": MIN_RELEVANCE_THRESHOLD,
                "scores": [round(s, 4) for s in scores] if scores else [],
            },
        )

    return needs_more, warnings


def refine_plan(
    query: str, current_plan: List[SearchQuery], run_id: Optional[str] = None
) -> List[SearchQuery]:
    """Produce additional search tasks based on the original query."""
    logger = get_logger(__name__, run_id=run_id)
    logger.info(
        "refine_plan_start",
        extra={"query": query, "current_plan_tasks": len(current_plan)},
    )
    if current_plan:
        # Reuse heuristic planner to broaden scope slightly.
        new_tasks = heuristic_plan(f"{query} statistics trends impact", max_tasks=2)
        logger.info(
            "refine_plan_complete",
            extra={"new_tasks": len(new_tasks), "strategy": "broadened_scope"},
        )
        return new_tasks
    new_tasks = heuristic_plan(query, max_tasks=2)
    logger.info(
        "refine_plan_complete",
        extra={"new_tasks": len(new_tasks), "strategy": "initial_plan"},
    )
    return new_tasks
