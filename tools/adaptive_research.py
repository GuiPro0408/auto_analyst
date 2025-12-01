"""Adaptive research agent utilities."""

from typing import List, Optional, Tuple

from api.logging_setup import get_logger
from api.state import Chunk, SearchQuery
from tools.planner import heuristic_plan


def assess_context(
    retrieved: List[Chunk], min_chunks: int = 1, run_id: Optional[str] = None
) -> Tuple[bool, List[str]]:
    """Determine if additional search is needed."""
    logger = get_logger(__name__, run_id=run_id)
    warnings: List[str] = []
    needs_more = len(retrieved) < min_chunks
    logger.info(
        "assess_context",
        extra={
            "retrieved_count": len(retrieved),
            "min_chunks": min_chunks,
            "needs_more": needs_more,
        },
    )
    if needs_more:
        warnings.append("Insufficient retrieved context; triggering adaptive search.")
        logger.warning(
            "assess_context_insufficient",
            extra={"retrieved": len(retrieved), "required": min_chunks},
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
