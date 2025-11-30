"""Adaptive research agent utilities."""

from typing import List, Tuple

from api.state import Chunk, SearchQuery
from tools.planner import heuristic_plan


def assess_context(
    retrieved: List[Chunk], min_chunks: int = 1
) -> Tuple[bool, List[str]]:
    """Determine if additional search is needed."""
    warnings: List[str] = []
    needs_more = len(retrieved) < min_chunks
    if needs_more:
        warnings.append("Insufficient retrieved context; triggering adaptive search.")
    return needs_more, warnings


def refine_plan(query: str, current_plan: List[SearchQuery]) -> List[SearchQuery]:
    """Produce additional search tasks based on the original query."""
    if current_plan:
        # Reuse heuristic planner to broaden scope slightly.
        return heuristic_plan(f"{query} statistics trends impact", max_tasks=2)
    return heuristic_plan(query, max_tasks=2)
