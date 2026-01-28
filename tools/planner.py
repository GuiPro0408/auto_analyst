"""Planner agent that decomposes a user query into searchable sub-tasks."""

import re
from datetime import datetime
from typing import List, Optional, Tuple

from api.logging_setup import get_logger
from api.state import SearchQuery
from tools.topic_utils import detect_query_topic
from tools.text_utils import STOPWORDS, detect_time_sensitive, extract_keywords


def _parse_lines(text: str) -> List[str]:
    lines = []
    for raw in text.splitlines():
        cleaned = re.sub(r"^[\\-\\d\\).\\s]+", "", raw).strip()
        if cleaned:
            lines.append(cleaned)
    return lines


def _current_season(_year: int, month: int) -> str:
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "fall"


def _build_search_query(text: str, rationale: str) -> SearchQuery:
    """Create a SearchQuery enriched with topic."""
    topic = detect_query_topic(text)
    return SearchQuery(
        text=text,
        rationale=rationale,
        topic=topic or "",
    )


def _extract_topic_keywords(query: str) -> List[str]:
    """Extract likely topic keywords from user query."""
    return list(extract_keywords(query, stopwords=STOPWORDS, min_len=3))


def heuristic_plan(
    query: str,
    max_tasks: int = 4,
    time_sensitive: bool = False,
    context: Optional[str] = None,
) -> List[SearchQuery]:
    """Fallback planner that derives keyword-based tasks.

    Args:
        query: User's search query
        max_tasks: Maximum number of search tasks to generate
        time_sensitive: If True, prioritize queries for current/recent content
        context: Optional prior conversation summary for follow-ups
    """
    logger = get_logger(__name__)
    logger.debug(
        "heuristic_plan_start",
        extra={
            "query": query,
            "max_tasks": max_tasks,
            "time_sensitive": time_sensitive,
        },
    )
    now = datetime.utcnow()
    base = query.strip()
    keywords_source = f"{base}\n{context}" if context else base

    if not base:
        # No query provided - return empty plan
        logger.warning("planner_empty_query", extra={"query": query})
        return []

    # Extract topic keywords for more targeted searches
    keywords = _extract_topic_keywords(keywords_source)
    logger.debug(
        "planner_keywords_extracted",
        extra={"keywords": keywords[:10], "keyword_count": len(keywords)},
    )
    topic = " ".join(keywords[:4]) if keywords else base

    # Build topic-aware queries without assuming any specific domain
    seeds = [
        # Primary: user's original query as-is (most direct)
        base,
    ]

    # Add topic-specific queries if we can identify meaningful keywords
    if keywords:
        if time_sensitive:
            # For time-sensitive queries, emphasize recency
            current_season = _current_season(now.year, now.month)
            seeds.extend(
                [
                    f"{topic} {now.year}",
                    f"{topic} {current_season} {now.year}",
                    f"{topic} latest releases {now.year}",
                ]
            )
        else:
            # Topic + current year for time-relevant results
            seeds.append(f"{topic} {now.year}")
            # Topic + guide/overview query for comprehensive info
            seeds.append(f"{topic} guide overview")
            # Topic + latest/news for recent updates
            seeds.append(f"{topic} latest news {now.year}")
    else:
        # If no keywords extracted, try variations of the original query
        seeds.extend(
            [
                f"{base} {now.year}",
                f"{base} guide",
                f"{base} information",
            ]
        )

    tasks = []
    for text in seeds[:max_tasks]:
        tasks.append(
            _build_search_query(text=text, rationale="Heuristic topic-aware query")
        )
        logger.debug(
            "planner_task_created",
            extra={"query": text, "rationale": "Heuristic topic-aware query"},
        )
    logger.info(
        "planner_heuristic_complete",
        extra={
            "tasks": len(tasks),
            "max_tasks": max_tasks,
            "topic": topic,
            "time_sensitive": time_sensitive,
            "context_included": bool(context),
        },
    )
    return tasks


def plan_query(
    query: str,
    max_tasks: int = 4,
    conversation_context: Optional[str] = None,
) -> Tuple[List[SearchQuery], bool]:
    """Generate search tasks using heuristics.

    Note: LLM planning moved to smart_search.py. This function now only
    uses heuristics for fallback scenarios.

    Args:
        query: The user's search query.
        max_tasks: Maximum number of search tasks to generate.
        conversation_context: Optional prior conversation summary.

    Returns:
        Tuple of (search_tasks, is_time_sensitive).
    """
    logger = get_logger(__name__)

    time_sensitive, matched_keywords = detect_time_sensitive(query)

    logger.info(
        "plan_query_start",
        extra={
            "query": query,
            "max_tasks": max_tasks,
            "time_sensitive": time_sensitive,
            "temporal_keywords": matched_keywords,
            "conversation_context": bool(conversation_context),
        },
    )

    tasks = heuristic_plan(
        query,
        max_tasks=max_tasks,
        time_sensitive=time_sensitive,
        context=conversation_context,
    )
    return tasks, time_sensitive
