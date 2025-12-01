"""Planner agent that decomposes a user query into searchable sub-tasks."""

import re
from datetime import datetime
from typing import List, Optional, Tuple

from api.logging_setup import get_logger
from api.state import SearchQuery


# Temporal keywords that indicate time-sensitive queries
TEMPORAL_KEYWORDS = {
    "upcoming",
    "coming",
    "next",
    "new",
    "latest",
    "recent",
    "current",
    "this season",
    "this year",
    "this month",
    "this week",
    "today",
    "2024",
    "2025",
    "2026",  # Current and near years
    "release",
    "releases",
    "premiere",
    "premieres",
    "launch",
    "launches",
    "announced",
    "announcement",
    "schedule",
    "scheduled",
}


def _parse_lines(text: str) -> List[str]:
    lines = []
    for raw in text.splitlines():
        cleaned = re.sub(r"^[\\-\\d\\).\\s]+", "", raw).strip()
        if cleaned:
            lines.append(cleaned)
    return lines


def _current_season(year: int, month: int) -> str:
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "fall"


def detect_time_sensitive(query: str) -> Tuple[bool, List[str]]:
    """Detect if a query is time-sensitive based on temporal keywords.

    Returns:
        Tuple of (is_time_sensitive, matched_keywords)
    """
    query_lower = query.lower()
    matched = []

    for keyword in TEMPORAL_KEYWORDS:
        if keyword in query_lower:
            matched.append(keyword)

    return len(matched) > 0, matched


def _extract_topic_keywords(query: str) -> List[str]:
    """Extract likely topic keywords from user query."""
    # Common filler words to skip
    stopwords = {
        "what",
        "which",
        "when",
        "where",
        "who",
        "how",
        "is",
        "are",
        "the",
        "a",
        "an",
        "for",
        "this",
        "that",
        "these",
        "those",
        "do",
        "does",
        "can",
        "could",
        "will",
        "would",
        "should",
        "may",
        "might",
        "to",
        "of",
        "in",
        "on",
        "at",
        "by",
        "with",
        "about",
        "from",
        "and",
        "or",
        "but",
        "if",
        "then",
        "so",
        "because",
        "as",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "under",
        "again",
        "further",
        "once",
        "here",
        "there",
        "all",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "than",
        "too",
        "very",
        "just",
        "now",
        "any",
        "me",
        "my",
        "i",
        "you",
        "your",
        "we",
        "our",
        "they",
        "their",
        "it",
        "its",
    }
    words = re.findall(r"\b[a-zA-Z]+\b", query.lower())
    keywords = [w for w in words if w not in stopwords and len(w) > 2]
    return keywords


def heuristic_plan(
    query: str, max_tasks: int = 4, time_sensitive: bool = False
) -> List[SearchQuery]:
    """Fallback planner that derives keyword-based tasks.

    Args:
        query: User's search query
        max_tasks: Maximum number of search tasks to generate
        time_sensitive: If True, prioritize queries for current/recent content
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

    if not base:
        # No query provided - return empty plan
        logger.warning("planner_empty_query", extra={"query": query})
        return []

    # Extract topic keywords for more targeted searches
    keywords = _extract_topic_keywords(base)
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
        tasks.append(SearchQuery(text=text, rationale="Heuristic topic-aware query"))
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
        },
    )
    return tasks


def plan_query(
    query: str, llm=None, max_tasks: int = 4
) -> Tuple[List[SearchQuery], bool]:
    """Use an instruct model (if provided) to plan; otherwise fall back to heuristics.

    Returns:
        Tuple of (search_tasks, is_time_sensitive)
    """
    logger = get_logger(__name__)

    # Detect if query is time-sensitive
    time_sensitive, matched_keywords = detect_time_sensitive(query)

    logger.info(
        "plan_query_start",
        extra={
            "query": query,
            "max_tasks": max_tasks,
            "has_llm": llm is not None,
            "time_sensitive": time_sensitive,
            "temporal_keywords": matched_keywords,
        },
    )

    if not llm:
        logger.debug("plan_query_using_heuristic", extra={"reason": "no_llm_provided"})
        tasks = heuristic_plan(
            query, max_tasks=max_tasks, time_sensitive=time_sensitive
        )
        return tasks, time_sensitive

    prompt = (
        "You are a research planner. Break the user question into at most "
        f"{max_tasks} focused web search tasks. Use concise, self-contained queries. "
        "Return each task on its own line as '<query> -- <rationale>'.\n"
        f"User question: {query}"
    )
    logger.debug("planner_llm_prompt", extra={"prompt_length": len(prompt)})
    try:
        result = llm(prompt)[0]["generated_text"]
        logger.debug(
            "planner_llm_response",
            extra={"response_length": len(result), "response_preview": result[:200]},
        )
        candidates = _parse_lines(result)
        logger.debug(
            "planner_parsed_candidates",
            extra={"candidate_count": len(candidates)},
        )
        tasks: List[SearchQuery] = []
        for line in candidates[:max_tasks]:
            if "--" in line:
                q, rationale = [part.strip() for part in line.split("--", 1)]
            else:
                q, rationale = line.strip(), "LLM derived search task"
            tasks.append(SearchQuery(text=q, rationale=rationale))
            logger.debug(
                "planner_llm_task_created",
                extra={"query": q, "rationale": rationale},
            )
        if tasks:
            logger.info(
                "planner_llm_complete",
                extra={"tasks": len(tasks), "max_tasks": max_tasks},
            )
            return tasks, time_sensitive
    except Exception as exc:
        logger.warning(
            "planner_llm_failed",
            extra={"error": str(exc), "error_type": type(exc).__name__},
        )
        # Swallow and fall back to heuristics; verification later can catch gaps.
        pass
    logger.warning(
        "planner_fallback_heuristic",
        extra={"max_tasks": max_tasks, "reason": "llm_failed_or_no_tasks"},
    )
    tasks = heuristic_plan(query, max_tasks=max_tasks, time_sensitive=time_sensitive)
    return tasks, time_sensitive
