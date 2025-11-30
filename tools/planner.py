"""Planner agent that decomposes a user query into searchable sub-tasks."""

import re
from typing import List, Optional

from api.logging_setup import get_logger
from api.state import SearchQuery


def _parse_lines(text: str) -> List[str]:
    lines = []
    for raw in text.splitlines():
        cleaned = re.sub(r"^[\\-\\d\\).\\s]+", "", raw).strip()
        if cleaned:
            lines.append(cleaned)
    return lines


def heuristic_plan(query: str, max_tasks: int = 4) -> List[SearchQuery]:
    """Fallback planner that derives keyword-based tasks."""
    logger = get_logger(__name__)
    parts = re.split(r"[?.]", query)
    keywords = [p.strip() for p in parts if p.strip()]
    if not keywords:
        keywords = [query]
    tasks = []
    for idx, text in enumerate(keywords[:max_tasks]):
        rationale = (
            "Derived from query segment"
            if len(keywords) > 1
            else "Direct query expansion"
        )
        tasks.append(SearchQuery(text=text, rationale=rationale))
    logger.info("planner_heuristic_complete", extra={"tasks": len(tasks), "max_tasks": max_tasks})
    return tasks


def plan_query(query: str, llm=None, max_tasks: int = 4) -> List[SearchQuery]:
    """Use an instruct model (if provided) to plan; otherwise fall back to heuristics."""
    logger = get_logger(__name__)
    if not llm:
        return heuristic_plan(query, max_tasks=max_tasks)

    prompt = (
        "You are a research planner. Break the user question into at most "
        f"{max_tasks} focused web search tasks. Use concise, self-contained queries. "
        "Return each task on its own line as '<query> -- <rationale>'.\n"
        f"User question: {query}"
    )
    try:
        result = llm(prompt)[0]["generated_text"]
        candidates = _parse_lines(result)
        tasks: List[SearchQuery] = []
        for line in candidates[:max_tasks]:
            if "--" in line:
                q, rationale = [part.strip() for part in line.split("--", 1)]
            else:
                q, rationale = line.strip(), "LLM derived search task"
            tasks.append(SearchQuery(text=q, rationale=rationale))
        if tasks:
            logger.info("planner_llm_complete", extra={"tasks": len(tasks), "max_tasks": max_tasks})
            return tasks
    except Exception:
        # Swallow and fall back to heuristics; verification later can catch gaps.
        pass
    logger.warning("planner_fallback_heuristic", extra={"max_tasks": max_tasks})
    return heuristic_plan(query, max_tasks=max_tasks)
