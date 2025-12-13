"""Smart search with LLM-powered query analysis and result validation."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from api.config import VALIDATE_RESULTS_ENABLED
from api.logging_setup import get_logger
from api.state import SearchQuery, SearchResult
from tools.models import load_llm
from tools.search import run_search_tasks

QUERY_ANALYSIS_PROMPT = """Analyze this search query and provide a strategy.\n\nQuery: {query}\n\nRespond in JSON only (no markdown):\n{{\n  \"intent\": \"news|factual|comparison|howto|opinion\",\n  \"entities\": [\"entity1\", \"entity2\"],\n  \"topic\": \"technology|science|sports|entertainment|news|finance|health|general\",\n  \"time_sensitivity\": \"realtime|recent|any\",\n  \"suggested_searches\": [\n    {{\"query\": \"specific search 1\", \"rationale\": \"why\"}},\n    {{\"query\": \"specific search 2\", \"rationale\": \"why\"}}\n  ],\n  \"authoritative_sources\": [\"domain1.com\", \"domain2.com\"]\n}}"""

RESULT_VALIDATION_PROMPT = """Given the user query, determine which search results are relevant.\n\nUser Query: {query}\n\nSearch Results:\n{results_text}\n\nReturn ONLY a JSON array of relevant result numbers (1-indexed).\nExample: [1, 3, 5, 7]\n\nIf none are relevant, return: []"""


def _extract_json_payload(text: str) -> str:
    """Extract JSON payload from free-form LLM output."""
    if "```json" in text:
        text = text.split("```json", 1)[1]
        if "```" in text:
            text = text.split("```", 1)[0]
    elif "```" in text:
        text = text.split("```", 1)[1]
        if "```" in text:
            text = text.split("```", 1)[0]
    return text.strip()


def analyze_query_with_llm(query: str, run_id: Optional[str] = None) -> Dict[str, Any]:
    """Use LLM to understand query intent and generate search strategy."""
    logger = get_logger(__name__, run_id=run_id)
    logger.info("query_analysis_start", extra={"query": query})

    llm = load_llm()
    prompt = QUERY_ANALYSIS_PROMPT.format(query=query)
    text = ""

    try:
        response = llm(prompt)
        text = response[0].get("generated_text", "") if response else ""
        payload = _extract_json_payload(text)
        analysis = json.loads(payload)
        logger.info(
            "query_analysis_complete",
            extra={
                "topic": analysis.get("topic"),
                "intent": analysis.get("intent"),
                "entities": analysis.get("entities"),
                "suggested_searches": len(analysis.get("suggested_searches", [])),
            },
        )
        return analysis
    except (json.JSONDecodeError, IndexError, KeyError) as exc:
        logger.warning(
            "query_analysis_failed",
            extra={
                "error": str(exc),
                "error_type": type(exc).__name__,
                "response_preview": text[:500] if text else "empty",
                "response_full_length": len(text) if text else 0,
            },
        )
        return {
            "intent": "general",
            "entities": [],
            "topic": "general",
            "time_sensitivity": "any",
            "suggested_searches": [{"query": query, "rationale": "original query"}],
            "authoritative_sources": [],
        }


def validate_results_with_llm(
    query: str,
    results: List[SearchResult],
    run_id: Optional[str] = None,
) -> List[SearchResult]:
    """Use LLM to filter irrelevant search results (batch validation)."""
    logger = get_logger(__name__, run_id=run_id)

    if not results:
        return []

    if not VALIDATE_RESULTS_ENABLED:
        logger.info("result_validation_skipped", extra={"reason": "disabled"})
        return results

    logger.info("result_validation_start", extra={"results": len(results)})

    results_text = "\n".join(
        [
            f"{i + 1}. [{res.title}]({res.url})\n   {(res.snippet or '')[:200]}..."
            for i, res in enumerate(results)
        ]
    )

    llm = load_llm()
    prompt = RESULT_VALIDATION_PROMPT.format(query=query, results_text=results_text)
    text = ""

    try:
        response = llm(prompt)
        text = response[0].get("generated_text", "") if response else "[]"

        if "[" in text and "]" in text:
            start = text.index("[")
            end = text.rindex("]") + 1
            text = text[start:end]

        indices = json.loads(text)
        validated = [results[i - 1] for i in indices if 0 < i <= len(results)]
        logger.info(
            "result_validation_complete",
            extra={
                "input": len(results),
                "valid": len(validated),
                "filtered_out": len(results) - len(validated),
            },
        )
        return validated
    except (json.JSONDecodeError, IndexError) as exc:
        logger.warning(
            "result_validation_failed",
            extra={
                "error": str(exc),
                "error_type": type(exc).__name__,
                "response_preview": text[:500] if text else "empty",
                "response_full_length": len(text) if text else 0,
            },
        )
        return results


def smart_search(
    query: str,
    max_results: int = 10,
    run_id: Optional[str] = None,
) -> Tuple[List[SearchResult], List[str]]:
    """Autonomous search pipeline with LLM analysis and validation."""
    logger = get_logger(__name__, run_id=run_id)
    logger.info(
        "smart_search_start", extra={"query": query, "max_results": max_results}
    )
    warnings: List[str] = []

    analysis = analyze_query_with_llm(query, run_id=run_id)

    llm_topic = analysis.get("topic", "general")
    tavily_topic = "general"
    if llm_topic in ("news", "politics", "sports", "entertainment"):
        tavily_topic = "news"
    elif llm_topic in ("finance", "stocks", "crypto", "economics"):
        tavily_topic = "finance"

    time_sensitivity = analysis.get("time_sensitivity", "any")
    time_range = None
    if time_sensitivity == "realtime":
        time_range = "day"
    elif time_sensitivity == "recent":
        time_range = "week"

    include_domains = analysis.get("authoritative_sources", [])

    from tools.search import TavilyBackend

    tavily = TavilyBackend()
    tavily_result_tuple = tavily.search(
        query=query,
        max_results=max_results,
        topic=tavily_topic,
        time_range=time_range,
        include_domains=include_domains if include_domains else None,
        run_id=run_id,
    )
    tavily_results = tavily_result_tuple[0]
    tavily_warnings = tavily_result_tuple[1]
    warnings.extend(tavily_warnings)

    if tavily_results:
        validated = validate_results_with_llm(query, tavily_results, run_id=run_id)
        if not validated:
            warning_msg = (
                f"LLM filtered all {len(tavily_results)} results as irrelevant; "
                "using unvalidated top results as fallback."
            )
            warnings.append(warning_msg)
            logger.warning(
                "smart_search_all_filtered", extra={"results": len(tavily_results)}
            )
            validated = tavily_results[:5]

        logger.info(
            "smart_search_complete",
            extra={
                "raw_results": len(tavily_results),
                "validated_results": len(validated),
                "topic": analysis.get("topic"),
                "intent": analysis.get("intent"),
                "source": "tavily_direct",
            },
        )
        return validated, warnings

    search_tasks: List[SearchQuery] = []
    for suggestion in analysis.get("suggested_searches", []):
        search_tasks.append(
            SearchQuery(
                text=suggestion.get("query", query),
                rationale=suggestion.get("rationale", "LLM suggested"),
                topic=analysis.get("topic", "general"),
            )
        )

    if not search_tasks:
        search_tasks.append(
            SearchQuery(
                text=query,
                rationale="Original query (no LLM suggestions)",
                topic=analysis.get("topic", "general"),
            )
        )

    search_result_tuple = run_search_tasks(
        search_tasks,
        max_results=max_results,
        run_id=run_id,
    )
    results = search_result_tuple[0]
    search_warnings = search_result_tuple[1]
    warnings.extend(search_warnings)

    validated_results = validate_results_with_llm(query, results, run_id=run_id)

    if not validated_results and results:
        warning_msg = (
            f"LLM filtered all {len(results)} results as irrelevant; "
            "using unvalidated top results as fallback."
        )
        warnings.append(warning_msg)
        logger.warning("smart_search_all_filtered", extra={"results": len(results)})
        validated_results = results[:5]

    logger.info(
        "smart_search_complete",
        extra={
            "raw_results": len(results),
            "validated_results": len(validated_results),
            "topic": analysis.get("topic"),
            "intent": analysis.get("intent"),
            "source": "fallback_run_search_tasks",
        },
    )

    return validated_results, warnings
