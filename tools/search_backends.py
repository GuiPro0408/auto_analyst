"""Search backend implementations for Auto-Analyst.

This module contains the abstract SearchBackend class and concrete implementations
for Gemini Grounding and Tavily search backends.
"""

import os
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import requests

from api.config import (
    GROUNDING_SNIPPET_PREVIEW_LEN,
    SYNTHETIC_SNIPPET_MAX_LEN,
    TAVILY_API_KEY,
    TAVILY_SNIPPET_MAX_LEN,
)
from api.logging_setup import get_logger
from api.state import SearchResult
from tools.gemini_grounding import GroundingResult, query_with_grounding

# Canonical source identifiers
SOURCE_GEMINI_GROUNDING = "gemini_grounding"
SOURCE_TAVILY = "tavily"


class SearchBackend(ABC):
    """Abstract base class for search backends."""

    name: str = "base"

    @abstractmethod
    def search(
        self,
        query: str,
        max_results: int = 5,
        run_id: Optional[str] = None,
    ) -> Tuple[List[SearchResult], List[str]]:
        """Execute a search and return (results, warnings)."""
        pass


class GeminiGroundingBackend(SearchBackend):
    """Gemini with Google Search grounding for web-augmented search."""

    name = SOURCE_GEMINI_GROUNDING

    def search(
        self,
        query: str,
        max_results: int = 5,
        run_id: Optional[str] = None,
    ) -> Tuple[List[SearchResult], List[str]]:
        """Search using Gemini's Google Search grounding.

        Args:
            query: Search query string.
            max_results: Maximum results (note: grounding returns variable results).
            run_id: Optional run ID for logging correlation.

        Returns:
            Tuple of (SearchResult list, warnings list).
        """
        logger = get_logger(__name__, run_id=run_id)
        warnings: List[str] = []
        logger.info(
            "gemini_grounding_search_start",
            extra={"query": query[:100], "max_results": max_results},
        )

        result: GroundingResult = query_with_grounding(query, run_id=run_id)

        if not result.success:
            logger.warning(
                "gemini_grounding_search_failed",
                extra={"error": result.error, "query": query},
            )
            warnings.append(f"Gemini grounding failed: {result.error}")
            return [], warnings

        results: List[SearchResult] = []
        for source in result.sources[:max_results]:
            results.append(
                SearchResult(
                    url=source.url,
                    title=source.title,
                    snippet=(
                        source.snippet or result.answer[:GROUNDING_SNIPPET_PREVIEW_LEN]
                        if result.answer
                        else ""
                    ),
                    source=SOURCE_GEMINI_GROUNDING,
                    # Store the grounded answer as content for direct use
                    content=result.answer if len(results) == 0 else "",
                )
            )

        # If grounding returned an answer but no sources, create a synthetic result
        if result.answer and not results:
            logger.debug("gemini_grounding_no_sources_synthetic_result")
            results.append(
                SearchResult(
                    url="",
                    title="Gemini Grounded Response",
                    snippet=result.answer[:SYNTHETIC_SNIPPET_MAX_LEN],
                    source=SOURCE_GEMINI_GROUNDING,
                    content=result.answer,
                )
            )

        logger.info(
            "gemini_grounding_search_complete",
            extra={
                "results": len(results),
                "query": query,
                "web_queries": result.web_search_queries,
                "answer_length": len(result.answer) if result.answer else 0,
            },
        )
        return results, warnings


class TavilyBackend(SearchBackend):
    """Tavily search backend - reliable, RAG-optimized."""

    name = SOURCE_TAVILY

    def search(
        self,
        query: str,
        max_results: int = 10,
        run_id: Optional[str] = None,
        topic: str = "general",
        time_range: Optional[str] = None,
        include_domains: Optional[List[str]] = None,
    ) -> Tuple[List[SearchResult], List[str]]:
        """Execute search and return (results, warnings)."""
        logger = get_logger(__name__, run_id=run_id)
        warnings: List[str] = []
        api_key = TAVILY_API_KEY or os.getenv("TAVILY_API_KEY", "")

        if not api_key:
            logger.warning(
                "tavily_no_api_key",
                extra={"query": query[:100]},
            )
            warnings.append("Tavily API key not configured; skipping Tavily search.")
            return [], warnings

        payload = {
            "api_key": api_key,
            "query": query,
            "max_results": min(max_results, 20),
            "search_depth": "advanced",
            "include_answer": False,
            "include_raw_content": True,  # Get full pre-rendered page content
        }

        if topic in ("general", "news", "finance"):
            payload["topic"] = topic
        if time_range in ("day", "week", "month", "year", "d", "w", "m", "y"):
            payload["time_range"] = time_range
        if include_domains:
            payload["include_domains"] = include_domains[:300]

        logger.info(
            "tavily_search_start",
            extra={
                "query": query[:100],
                "max_results": max_results,
                "topic": topic,
                "include_domains": include_domains[:5] if include_domains else None,
                "time_range": time_range,
            },
        )

        try:
            resp = requests.post(
                "https://api.tavily.com/search",
                json=payload,
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()

            results: List[SearchResult] = []
            for item in data.get("results", []):
                # Tavily returns 'content' (snippet) and optionally 'raw_content' (full page)
                # raw_content is pre-rendered by Tavily's headless browser - use it!
                raw_content = item.get("raw_content", "") or ""
                snippet_content = item.get("content", "") or ""

                results.append(
                    SearchResult(
                        url=item.get("url", ""),
                        title=item.get("title", ""),
                        snippet=snippet_content[:TAVILY_SNIPPET_MAX_LEN],
                        source=SOURCE_TAVILY,
                        # Store raw_content if available, otherwise use snippet as content
                        content=raw_content if raw_content else snippet_content,
                        metadata={
                            "score": item.get("score", 0),
                            "has_raw_content": bool(raw_content),
                        },
                    )
                )

            logger.info(
                "tavily_search_complete",
                extra={
                    "results": len(results),
                    "query": query[:100],
                    "topic": topic,
                    "time_range": time_range,
                },
            )
            return results, warnings

        except requests.exceptions.Timeout as exc:
            logger.warning(
                "tavily_search_timeout",
                extra={"error": str(exc)[:200], "query": query[:100]},
            )
            warnings.append(f"Tavily search timed out: {exc}")
            return [], warnings
        except requests.exceptions.HTTPError as exc:
            logger.warning(
                "tavily_search_http_error",
                extra={
                    "error": str(exc)[:200],
                    "query": query[:100],
                    "status": getattr(exc.response, "status_code", None),
                },
            )
            warnings.append(f"Tavily HTTP error: {exc}")
            return [], warnings
        except Exception as exc:
            logger.warning(
                "tavily_search_failed",
                extra={"error": str(exc)[:200], "query": query[:100]},
            )
            warnings.append(f"Tavily search error: {exc}")
            return [], warnings


def canonical_source(name: str) -> str:
    """Normalize a source name to its canonical form.

    Args:
        name: Source name to canonicalize.

    Returns:
        Canonical source name.
    """
    normalized = name.strip().lower()
    if normalized in ("gemini_grounding", "gemini"):
        return SOURCE_GEMINI_GROUNDING
    if normalized == "tavily":
        return SOURCE_TAVILY
    return name


def get_backend(name: str) -> Optional[SearchBackend]:
    """Get a search backend instance by name."""
    normalized = name.strip().lower()
    if normalized == SOURCE_GEMINI_GROUNDING or normalized == "gemini":
        return GeminiGroundingBackend()
    if normalized == SOURCE_TAVILY or normalized == "tavily":
        return TavilyBackend()
    return None
