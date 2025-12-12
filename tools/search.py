"""Search tools using Gemini grounding and optional fallback backends."""

import os
from abc import ABC, abstractmethod
from typing import Dict, Iterable, List, Optional, Tuple

import requests

from api.config import (
    SEARCH_BACKENDS,
    TAVILY_API_KEY,
)
from api.logging_setup import get_logger
from api.state import SearchQuery, SearchResult
from tools.gemini_grounding import GroundingResult, query_with_grounding
from tools.text_utils import STOPWORDS, detect_time_sensitive, extract_keywords

BLOCKED_DOMAINS = {
    "support.google.com",
    "mail.google.com",
    "zhidao.baidu.com",
    "zhihu.com",
    "baidu.com",
    "quora.com",
    "reddit.com/user/",
    "facebook.com",
    "twitter.com",
    "instagram.com",
    "tiktok.com",
    "pinterest.com",
    "linkedin.com",
}

# Domains that often return meta-content (about language, grammar) rather than topic content
META_CONTENT_DOMAINS = {
    "stackexchange.com",
    "ell.stackexchange.com",
    "english.stackexchange.com",
    "linguistics.stackexchange.com",
    "grammar.com",
    "merriam-webster.com/grammar",
    "dictionary.com",
    "thesaurus.com",
}

KNOWN_ROBOTS_BLOCKED_DOMAINS = {
    "anime-planet.com",
    "usingenglish.com",
}

FALLBACK_BACKEND_ORDER = ["tavily", "gemini_grounding"]

# Canonical source identifiers (limited to tavily + gemini)
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
        logger.debug(
            "gemini_grounding_search_start",
            extra={"query": query, "max_results": max_results},
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
                        source.snippet or result.answer[:200] if result.answer else ""
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
                    snippet=result.answer[:300],
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
        }

        if topic in ("general", "news", "finance"):
            payload["topic"] = topic
        if time_range in ("day", "week", "month", "year", "d", "w", "m", "y"):
            payload["time_range"] = time_range
        if include_domains:
            payload["include_domains"] = include_domains[:300]

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
                results.append(
                    SearchResult(
                        url=item.get("url", ""),
                        title=item.get("title", ""),
                        snippet=item.get("content", "")[:500],
                        source=SOURCE_TAVILY,
                        metadata={"score": item.get("score", 0)},
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
                extra={"error": str(exc)[:200], "query": query[:100], "status": getattr(exc.response, 'status_code', None)},
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


def dedupe_results(results: Iterable[SearchResult]) -> List[SearchResult]:
    logger = get_logger(__name__)
    seen = set()
    unique: List[SearchResult] = []
    duplicates = 0
    for res in results:
        key = res.url.split("#")[0]
        if not key or key in seen:
            duplicates += 1
            continue
        seen.add(key)
        unique.append(res)
    logger.debug(
        "dedupe_complete",
        extra={
            "input": len(list(results)) if hasattr(results, "__len__") else "unknown",
            "unique": len(unique),
            "duplicates": duplicates,
        },
    )
    return unique


def filter_results(
    query: str, results: List[SearchResult], preferred_domains: Optional[set] = None
) -> List[SearchResult]:
    """Drop obviously irrelevant or blocked domains; keep query-related results."""
    logger = get_logger(__name__)
    logger.debug(
        "filter_results_start",
        extra={"query": query, "input_count": len(results)},
    )
    filtered: List[SearchResult] = []
    query_keywords = extract_keywords(query, stopwords=STOPWORDS)
    blocked_count = 0
    robots_blocked_count = 0
    meta_filtered_count = 0
    irrelevant_count = 0

    for res in results:
        url = res.url or ""

        # Skip blocked domains
        if any(domain in url for domain in BLOCKED_DOMAINS):
            blocked_count += 1
            logger.debug("filter_blocked_domain", extra={"url": url})
            continue
        if any(domain in url for domain in KNOWN_ROBOTS_BLOCKED_DOMAINS):
            robots_blocked_count += 1
            logger.debug("filter_robots_blocked_domain", extra={"url": url})
            continue

        # Skip meta-content domains (grammar/language sites) unless query is about language
        language_terms = {
            "grammar",
            "word",
            "meaning",
            "definition",
            "synonym",
            "language",
        }
        if any(domain in url for domain in META_CONTENT_DOMAINS):
            if not query_keywords & language_terms:
                meta_filtered_count += 1
                logger.debug("filter_meta_content_domain", extra={"url": url})
                continue

        title = (res.title or "").lower()
        snippet = (res.snippet or "").lower()
        combined_text = f"{title} {snippet}"
        result_keywords = extract_keywords(combined_text, stopwords=STOPWORDS)

        # Check for keyword overlap between query and result
        overlap = query_keywords & result_keywords

        # Include result if there's keyword overlap, no query keywords, or the source
        # is gemini grounding (already relevance-filtered).
        if overlap or not query_keywords or res.source == SOURCE_GEMINI_GROUNDING:
            filtered.append(res)
        else:
            irrelevant_count += 1
            logger.debug(
                "filter_no_keyword_overlap",
                extra={
                    "url": url,
                    "query_keywords": list(query_keywords),
                    "result_keywords": list(result_keywords)[:10],
                },
            )

    logger.info(
        "filter_results_complete",
        extra={
            "input": len(results),
            "output": len(filtered),
            "blocked": blocked_count,
            "robots_blocked": robots_blocked_count,
            "meta_filtered": meta_filtered_count,
            "irrelevant": irrelevant_count,
        },
    )
    return filtered


def run_search_tasks(
    tasks: List[SearchQuery],
    max_results: int = 5,
    run_id: Optional[str] = None,
) -> Tuple[List[SearchResult], List[str]]:
    """Run searches across configured free providers. Returns (results, warnings)."""
    logger = get_logger(__name__, run_id=run_id)
    logger.info(
        "run_search_tasks_start",
        extra={
            "task_count": len(tasks),
            "max_results": max_results,
            "backends": SEARCH_BACKENDS,
        },
    )
    all_results: List[SearchResult] = []
    warnings: List[str] = []
    time_sensitive = any(detect_time_sensitive(t.text)[0] for t in tasks)
    backend_names = [
        backend.strip().lower() for backend in SEARCH_BACKENDS if backend.strip()
    ]
    backend_instances: List[SearchBackend] = []
    attempted_backends: set[str] = set()

    for name in backend_names:
        backend = get_backend(name)
        if not backend:
            logger.warning("search_backend_unknown", extra={"backend": name})
            continue
        backend_instances.append(backend)
        attempted_backends.add(backend.name)

    # Always include Gemini grounding for time-sensitive queries even if not configured explicitly
    if time_sensitive and SOURCE_GEMINI_GROUNDING not in attempted_backends:
        gemini_backend = get_backend(SOURCE_GEMINI_GROUNDING)
        if gemini_backend:
            backend_instances.append(gemini_backend)
            attempted_backends.add(gemini_backend.name)

    for idx, task in enumerate(tasks):
        logger.debug(
            "search_task_processing",
            extra={
                "task_index": idx,
                "query": task.text,
                "rationale": task.rationale,
            },
        )
        task_results_before = len(all_results)
        for backend in backend_instances:
            results, backend_warnings = backend.search(
                task.text,
                max_results=max_results,
                run_id=run_id,
            )
            warnings.extend(backend_warnings)
            for res in results:
                res.source = res.source or backend.name
            logger.debug(
                "search_backend_complete",
                extra={
                    "backend": backend.name,
                    "results": len(results),
                    "task_index": idx,
                },
            )
            all_results.extend(results)
        logger.debug(
            "search_task_complete",
            extra={
                "task_index": idx,
                "results_added": len(all_results) - task_results_before,
            },
        )

    combined_query = " ".join([t.text for t in tasks])
    deduped = filter_results(combined_query, dedupe_results(all_results), preferred_domains=None)

    if not deduped:
        fallback_chain = [
            name for name in FALLBACK_BACKEND_ORDER if name and name not in attempted_backends
        ]
        for backend_name in fallback_chain:
            backend = get_backend(backend_name)
            if not backend:
                logger.warning(
                    "search_backend_unknown",
                    extra={"backend": backend_name, "phase": "fallback"},
                )
                continue
            attempted_backends.add(backend.name)

            for idx, task in enumerate(tasks):
                logger.debug(
                    "search_fallback_task_processing",
                    extra={"task_index": idx, "backend": backend.name, "query": task.text},
                )
                results, backend_warnings = backend.search(
                    task.text,
                    max_results=max_results,
                    run_id=run_id,
                )
                warnings.extend(backend_warnings)
                for res in results:
                    res.source = res.source or backend.name
                all_results.extend(results)

            deduped = filter_results(
                combined_query,
                dedupe_results(all_results),
                preferred_domains=None,
            )
            if deduped:
                logger.info(
                    "search_fallback_succeeded",
                    extra={"backend": backend.name, "results": len(deduped)},
                )
                break

    logger.info(
        "search_tasks_complete",
        extra={
            "tasks": len(tasks),
            "results": len(deduped),
            "backends": SEARCH_BACKENDS,
            "fallback_chain": FALLBACK_BACKEND_ORDER,
        },
    )
    if not deduped:
        warnings.append("No search results found; consider refining the query.")
    return deduped, warnings
