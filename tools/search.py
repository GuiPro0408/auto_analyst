"""Search tools using Gemini grounding and optional fallback backends.

This module provides the public API for search functionality.
Implementation details are split into:
- search_backends.py: Backend implementations (GeminiGroundingBackend, TavilyBackend)
- search_filters.py: Filtering and deduplication utilities
"""

from typing import List, Optional, Tuple

from api.config import SEARCH_BACKENDS
from api.logging_setup import get_logger
from api.state import SearchQuery, SearchResult
from tools.text_utils import detect_time_sensitive

# Re-export backend classes and utilities for backward compatibility
from tools.search_backends import (
    GeminiGroundingBackend,
    SearchBackend,
    SOURCE_GEMINI_GROUNDING,
    SOURCE_TAVILY,
    TavilyBackend,
    canonical_source,
    get_backend,
)

# Re-export filter utilities for backward compatibility
from tools.search_filters import (
    BLOCKED_DOMAINS,
    KNOWN_ROBOTS_BLOCKED_DOMAINS,
    META_CONTENT_DOMAINS,
    dedupe_results,
    filter_results,
)

# Re-export grounding utilities for backward compatibility
from tools.gemini_grounding import query_with_grounding, GroundingResult, GroundingSource

# Fallback order when primary backends return no results
FALLBACK_BACKEND_ORDER = ["tavily", "gemini_grounding"]

# Public exports
__all__ = [
    # Constants
    "SOURCE_GEMINI_GROUNDING",
    "SOURCE_TAVILY",
    "FALLBACK_BACKEND_ORDER",
    "BLOCKED_DOMAINS",
    "META_CONTENT_DOMAINS",
    "KNOWN_ROBOTS_BLOCKED_DOMAINS",
    # Classes
    "SearchBackend",
    "GeminiGroundingBackend",
    "TavilyBackend",
    "GroundingResult",
    "GroundingSource",
    # Functions
    "canonical_source",
    "get_backend",
    "dedupe_results",
    "filter_results",
    "run_search_tasks",
    "query_with_grounding",
]


def run_search_tasks(
    tasks: List[SearchQuery],
    max_results: int = 5,
    run_id: Optional[str] = None,
) -> Tuple[List[SearchResult], List[str]]:
    """Run searches across configured free providers.

    Args:
        tasks: List of SearchQuery objects to execute.
        max_results: Maximum results per task.
        run_id: Optional run ID for logging correlation.

    Returns:
        Tuple of (results list, warnings list).
    """
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

    # Always include Gemini grounding for time-sensitive queries even if not configured
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
    deduped = filter_results(
        combined_query, dedupe_results(all_results), preferred_domains=None
    )

    if not deduped:
        fallback_chain = [
            name
            for name in FALLBACK_BACKEND_ORDER
            if name and name not in attempted_backends
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
                    extra={
                        "task_index": idx,
                        "backend": backend.name,
                        "query": task.text,
                    },
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
