"""Search tools using only free endpoints."""

import time
import warnings
from typing import Iterable, List, Optional, Tuple

import requests
import wikipedia
from duckduckgo_search import DDGS

from api.config import (
    SEARCH_BACKENDS,
    SEARCH_RATE_LIMIT_SECONDS,
    SEARCH_RETRIES,
)
from api.logging_setup import get_logger
from api.state import SearchQuery, SearchResult

# Suppress rename warning emitted by duckduckgo_search package
warnings.filterwarnings(
    "ignore",
    message="This package (`duckduckgo_search`) has been renamed to `ddgs`!",
    category=RuntimeWarning,
    module="duckduckgo_search",
)

def search_duckduckgo(
    query: str, max_results: int = 5, run_id: Optional[str] = None
) -> List[SearchResult]:
    logger = get_logger(__name__, run_id=run_id)
    results: List[SearchResult] = []
    with DDGS() as ddgs:
        for attempt in range(1, SEARCH_RETRIES + 2):
            try:
                for item in ddgs.text(query, max_results=max_results):
                    url = item.get("href") or ""
                    title = item.get("title") or ""
                    snippet = item.get("body") or ""
                    if not url:
                        continue
                    results.append(
                        SearchResult(
                            url=url,
                            title=title,
                            snippet=snippet,
                            source="duckduckgo",
                        )
                    )
                break
            except Exception as exc:
                logger.warning(
                    "duckduckgo_search_retry",
                    extra={"attempt": attempt, "error": str(exc)},
                )
                time.sleep(SEARCH_RATE_LIMIT_SECONDS)
    logger.info("duckduckgo_search_complete", extra={"results": len(results), "query": query})
    return results


def search_wikipedia(
    query: str, max_results: int = 3, run_id: Optional[str] = None
) -> List[SearchResult]:
    logger = get_logger(__name__, run_id=run_id)
    results: List[SearchResult] = []
    for attempt in range(1, SEARCH_RETRIES + 2):
        try:
            titles = wikipedia.search(query, results=max_results)
            for title in titles:
                try:
                    summary = wikipedia.summary(title, sentences=2, auto_suggest=False)
                except Exception:
                    summary = ""
                page_url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
                results.append(
                    SearchResult(
                        url=page_url, title=title, snippet=summary, source="wikipedia"
                    )
                )
            break
        except Exception as exc:
            logger.warning(
                "wikipedia_search_retry",
                extra={"attempt": attempt, "error": str(exc)},
            )
            time.sleep(SEARCH_RATE_LIMIT_SECONDS)
    logger.info("wikipedia_search_complete", extra={"results": len(results)})
    return results


def search_searx(
    query: str, host: str, max_results: int = 5, run_id: Optional[str] = None
) -> List[SearchResult]:
    """Optional SearxNG search if a host is provided."""
    logger = get_logger(__name__, run_id=run_id)
    params = {"q": query, "format": "json", "language": "en", "safesearch": 1}
    results: List[SearchResult] = []
    for attempt in range(1, SEARCH_RETRIES + 2):
        try:
            resp = requests.get(f"{host}/search", params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            for item in data.get("results", [])[:max_results]:
                results.append(
                    SearchResult(
                        url=item.get("url", ""),
                        title=item.get("title", ""),
                        snippet=item.get("content", ""),
                        source="searx",
                    )
                )
            break
        except Exception as exc:
            logger.warning(
                "searx_search_retry",
                extra={"attempt": attempt, "error": str(exc), "host": host},
            )
            time.sleep(SEARCH_RATE_LIMIT_SECONDS)
    logger.info("searx_search_complete", extra={"results": len(results), "host": host})
    return results


def dedupe_results(results: Iterable[SearchResult]) -> List[SearchResult]:
    seen = set()
    unique: List[SearchResult] = []
    for res in results:
        key = res.url.split("#")[0]
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(res)
    return unique


def run_search_tasks(
    tasks: List[SearchQuery],
    max_results: int = 5,
    searx_host: Optional[str] = None,
    run_id: Optional[str] = None,
) -> Tuple[List[SearchResult], List[str]]:
    """Run searches across configured free providers. Returns (results, warnings)."""
    logger = get_logger(__name__, run_id=run_id)
    all_results: List[SearchResult] = []
    warnings: List[str] = []
    for task in tasks:
        if "duckduckgo" in SEARCH_BACKENDS:
            all_results.extend(
                search_duckduckgo(task.text, max_results=max_results, run_id=run_id)
            )
        if "wikipedia" in SEARCH_BACKENDS:
            all_results.extend(
                search_wikipedia(
                    task.text, max_results=max_results // 2 or 1, run_id=run_id
                )
            )
        if "searx" in SEARCH_BACKENDS and searx_host:
            all_results.extend(
                search_searx(
                    task.text, host=searx_host, max_results=max_results, run_id=run_id
                )
            )
    deduped = dedupe_results(all_results)
    logger.info(
        "search_tasks_complete",
        extra={
            "tasks": len(tasks),
            "results": len(deduped),
            "backends": SEARCH_BACKENDS,
        },
    )
    if not deduped:
            warnings.append("No search results found; consider refining the query.")
    return deduped, warnings
