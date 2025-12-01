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


def search_duckduckgo(
    query: str, max_results: int = 5, run_id: Optional[str] = None
) -> List[SearchResult]:
    logger = get_logger(__name__, run_id=run_id)
    logger.debug(
        "duckduckgo_search_start",
        extra={"query": query, "max_results": max_results},
    )
    results: List[SearchResult] = []
    with DDGS() as ddgs:
        for attempt in range(1, SEARCH_RETRIES + 2):
            try:
                for item in ddgs.text(query, max_results=max_results):
                    url = item.get("href") or ""
                    title = item.get("title") or ""
                    snippet = item.get("body") or ""
                    if not url:
                        logger.debug("duckduckgo_skip_no_url", extra={"title": title})
                        continue
                    logger.debug(
                        "duckduckgo_result_found",
                        extra={"url": url, "title": title[:50]},
                    )
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
    logger.info(
        "duckduckgo_search_complete", extra={"results": len(results), "query": query}
    )
    return results


def search_wikipedia(
    query: str, max_results: int = 3, run_id: Optional[str] = None
) -> List[SearchResult]:
    """Search Wikipedia and return results with full summaries as content.

    Wikipedia blocks most user agents via robots.txt, so we use the wikipedia
    API to get summaries directly rather than trying to fetch pages later.
    """
    logger = get_logger(__name__, run_id=run_id)
    logger.debug(
        "wikipedia_search_start",
        extra={"query": query, "max_results": max_results},
    )
    results: List[SearchResult] = []
    for attempt in range(1, SEARCH_RETRIES + 2):
        try:
            titles = wikipedia.search(query, results=max_results)
            logger.debug(
                "wikipedia_titles_found",
                extra={"titles": titles, "count": len(titles)},
            )
            for title in titles:
                try:
                    # Get a longer summary (5 sentences) since we'll use this as content
                    summary = wikipedia.summary(title, sentences=5, auto_suggest=False)
                    logger.debug(
                        "wikipedia_summary_fetched",
                        extra={"title": title, "chars": len(summary)},
                    )
                except Exception as exc:
                    logger.debug(
                        "wikipedia_summary_failed",
                        extra={"title": title, "error": str(exc)},
                    )
                    summary = ""
                if not summary:
                    logger.debug("wikipedia_skip_no_summary", extra={"title": title})
                    continue  # Skip results without content
                page_url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
                results.append(
                    SearchResult(
                        url=page_url,
                        title=title,
                        snippet=summary,
                        source="wikipedia",
                        # Store full content so we can use it without fetching
                        content=summary,
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
    logger.debug(
        "searx_search_start",
        extra={"query": query, "host": host, "max_results": max_results},
    )
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


def _extract_keywords(text: str) -> set:
    """Extract meaningful keywords from text."""
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
        "do",
        "does",
        "can",
        "will",
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
        "should",
        "use",
        "using",
        "i",
        "you",
        "we",
        "they",
        "it",
        "my",
    }
    words = set(text.lower().split())
    return {w.strip(".,!?;:") for w in words if w not in stopwords and len(w) > 2}


def filter_results(query: str, results: List[SearchResult]) -> List[SearchResult]:
    """Drop obviously irrelevant or blocked domains; keep query-related results."""
    logger = get_logger(__name__)
    logger.debug(
        "filter_results_start",
        extra={"query": query, "input_count": len(results)},
    )
    filtered: List[SearchResult] = []
    query_keywords = _extract_keywords(query)
    blocked_count = 0
    meta_filtered_count = 0
    irrelevant_count = 0

    for res in results:
        url = res.url or ""

        # Skip blocked domains
        if any(domain in url for domain in BLOCKED_DOMAINS):
            blocked_count += 1
            logger.debug("filter_blocked_domain", extra={"url": url})
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
        result_keywords = _extract_keywords(combined_text)

        # Check for keyword overlap between query and result
        overlap = query_keywords & result_keywords

        # Include result if:
        # 1. There's keyword overlap, OR
        # 2. No keywords could be extracted from query, OR
        # 3. The result is from Wikipedia (generally reliable and on-topic)
        if overlap or not query_keywords or res.source == "wikipedia":
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
            "meta_filtered": meta_filtered_count,
            "irrelevant": irrelevant_count,
        },
    )
    return filtered


def run_search_tasks(
    tasks: List[SearchQuery],
    max_results: int = 5,
    searx_host: Optional[str] = None,
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
            "searx_host": searx_host,
        },
    )
    all_results: List[SearchResult] = []
    warnings: List[str] = []
    for idx, task in enumerate(tasks):
        logger.debug(
            "search_task_processing",
            extra={"task_index": idx, "query": task.text, "rationale": task.rationale},
        )
        task_results_before = len(all_results)
        if "duckduckgo" in SEARCH_BACKENDS:
            ddg_results = search_duckduckgo(
                task.text, max_results=max_results, run_id=run_id
            )
            logger.debug(
                "search_backend_complete",
                extra={
                    "backend": "duckduckgo",
                    "results": len(ddg_results),
                    "task_index": idx,
                },
            )
            all_results.extend(ddg_results)
        if "wikipedia" in SEARCH_BACKENDS:
            wiki_results = search_wikipedia(
                task.text, max_results=max_results // 2 or 1, run_id=run_id
            )
            logger.debug(
                "search_backend_complete",
                extra={
                    "backend": "wikipedia",
                    "results": len(wiki_results),
                    "task_index": idx,
                },
            )
            all_results.extend(wiki_results)
        if "searx" in SEARCH_BACKENDS and searx_host:
            searx_results = search_searx(
                task.text, host=searx_host, max_results=max_results, run_id=run_id
            )
            logger.debug(
                "search_backend_complete",
                extra={
                    "backend": "searx",
                    "results": len(searx_results),
                    "task_index": idx,
                },
            )
            all_results.extend(searx_results)
        logger.debug(
            "search_task_complete",
            extra={
                "task_index": idx,
                "results_added": len(all_results) - task_results_before,
            },
        )
    deduped = dedupe_results(all_results)
    deduped = filter_results(" ".join([t.text for t in tasks]), deduped)
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
