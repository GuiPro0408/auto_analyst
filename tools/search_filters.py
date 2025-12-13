"""Search result filtering and deduplication for Auto-Analyst.

This module contains utilities for filtering, deduplicating, and validating
search results before they enter the RAG pipeline.
"""

from typing import Iterable, List, Optional, Set

from api.logging_setup import get_logger
from api.state import SearchResult
from tools.text_utils import STOPWORDS, extract_keywords

# Domains that are blocked due to low quality or policy violations
BLOCKED_DOMAINS: Set[str] = {
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
META_CONTENT_DOMAINS: Set[str] = {
    "stackexchange.com",
    "ell.stackexchange.com",
    "english.stackexchange.com",
    "linguistics.stackexchange.com",
    "grammar.com",
    "merriam-webster.com/grammar",
    "dictionary.com",
    "thesaurus.com",
}

# Domains known to block automated fetching via robots.txt
KNOWN_ROBOTS_BLOCKED_DOMAINS: Set[str] = {
    "anime-planet.com",
    "usingenglish.com",
}

# Source identifier for Gemini grounding (used in filter logic)
SOURCE_GEMINI_GROUNDING = "gemini_grounding"


def dedupe_results(results: Iterable[SearchResult]) -> List[SearchResult]:
    """Remove duplicate search results based on URL.

    Args:
        results: Iterable of SearchResult objects.

    Returns:
        List of unique SearchResult objects.
    """
    logger = get_logger(__name__)
    seen: Set[str] = set()
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
    query: str,
    results: List[SearchResult],
    preferred_domains: Optional[Set[str]] = None,
) -> List[SearchResult]:
    """Drop obviously irrelevant or blocked domains; keep query-related results.

    Args:
        query: The search query for relevance checking.
        results: List of search results to filter.
        preferred_domains: Optional set of preferred domains to prioritize.

    Returns:
        Filtered list of SearchResult objects.
    """
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
