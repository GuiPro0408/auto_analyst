"""Search tools using Gemini grounding and optional fallback backends."""

import os
import time
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from typing import Dict, Iterable, List, Optional, Tuple

import requests

from api.config import (
    SEARCH_BACKENDS,
    SEARCH_RATE_LIMIT_SECONDS,
    SEARCH_RETRIES,
    TAVILY_API_KEY,
)
from api.logging_setup import get_logger
from api.state import SearchQuery, SearchResult
from tools.gemini_grounding import GroundingResult, query_with_grounding
from tools.topic_utils import detect_query_topic

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

PREFERRED_SPORTS_DOMAINS = {
    "premierleague.com",
    "bbc.co.uk/sport",
    "bbc.com/sport",
    "skysports.com",
    "espn.com",
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

PREFERRED_DOMAINS_BY_TOPIC = {
    "entertainment": {
        "animeschedule.net",
        "myanimelist.net",
        "animenewsnetwork.com",
        "crunchyroll.com",
        "polygon.com",
        "ign.com",
    },
    "technology": {
        "arstechnica.com",
        "theverge.com",
        "techcrunch.com",
        "wired.com",
        "github.com",
    },
    "science": {
        "nature.com",
        "science.org",
        "nih.gov",
        "nasa.gov",
        "arxiv.org",
    },
    "news": {
        "reuters.com",
        "apnews.com",
        "bbc.com",
        "bbc.co.uk",
        "theguardian.com",
    },
    "sports": set(PREFERRED_SPORTS_DOMAINS),
}

DEFAULT_FALLBACK_CHAIN = "tavily,gemini_grounding,wikipedia,ddgs,arxiv"
FALLBACK_BACKEND_ORDER = [
    name.strip().lower()
    for name in os.getenv("AUTO_ANALYST_FALLBACK_BACKENDS", DEFAULT_FALLBACK_CHAIN).split(",")
    if name.strip()
]

class SearchBackend(ABC):
    """Abstract base class for search backends."""

    name: str = "base"

    @abstractmethod
    def search(
        self,
        query: str,
        max_results: int = 5,
        run_id: Optional[str] = None,
    ) -> List[SearchResult]:
        """Execute a search and return results."""
        pass

    def supports_topic(self, topic: str) -> bool:
        """Check if this backend is suitable for a given topic category."""
        return True  # Default: supports all topics


class ArxivBackend(SearchBackend):
    """ArXiv API backend for scientific literature."""

    name = "arxiv"

    def search(
        self,
        query: str,
        max_results: int = 5,
        run_id: Optional[str] = None,
    ) -> List[SearchResult]:
        logger = get_logger(__name__, run_id=run_id)
        logger.debug(
            "arxiv_search_start",
            extra={"query": query, "max_results": max_results},
        )
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
        }
        try:
            resp = requests.get(
                "https://export.arxiv.org/api/query",
                params=params,
                timeout=10,
            )
            resp.raise_for_status()
        except Exception as exc:
            logger.warning("arxiv_search_failed", extra={"error": str(exc)})
            return []

        entries: List[SearchResult] = []
        try:
            root = ET.fromstring(resp.text)
        except ET.ParseError as exc:
            logger.warning("arxiv_parse_failed", extra={"error": str(exc)})
            return []
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        for entry in root.findall("atom:entry", ns):
            title = entry.findtext("atom:title", default="", namespaces=ns).strip()
            summary = entry.findtext("atom:summary", default="", namespaces=ns).strip()
            link = entry.findtext("atom:id", default="", namespaces=ns).strip()
            entries.append(
                SearchResult(
                    url=link,
                    title=title,
                    snippet=summary,
                    source="arxiv",
                    content=summary,
                )
            )
        logger.info(
            "arxiv_search_complete",
            extra={"results": len(entries), "query": query},
        )
        return entries

    def supports_topic(self, topic: str) -> bool:
        return topic in {"science", "technology"}


class OpenAlexBackend(SearchBackend):
    """OpenAlex search backend for scholarly works."""

    name = "openalex"

    @staticmethod
    def _abstract_from_index(index: Optional[Dict[str, List[int]]]) -> str:
        if not index:
            return ""
        max_pos = max(
            (pos for positions in index.values() for pos in positions), default=-1
        )
        if max_pos < 0:
            return ""
        tokens = ["" for _ in range(max_pos + 1)]
        for word, positions in index.items():
            for pos in positions:
                if 0 <= pos < len(tokens):
                    tokens[pos] = word
        return " ".join(t for t in tokens if t).strip()

    def search(
        self,
        query: str,
        max_results: int = 5,
        run_id: Optional[str] = None,
    ) -> List[SearchResult]:
        logger = get_logger(__name__, run_id=run_id)
        logger.debug(
            "openalex_search_start",
            extra={"query": query, "max_results": max_results},
        )
        params = {
            "search": query,
            "per_page": max(1, max_results),
            "sort": "relevance_score:desc",
        }
        try:
            resp = requests.get(
                "https://api.openalex.org/works",
                params=params,
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning("openalex_search_failed", extra={"error": str(exc)})
            return []

        results: List[SearchResult] = []
        for item in data.get("results", [])[:max_results]:
            title = item.get("display_name", "")
            url = item.get("id", "")
            snippet = self._abstract_from_index(item.get("abstract_inverted_index"))
            if not snippet:
                snippet = item.get("primary_topic", {}).get("display_name", "")
            results.append(
                SearchResult(
                    url=url,
                    title=title,
                    snippet=snippet,
                    source="openalex",
                    content=snippet,
                )
            )
        logger.info(
            "openalex_search_complete",
            extra={"results": len(results), "query": query},
        )
        return results

    def supports_topic(self, topic: str) -> bool:
        return topic in {"science", "technology"}


class GeminiGroundingBackend(SearchBackend):
    """Gemini with Google Search grounding for web-augmented search.

    This backend uses Gemini's native Google Search grounding to get fresh
    web results. It is the primary search backend for Auto-Analyst.
    """

    name = "gemini_grounding"

    def search(
        self,
        query: str,
        max_results: int = 5,
        run_id: Optional[str] = None,
    ) -> List[SearchResult]:
        """Search using Gemini's Google Search grounding.

        Args:
            query: Search query string.
            max_results: Maximum results (note: grounding returns variable results).
            run_id: Optional run ID for logging correlation.

        Returns:
            List of SearchResult from grounding sources.
        """
        logger = get_logger(__name__, run_id=run_id)
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
            return []

        results: List[SearchResult] = []
        for source in result.sources[:max_results]:
            results.append(
                SearchResult(
                    url=source.url,
                    title=source.title,
                    snippet=(
                        source.snippet or result.answer[:200] if result.answer else ""
                    ),
                    source="gemini_grounding",
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
                    source="gemini_grounding",
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
        return results

    def supports_topic(self, topic: str) -> bool:
        """Gemini grounding supports all topics via Google Search."""
        return True


class TavilyBackend(SearchBackend):
    """Tavily search backend - reliable, RAG-optimized."""

    name = "tavily"

    def search(
        self,
        query: str,
        max_results: int = 10,
        topic: str = "general",
        time_range: Optional[str] = None,
        include_domains: Optional[List[str]] = None,
        run_id: Optional[str] = None,
    ) -> List[SearchResult]:
        logger = get_logger(__name__, run_id=run_id)
        api_key = TAVILY_API_KEY or os.getenv("TAVILY_API_KEY", "")

        if not api_key:
            logger.warning(
                "tavily_no_api_key",
                extra={"query": query[:100]},
            )
            return []

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
                        source="tavily",
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
            return results

        except Exception as exc:
            logger.warning(
                "tavily_search_failed",
                extra={"error": str(exc)[:200], "query": query[:100]},
            )
            return []


class DDGSBackend(SearchBackend):
    """DuckDuckGo Search backend using the duckduckgo_search library.

    Uses the duckduckgo_search library with fallback to HTML scraping
    if the library fails.
    """

    name = "ddgs"

    def search(
        self,
        query: str,
        max_results: int = 5,
        run_id: Optional[str] = None,
    ) -> List[SearchResult]:
        """Search using DuckDuckGo.

        Args:
            query: Search query string.
            max_results: Maximum number of results.
            run_id: Optional run ID for logging correlation.

        Returns:
            List of SearchResult from DuckDuckGo.
        """
        logger = get_logger(__name__, run_id=run_id)
        logger.debug(
            "ddgs_search_start",
            extra={"query": query, "max_results": max_results},
        )

        # Try HTML.duckduckgo.com which works better for specific queries
        results = self._search_html_duckduckgo(query, max_results, logger)
        if results:
            return results

        # Fallback to the library
        results = self._search_ddgs_library(query, max_results, logger)
        if results:
            return results

        logger.warning("ddgs_search_all_methods_failed", extra={"query": query[:50]})
        return []

    def _search_html_duckduckgo(
        self, query: str, max_results: int, logger
    ) -> List[SearchResult]:
        """Search using html.duckduckgo.com with POST."""
        try:
            from bs4 import BeautifulSoup
            from urllib.parse import unquote

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Content-Type": "application/x-www-form-urlencoded",
            }

            # Use POST to html.duckduckgo.com
            url = "https://html.duckduckgo.com/html/"
            data = {"q": query, "kl": "us-en", "kp": "-2"}  # kp=-2 disables safe search

            resp = requests.post(url, data=data, headers=headers, timeout=15)

            if resp.status_code != 200:
                # DDG occasionally returns 202; retry with GET HTML endpoint
                logger.debug("ddgs_html_bad_status", extra={"status": resp.status_code})
                alt_url = "https://duckduckgo.com/html/"
                resp = requests.get(
                    alt_url, params={"q": query, "kl": "us-en", "kp": "-2"}, headers=headers, timeout=15
                )
                if resp.status_code != 200:
                    # Final fallback to lite endpoint
                    lite_url = "https://lite.duckduckgo.com/lite/"
                    resp = requests.get(
                        lite_url,
                        params={"q": query},
                        headers=headers,
                        timeout=15,
                    )
                    if resp.status_code != 200:
                        return []

            soup = BeautifulSoup(resp.text, "html.parser")
            results: List[SearchResult] = []

            # Find result divs
            for result_div in soup.find_all("div", class_="result"):
                if len(results) >= max_results:
                    break

                # Get title and URL
                title_link = result_div.find("a", class_="result__a")
                if not title_link:
                    continue

                title = title_link.get_text(strip=True)
                href = title_link.get("href", "")

                # Extract actual URL from DuckDuckGo redirect
                actual_url = href
                if "uddg=" in href:
                    try:
                        actual_url = unquote(href.split("uddg=")[1].split("&")[0])
                    except (IndexError, ValueError):
                        pass

                if not actual_url or not title:
                    continue

                # Skip blocked domains
                if any(domain in actual_url for domain in BLOCKED_DOMAINS):
                    continue

                # Get snippet
                snippet_elem = result_div.find("a", class_="result__snippet")
                snippet = (
                    snippet_elem.get_text(strip=True)[:300] if snippet_elem else ""
                )

                results.append(
                    SearchResult(
                        url=actual_url,
                        title=title,
                        snippet=snippet,
                        source="ddgs",
                    )
                )

            if results:
                logger.info(
                    "ddgs_search_complete",
                    extra={
                        "results": len(results),
                        "query": query[:50],
                        "method": "html",
                    },
                )
            return results

        except Exception as exc:
            logger.debug("ddgs_html_failed", extra={"error": str(exc)[:100]})
            return []

    def _search_ddgs_library(
        self, query: str, max_results: int, logger
    ) -> List[SearchResult]:
        """Search using the duckduckgo_search library."""
        try:
            # Prefer new ddgs package; fall back to legacy duckduckgo_search
            try:
                from ddgs import DDGS  # type: ignore

                logger.debug("ddgs_library_using_pkg", extra={"package": "ddgs"})
            except ImportError:
                from duckduckgo_search import DDGS  # type: ignore

                logger.debug(
                    "ddgs_library_using_pkg", extra={"package": "duckduckgo_search"}
                )

            results: List[SearchResult] = []
            with DDGS() as ddgs:
                # Use wt-wt region for worldwide English results
                search_results = list(
                    ddgs.text(query, max_results=max_results * 2, region="wt-wt")
                )

            for r in search_results:
                if len(results) >= max_results:
                    break

                url = r.get("href", "")
                title = r.get("title", "")

                # Skip blocked domains and Chinese results
                if any(domain in url for domain in BLOCKED_DOMAINS):
                    continue
                if any(
                    domain in url for domain in ["baidu.com", "zhihu.com", "zhidao"]
                ):
                    continue

                results.append(
                    SearchResult(
                        url=url,
                        title=title,
                        snippet=r.get("body", "")[:300],
                        source="ddgs",
                    )
                )

            if results:
                logger.info(
                    "ddgs_search_complete",
                    extra={
                        "results": len(results),
                        "query": query[:50],
                        "method": "library",
                    },
                )
            return results

        except Exception as exc:
            logger.debug("ddgs_library_failed", extra={"error": str(exc)[:100]})
            return []

    def supports_topic(self, topic: str) -> bool:
        """DDGS supports all topics via metasearch."""
        return True


class WikipediaBackend(SearchBackend):
    """Wikipedia API search backend.

    Free and reliable, good for factual queries.
    Uses Wikipedia's official API.
    """

    name = "wikipedia"

    def search(
        self,
        query: str,
        max_results: int = 5,
        run_id: Optional[str] = None,
    ) -> List[SearchResult]:
        """Search Wikipedia using the official API."""
        from urllib.parse import quote_plus

        logger = get_logger(__name__, run_id=run_id)
        logger.debug("wikipedia_search_start", extra={"query": query[:50]})

        results: List[SearchResult] = []

        try:
            # Use Wikipedia's API for search
            api_url = "https://en.wikipedia.org/w/api.php"
            params = {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": max_results,
                "format": "json",
                "srprop": "snippet|titlesnippet",
            }

            resp = requests.get(api_url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            for item in data.get("query", {}).get("search", []):
                title = item.get("title", "")
                snippet = item.get("snippet", "")
                # Clean HTML from snippet
                snippet = snippet.replace('<span class="searchmatch">', "").replace(
                    "</span>", ""
                )

                url = f"https://en.wikipedia.org/wiki/{quote_plus(title.replace(' ', '_'))}"

                results.append(
                    SearchResult(
                        url=url,
                        title=title,
                        snippet=snippet[:500],
                        source="wikipedia",
                    )
                )

            logger.info(
                "wikipedia_search_complete",
                extra={"results": len(results), "query": query[:50]},
            )
            return results

        except Exception as exc:
            logger.warning(
                "wikipedia_search_failed",
                extra={"error": str(exc)[:100], "query": query[:50]},
            )
            return []

    def supports_topic(self, topic: str) -> bool:
        """Wikipedia is good for most factual topics."""
        return True


# Registry of available search backends
_BACKEND_REGISTRY: dict[str, type[SearchBackend]] = {
    "gemini_grounding": GeminiGroundingBackend,
    "tavily": TavilyBackend,
    "wikipedia": WikipediaBackend,
    "ddgs": DDGSBackend,
    "duckduckgo": DDGSBackend,  # Alias for convenience
    "arxiv": ArxivBackend,
    "openalex": OpenAlexBackend,
}


def register_backend(name: str, backend_class: type[SearchBackend]) -> None:
    """Register a custom search backend."""
    _BACKEND_REGISTRY[name] = backend_class


def get_backend(name: str, **kwargs) -> Optional[SearchBackend]:
    """Get a search backend instance by name."""
    normalized = name.strip().lower()
    if normalized == "tavily":
        return TavilyBackend()
    backend_class = _BACKEND_REGISTRY.get(normalized)
    if backend_class:
        return backend_class()
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


def _is_time_sensitive_query(query: str) -> bool:
    """Heuristic to detect queries that need fresh/live data."""
    q = query.lower()
    temporal_terms = {
        "current",
        "currently",
        "today",
        "latest",
        "now",
        "live",
        "standing",
        "standings",
        "table",
        "ranking",
        "released",
        "releasing",
        "release",
        "upcoming",
        "schedule",
        "fall",
        "spring",
        "summer",
        "winter",
    }
    return any(term in q for term in temporal_terms)


def _boost_preferred_domains(
    results: List[SearchResult], preferred_domains: set
) -> List[SearchResult]:
    """Return results with preferred domains first, preserving relative order."""
    if not preferred_domains:
        return results
    preferred: List[SearchResult] = []
    rest: List[SearchResult] = []
    for res in results:
        if any(domain in (res.url or "") for domain in preferred_domains):
            preferred.append(res)
        else:
            rest.append(res)
    return preferred + rest


def _preferred_domains_for_query(query: str) -> set:
    """Return domains to prioritize for topic-aligned questions."""
    topic = detect_query_topic(query)
    preferred: set[str] = set()
    if topic and topic in PREFERRED_DOMAINS_BY_TOPIC:
        preferred.update(PREFERRED_DOMAINS_BY_TOPIC[topic])
    # Always include sport-specific overrides
    if topic == "sports":
        preferred.update(PREFERRED_SPORTS_DOMAINS)
    return preferred


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
    query_keywords = _extract_keywords(query)
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

        preferred_hit = preferred_domains and any(
            domain in url for domain in preferred_domains
        )

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
        # 1. It's on a preferred domain for the query,
        # 2. There's keyword overlap, OR
        # 3. No keywords could be extracted from query, OR
        # 4. The result is from gemini_grounding (already relevance-filtered)
        if preferred_hit or overlap or not query_keywords or res.source == "gemini_grounding":
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
    time_sensitive = any(_is_time_sensitive_query(t.text) for t in tasks)
    backend_names = [
        backend.strip().lower() for backend in SEARCH_BACKENDS if backend.strip()
    ]
    backend_instances: List[SearchBackend] = []
    attempted_backends: set[str] = set()

    for name in backend_names:
        backend = get_backend(name, host=searx_host)
        if not backend:
            logger.warning("search_backend_unknown", extra={"backend": name})
            continue
        backend_instances.append(backend)
        attempted_backends.add(backend.name)

    # Always include Gemini grounding for time-sensitive queries even if not configured explicitly
    if time_sensitive and "gemini_grounding" not in attempted_backends:
        gemini_backend = get_backend("gemini_grounding")
        if gemini_backend:
            backend_instances.append(gemini_backend)
            attempted_backends.add(gemini_backend.name)

    def run_backends(instances: List[SearchBackend]) -> None:
        nonlocal all_results
        for idx, task in enumerate(tasks):
            logger.debug(
                "search_task_processing",
                extra={"task_index": idx, "query": task.text, "rationale": task.rationale},
            )
            task_results_before = len(all_results)
            topic = task.topic or detect_query_topic(task.text)
            for backend in instances:
                if topic and not backend.supports_topic(topic):
                    logger.debug(
                        "search_backend_skipped_topic",
                        extra={"backend": backend.name, "topic": topic},
                    )
                    continue
                results = backend.search(
                    task.text,
                    max_results=max_results,
                    run_id=run_id,
                )
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

    def finalize_results() -> List[SearchResult]:
        combined_query = " ".join([t.text for t in tasks])
        preferred_domains = _preferred_domains_for_query(combined_query)
        for task in tasks:
            preferred_domains.update(task.preferred_domains or [])
        deduped = dedupe_results(all_results)
        deduped = filter_results(
            combined_query, deduped, preferred_domains=preferred_domains
        )
        deduped = _boost_preferred_domains(deduped, preferred_domains)
        return deduped

    run_backends(backend_instances)
    deduped = finalize_results()

    if not deduped:
        fallback_chain = [
            name
            for name in FALLBACK_BACKEND_ORDER
            if name and name not in attempted_backends
        ]
        for backend_name in fallback_chain:
            backend = get_backend(backend_name, host=searx_host)
            if not backend:
                logger.warning(
                    "search_backend_unknown",
                    extra={"backend": backend_name, "phase": "fallback"},
                )
                continue
            attempted_backends.add(backend.name)
            run_backends([backend])
            deduped = finalize_results()
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
