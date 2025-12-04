"""Search tools using Gemini grounding and optional fallback backends."""

import time
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from typing import Dict, Iterable, List, Optional, Tuple

import requests

from api.config import (
    SEARCH_BACKENDS,
    SEARCH_RATE_LIMIT_SECONDS,
    SEARCH_RETRIES,
)
from api.logging_setup import get_logger
from api.state import SearchQuery, SearchResult
from tools.gemini_grounding import GroundingResult, query_with_grounding

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

# Topic categories for routing queries to appropriate backends
TOPIC_KEYWORDS = {
    "entertainment": {
        "anime",
        "manga",
        "movie",
        "film",
        "series",
        "tv",
        "show",
        "season",
        "episode",
        "game",
        "gaming",
        "music",
        "album",
        "song",
        "artist",
        "concert",
        "theater",
        "drama",
        "comedy",
        "horror",
        "action",
    },
    "technology": {
        "programming",
        "software",
        "hardware",
        "computer",
        "code",
        "coding",
        "python",
        "javascript",
        "api",
        "framework",
        "library",
        "database",
        "cloud",
        "server",
        "network",
        "cybersecurity",
        "ai",
        "machine learning",
    },
    "news": {
        "news",
        "breaking",
        "today",
        "latest",
        "update",
        "announcement",
        "report",
        "press",
        "media",
        "headline",
    },
    "science": {
        "research",
        "study",
        "experiment",
        "scientific",
        "physics",
        "chemistry",
        "biology",
        "medicine",
        "health",
        "disease",
        "treatment",
        "vaccine",
    },
}


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


class SearxBackend(SearchBackend):
    """SearxNG search backend."""

    name = "searx"

    def __init__(self, host: str):
        self.host = host

    def search(
        self,
        query: str,
        max_results: int = 5,
        run_id: Optional[str] = None,
    ) -> List[SearchResult]:
        """Optional SearxNG search if a host is provided."""
        logger = get_logger(__name__, run_id=run_id)
        logger.debug(
            "searx_search_start",
            extra={"query": query, "host": self.host, "max_results": max_results},
        )
        params = {"q": query, "format": "json", "language": "en", "safesearch": 1}
        results: List[SearchResult] = []
        for attempt in range(1, SEARCH_RETRIES + 2):
            try:
                resp = requests.get(f"{self.host}/search", params=params, timeout=10)
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
                    extra={"attempt": attempt, "error": str(exc), "host": self.host},
                )
                time.sleep(SEARCH_RATE_LIMIT_SECONDS)
        logger.info(
            "searx_search_complete", extra={"results": len(results), "host": self.host}
        )
        return results


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


# Registry of available search backends
_BACKEND_REGISTRY: dict[str, type[SearchBackend]] = {
    "gemini_grounding": GeminiGroundingBackend,
    "arxiv": ArxivBackend,
    "openalex": OpenAlexBackend,
}


def register_backend(name: str, backend_class: type[SearchBackend]) -> None:
    """Register a custom search backend."""
    _BACKEND_REGISTRY[name] = backend_class


def get_backend(name: str, **kwargs) -> Optional[SearchBackend]:
    """Get a search backend instance by name."""
    if name == "searx" and "host" in kwargs:
        return SearxBackend(host=kwargs["host"])
    backend_class = _BACKEND_REGISTRY.get(name)
    if backend_class:
        return backend_class()
    return None


def detect_query_topic(query: str) -> Optional[str]:
    """Detect the likely topic category of a query."""
    query_lower = query.lower()
    query_words = set(query_lower.split())

    best_topic = None
    best_overlap = 0

    for topic, keywords in TOPIC_KEYWORDS.items():
        overlap = len(query_words & keywords)
        # Also check for partial matches in the full query string
        for keyword in keywords:
            if keyword in query_lower and len(keyword) > 3:
                overlap += 0.5

        if overlap > best_overlap:
            best_overlap = overlap
            best_topic = topic

    return best_topic if best_overlap >= 1 else None


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
        # 3. The result is from gemini_grounding (already relevance-filtered)
        if overlap or not query_keywords or res.source == "gemini_grounding":
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
    backend_names = [
        backend.strip().lower() for backend in SEARCH_BACKENDS if backend.strip()
    ]
    backend_instances: List[SearchBackend] = []
    for name in backend_names:
        if name == "searx" and not searx_host:
            logger.info("search_backend_skip_searx_no_host")
            continue
        backend = get_backend(name, host=searx_host)
        if not backend:
            logger.warning("search_backend_unknown", extra={"backend": name})
            continue
        backend_instances.append(backend)

    for idx, task in enumerate(tasks):
        logger.debug(
            "search_task_processing",
            extra={"task_index": idx, "query": task.text, "rationale": task.rationale},
        )
        task_results_before = len(all_results)
        topic = detect_query_topic(task.text)
        for backend in backend_instances:
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
