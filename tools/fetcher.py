"""Fetcher that retrieves web pages and PDFs, respecting robots.txt."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple
import time
import threading
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests

from api.config import (
    FETCH_BACKOFF_SECONDS,
    FETCH_CONCURRENCY,
    FETCH_RETRIES,
    FETCH_TIMEOUT,
    MIN_CONTENT_LENGTH,
    ROBOTS_CACHE_TTL_SECONDS,
    ROBOTS_ON_ERROR,
    USER_AGENT,
)
from api.logging_setup import get_logger
from api.state import Document, SearchResult
from tools.parser import parse_html, parse_pdf

# Timeout for robots.txt fetch (shorter than main fetch timeout)
ROBOTS_FETCH_TIMEOUT = 5

_robots_cache: Dict[str, Tuple[bool, float]] = {}
_robots_lock = threading.Lock()


def is_valid_url(url: str) -> bool:
    """Check if URL has valid scheme and netloc."""
    if not url:
        return False
    try:
        parsed = urlparse(url)
        return parsed.scheme in ("http", "https") and bool(parsed.netloc)
    except (ValueError, AttributeError):
        return False


def is_allowed(url: str, run_id: Optional[str] = None) -> bool:
    logger = get_logger(__name__, run_id=run_id)
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    domain = parsed.netloc
    now = time.time()

    with _robots_lock:
        cached = _robots_cache.get(domain)
        if cached and (now - cached[1]) < ROBOTS_CACHE_TTL_SECONDS:
            return cached[0]

    rp = RobotFileParser()
    try:
        # Fetch robots.txt with timeout using requests instead of urllib
        # RobotFileParser.read() uses urllib which has no timeout by default
        resp = requests.get(
            robots_url,
            timeout=ROBOTS_FETCH_TIMEOUT,
            headers={"User-Agent": USER_AGENT},
        )
        if resp.status_code == 200:
            rp.parse(resp.text.splitlines())
        else:
            # No robots.txt or error - default to allowed
            logger.debug(
                "robots_not_found",
                extra={
                    "url": url,
                    "robots_url": robots_url,
                    "status": resp.status_code,
                },
            )
            with _robots_lock:
                _robots_cache[domain] = (True, now)
            return True

        allowed = rp.can_fetch(USER_AGENT, url)
        logger.debug(
            "robots_check",
            extra={"url": url, "robots_url": robots_url, "allowed": allowed},
        )
        with _robots_lock:
            _robots_cache[domain] = (allowed, now)
        return allowed
    except (
        requests.RequestException,
        OSError,
        IOError,
        ValueError,
        UnicodeDecodeError,
        RuntimeError,
    ) as exc:
        # Behavior on robots.txt fetch failure is configurable via ROBOTS_ON_ERROR
        default_allow = ROBOTS_ON_ERROR == "allow"
        logger.warning(
            "robots_check_failed",
            extra={
                "url": url,
                "error": str(exc),
                "defaulting_to": default_allow,
                "robots_on_error": ROBOTS_ON_ERROR,
            },
        )
        with _robots_lock:
            _robots_cache[domain] = (default_allow, now)
        return default_allow


# Backwards compatibility alias for explicit naming
is_allowed_to_fetch = is_allowed


def fetch_url(
    result: SearchResult, timeout: int = FETCH_TIMEOUT, run_id: Optional[str] = None
) -> Tuple[Optional[Document], Optional[str]]:
    logger = get_logger(__name__, run_id=run_id)

    # Validate URL before attempting fetch
    if not is_valid_url(result.url):
        logger.warning(
            "fetch_invalid_url",
            extra={"url": result.url[:200] if result.url else "None"},
        )
        return None, f"Invalid URL: {result.url[:100] if result.url else 'None'}"

    logger.info(
        "fetch_url_start",
        extra={"url": result.url, "source": result.source, "timeout": timeout},
    )

    # If result already has pre-fetched content (e.g., grounded search), use it directly
    if hasattr(result, "content") and result.content:
        logger.info(
            "fetch_prefetched_content",
            extra={"url": result.url, "chars": len(result.content)},
        )
        return (
            Document(
                url=result.url,
                title=result.title or "Source",
                content=result.content,
                media_type="text",
            ),
            None,
        )

    if not is_allowed(result.url, run_id=run_id):
        logger.warning("fetch_blocked_by_robots", extra={"url": result.url})
        return None, "Blocked by robots.txt"

    headers = {"User-Agent": USER_AGENT}
    logger.debug(
        "fetch_http_request",
        extra={"url": result.url, "user_agent": USER_AGENT},
    )
    resp = None
    for attempt in range(1, FETCH_RETRIES + 2):
        try:
            resp = requests.get(result.url, headers=headers, timeout=timeout)
            resp.raise_for_status()
            break
        except (requests.RequestException, OSError) as exc:
            logger.warning(
                "fetch_retry",
                extra={"url": result.url, "attempt": attempt, "error": str(exc)},
            )
            time.sleep(FETCH_BACKOFF_SECONDS * attempt)
    if resp is None or not resp.ok:
        logger.error(
            "fetch_failed",
            extra={
                "url": result.url,
                "status_code": resp.status_code if resp else None,
            },
        )
        return None, "Failed to fetch URL"

    content_type = resp.headers.get("content-type", "").lower()
    is_pdf = result.url.lower().endswith(".pdf") or "pdf" in content_type
    logger.debug(
        "fetch_content_type",
        extra={
            "url": result.url,
            "content_type": content_type,
            "is_pdf": is_pdf,
            "content_length": len(resp.content),
        },
    )

    if is_pdf:
        try:
            text = parse_pdf(resp.content)
            doc = Document(
                url=result.url,
                title=result.title or "PDF",
                content=text,
                media_type="pdf",
            )
            logger.info(
                "fetch_pdf_success", extra={"url": result.url, "chars": len(text)}
            )
            return doc, None
        except (ValueError, OSError, IOError):
            logger.exception("parse_pdf_failed", extra={"url": result.url})
            return None, "Failed to parse PDF"

    title, text = parse_html(resp.text)
    if not result.title:
        result.title = title

    # Filter out pages with too little meaningful content (likely boilerplate only)
    if len(text) < MIN_CONTENT_LENGTH:
        logger.warning(
            "fetch_content_too_short",
            extra={
                "url": result.url,
                "chars": len(text),
                "min_required": MIN_CONTENT_LENGTH,
            },
        )
        return None, f"Content too short ({len(text)} chars < {MIN_CONTENT_LENGTH})"

    logger.info(
        "fetch_html_success",
        extra={
            "url": result.url,
            "chars": len(text),
            "title": title[:50] if title else "no_title",
        },
    )
    return (
        Document(
            url=result.url,
            title=result.title or title or "Web Page",
            content=text,
            media_type="html",
        ),
        None,
    )


def fetch_documents_parallel(
    results: List[SearchResult],
    max_workers: int = FETCH_CONCURRENCY,
    timeout: int = FETCH_TIMEOUT,
    run_id: Optional[str] = None,
) -> Tuple[List[Document], List[str]]:
    """Fetch multiple results concurrently using a thread pool.

    If a SearchResult has pre-fetched content (e.g., from Tavily's raw_content),
    it will be used directly without re-fetching the URL.
    """

    logger = get_logger(__name__, run_id=run_id)
    if not results:
        return [], []

    documents: List[Document] = []
    warnings: List[str] = []
    results_to_fetch: List[SearchResult] = []

    # Use pre-fetched content when available (e.g., from Tavily raw_content)
    for res in results:
        if res.content and len(res.content) >= MIN_CONTENT_LENGTH:
            logger.info(
                "fetch_using_prefetched_content",
                extra={
                    "url": res.url,
                    "chars": len(res.content),
                    "source": res.source,
                },
            )
            documents.append(
                Document(
                    url=res.url,
                    title=res.title or "Web Page",
                    content=res.content,
                    media_type="html",
                    metadata={"prefetched": True, "source": res.source},
                )
            )
        else:
            results_to_fetch.append(res)

    if not results_to_fetch:
        logger.info(
            "fetch_all_prefetched",
            extra={"documents": len(documents)},
        )
        return documents, warnings

    logger.info(
        "fetch_parallel_start",
        extra={
            "count": len(results_to_fetch),
            "prefetched": len(documents),
            "max_workers": max_workers,
        },
    )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(fetch_url, res, timeout, run_id): res
            for res in results_to_fetch
        }
        for future in as_completed(future_map):
            res = future_map[future]
            try:
                doc, warn = future.result()
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception(
                    "fetch_parallel_exception",
                    extra={"url": res.url, "error": str(exc)},
                )
                warnings.append(f"{res.url}: parallel fetch error {exc}")
                continue
            if doc:
                documents.append(doc)
            if warn:
                warnings.append(f"{res.url}: {warn}")

    logger.info(
        "fetch_parallel_complete",
        extra={"documents": len(documents), "warnings": len(warnings)},
    )
    return documents, warnings
