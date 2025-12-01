"""Fetcher that retrieves web pages and PDFs, respecting robots.txt."""

from typing import Optional, Tuple
import time
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests

from api.config import USER_AGENT, FETCH_BACKOFF_SECONDS, FETCH_RETRIES
from api.logging_setup import get_logger
from api.state import Document, SearchResult
from tools.parser import parse_html, parse_pdf


def is_allowed(url: str, run_id: Optional[str] = None) -> bool:
    logger = get_logger(__name__, run_id=run_id)
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    rp = RobotFileParser()
    try:
        rp.set_url(robots_url)
        rp.read()
        allowed = rp.can_fetch(USER_AGENT, url)
        logger.debug(
            "robots_check",
            extra={"url": url, "robots_url": robots_url, "allowed": allowed},
        )
        return allowed
    except Exception as exc:
        # If robots cannot be fetched, err on the side of fetching since sources are public.
        logger.debug(
            "robots_check_failed",
            extra={"url": url, "error": str(exc), "defaulting_to": True},
        )
        return True


def fetch_url(
    result: SearchResult, timeout: int = 15, run_id: Optional[str] = None
) -> Tuple[Optional[Document], Optional[str]]:
    logger = get_logger(__name__, run_id=run_id)
    logger.info(
        "fetch_url_start",
        extra={"url": result.url, "source": result.source, "timeout": timeout},
    )

    # If result already has pre-fetched content (e.g., Wikipedia), use it directly
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
        except Exception as exc:
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
        except Exception:
            logger.exception("parse_pdf_failed", extra={"url": result.url})
            return None, "Failed to parse PDF"

    title, text = parse_html(resp.text)
    if not result.title:
        result.title = title
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
