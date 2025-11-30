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


def is_allowed(url: str) -> bool:
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    rp = RobotFileParser()
    try:
        rp.set_url(robots_url)
        rp.read()
        return rp.can_fetch(USER_AGENT, url)
    except Exception:
        # If robots cannot be fetched, err on the side of fetching since sources are public.
        return True


def fetch_url(
    result: SearchResult, timeout: int = 15, run_id: Optional[str] = None
) -> Tuple[Optional[Document], Optional[str]]:
    logger = get_logger(__name__, run_id=run_id)
    if not is_allowed(result.url):
        logger.warning("fetch_blocked_by_robots", extra={"url": result.url})
        return None, "Blocked by robots.txt"

    headers = {"User-Agent": USER_AGENT}
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
        logger.error("fetch_failed", extra={"url": result.url})
        return None, "Failed to fetch URL"

    content_type = resp.headers.get("content-type", "").lower()
    is_pdf = result.url.lower().endswith(".pdf") or "pdf" in content_type

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
    logger.info("fetch_html_success", extra={"url": result.url, "chars": len(text)})
    return (
        Document(
            url=result.url,
            title=result.title or title or "Web Page",
            content=text,
            media_type="html",
        ),
        None,
    )
