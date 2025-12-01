"""Parsing helpers for HTML and PDF content."""

import io
import re
from typing import Tuple

import pdfplumber
from bs4 import BeautifulSoup

from api.logging_setup import get_logger


def clean_text(text: str) -> str:
    text = text.replace("\\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_html(content: str) -> Tuple[str, str]:
    """Return (title, text) from HTML."""
    logger = get_logger(__name__)
    logger.debug("parse_html_start", extra={"content_length": len(content)})
    soup = BeautifulSoup(content, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    title = soup.title.string.strip() if soup.title and soup.title.string else ""
    text = clean_text(soup.get_text(separator=" "))
    logger.debug(
        "parse_html_complete",
        extra={"title": title[:50] if title else "no_title", "text_length": len(text)},
    )
    return title, text


def parse_pdf(content: bytes) -> str:
    """Extract text from a PDF byte stream."""
    logger = get_logger(__name__)
    logger.debug("parse_pdf_start", extra={"content_bytes": len(content)})
    with pdfplumber.open(io.BytesIO(content)) as pdf:
        page_count = len(pdf.pages)
        pages = [page.extract_text() or "" for page in pdf.pages]
    text = clean_text(" ".join(pages))
    logger.debug(
        "parse_pdf_complete",
        extra={"pages": page_count, "text_length": len(text)},
    )
    return text
