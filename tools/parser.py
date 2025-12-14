"""Parsing helpers for HTML and PDF content."""

import io
import re
from typing import List, Optional, Tuple

import pdfplumber
from bs4 import BeautifulSoup, Tag

from api.logging_setup import get_logger


# Tags that typically contain navigation, boilerplate, or non-content elements
BOILERPLATE_TAGS = [
    "script",
    "style",
    "noscript",
    "nav",
    "header",
    "footer",
    "aside",
    "form",
    "iframe",
    "svg",
]

# Class/ID patterns that indicate non-content areas
BOILERPLATE_PATTERNS = [
    r"nav",
    r"menu",
    r"sidebar",
    r"footer",
    r"header",
    r"banner",
    r"advert",
    r"cookie",
    r"popup",
    r"modal",
    r"comment",
    r"share",
    r"social",
    r"related",
    r"recommend",
]


def clean_text(text: str) -> str:
    text = text.replace("\\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _is_boilerplate_element(tag: Tag) -> bool:
    """Check if an element is likely boilerplate based on class/id."""
    class_attr = tag.get("class")
    if class_attr is None:
        classes = ""
    elif isinstance(class_attr, list):
        classes = " ".join(str(c) for c in class_attr).lower()
    else:
        classes = str(class_attr).lower()
    tag_id = str(tag.get("id") or "").lower()
    combined = f"{classes} {tag_id}"
    return any(re.search(pattern, combined) for pattern in BOILERPLATE_PATTERNS)


def _extract_main_content(soup: BeautifulSoup) -> Optional[str]:
    """Try to extract main content from semantic HTML5 or common patterns."""
    # Priority 1: <main> tag
    main = soup.find("main")
    if main:
        return clean_text(main.get_text(separator=" "))

    # Priority 2: <article> tag
    article = soup.find("article")
    if article:
        return clean_text(article.get_text(separator=" "))

    # Priority 3: Common content div patterns
    content_patterns = [
        {"class_": re.compile(r"(article|content|post|entry|body)", re.I)},
        {"id": re.compile(r"(article|content|post|entry|main)", re.I)},
        {"role": "main"},
    ]
    for pattern in content_patterns:
        element = soup.find("div", **pattern)
        if element:
            text = clean_text(element.get_text(separator=" "))
            if len(text) > 200:  # Only use if substantial content
                return text

    return None


def parse_html(content: str) -> Tuple[str, str]:
    """Return (title, text) from HTML.

    Attempts to extract main content first, falling back to full page text
    with boilerplate removed if no semantic content container is found.
    """
    logger = get_logger(__name__)
    logger.debug("parse_html_start", extra={"content_length": len(content)})
    soup = BeautifulSoup(content, "html.parser")

    # Remove obvious boilerplate tags
    for tag in soup(BOILERPLATE_TAGS):
        tag.decompose()

    # Collect boilerplate elements first, then decompose
    # (decomposing during iteration can invalidate subsequent tags)
    boilerplate_elements = [
        tag
        for tag in soup.find_all(True)
        if hasattr(tag, "attrs")
        and tag.attrs is not None
        and _is_boilerplate_element(tag)
    ]
    for tag in boilerplate_elements:
        tag.decompose()

    title = soup.title.string.strip() if soup.title and soup.title.string else ""

    # Try to extract main content from semantic elements
    main_content = _extract_main_content(soup)
    if main_content and len(main_content) > 200:
        logger.debug(
            "parse_html_main_content_extracted",
            extra={
                "title": title[:50] if title else "no_title",
                "text_length": len(main_content),
            },
        )
        return title, main_content

    # Fallback: get all remaining text (boilerplate already removed)
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
