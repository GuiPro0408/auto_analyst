"""Parsing helpers for HTML and PDF content."""

import io
import re
from typing import Tuple

import pdfplumber
from bs4 import BeautifulSoup


def clean_text(text: str) -> str:
    text = text.replace("\\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_html(content: str) -> Tuple[str, str]:
    """Return (title, text) from HTML."""
    soup = BeautifulSoup(content, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    title = soup.title.string.strip() if soup.title and soup.title.string else ""
    text = clean_text(soup.get_text(separator=" "))
    return title, text


def parse_pdf(content: bytes) -> str:
    """Extract text from a PDF byte stream."""
    with pdfplumber.open(io.BytesIO(content)) as pdf:
        pages = [page.extract_text() or "" for page in pdf.pages]
    return clean_text(" ".join(pages))
