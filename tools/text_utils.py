"""Shared text utilities for keyword extraction and heuristics."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional, Set, Tuple

from api.logging_setup import get_logger


def _load_stopwords() -> Set[str]:
    """Load stopwords from data/stopwords.json, with inline fallback."""
    logger = get_logger(__name__)
    data_path = Path(__file__).parent.parent / "data" / "stopwords.json"

    try:
        if data_path.exists():
            with open(data_path, encoding="utf-8") as f:
                data = json.load(f)
                words = set(data.get("stopwords", []))
                logger.debug(
                    "stopwords_loaded",
                    extra={"path": str(data_path), "count": len(words)},
                )
                return words
    except (OSError, IOError, json.JSONDecodeError, TypeError) as exc:
        logger.warning(
            "stopwords_load_failed", extra={"error": str(exc), "path": str(data_path)}
        )

    # Inline fallback if file doesn't exist or fails to load
    return {
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
        "these",
        "those",
        "do",
        "does",
        "can",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
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
        "but",
        "if",
        "then",
        "so",
        "because",
        "as",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "under",
        "again",
        "further",
        "once",
        "here",
        "there",
        "all",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "than",
        "too",
        "very",
        "just",
        "now",
        "any",
        "me",
        "my",
        "i",
        "you",
        "your",
        "we",
        "our",
        "they",
        "their",
        "it",
        "its",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "having",
    }


# Consolidated stopwords used across planner, search, and generator
STOPWORDS: Set[str] = _load_stopwords()

# Temporal keywords that indicate time-sensitive queries
TEMPORAL_KEYWORDS: Set[str] = {
    "upcoming",
    "coming",
    "next",
    "new",
    "latest",
    "recent",
    "current",
    "this season",
    "this year",
    "this month",
    "this week",
    "today",
    "release",
    "releases",
    "premiere",
    "premieres",
    "launch",
    "launches",
    "announced",
    "announcement",
    "schedule",
    "scheduled",
    "2024",
    "2025",
    "2026",
}

# Triggers for structured answers (merged from generator + quality control)
STRUCTURED_TRIGGERS: Set[str] = {
    "list",
    "releases",
    "releasing",
    "release",
    "lineup",
    "schedule",
    "standings",
    "ranking",
    "table",
    "top",
    "currently",
    "current",
    "today",
    "upcoming",
    "fall",
    "spring",
    "summer",
    "winter",
}


def extract_keywords(
    text: str,
    *,
    stopwords: Optional[Set[str]] = None,
    min_len: int = 3,
    strip_punct: bool = True,
) -> Set[str]:
    """Extract meaningful keywords from text."""
    if not text:
        return set()
    tokens = text.lower().split()
    cleaned = []
    for tok in tokens:
        if strip_punct:
            tok = tok.strip(".,!?;:")
        tok = re.sub(r"[^a-z0-9_-]", "", tok)
        if len(tok) >= min_len:
            cleaned.append(tok)
    stop = stopwords or set()
    return {w for w in cleaned if w and w not in stop}


def detect_time_sensitive(query: str) -> Tuple[bool, list[str]]:
    """Detect if a query is time-sensitive based on temporal keywords."""
    query_lower = query.lower()
    matched = [kw for kw in TEMPORAL_KEYWORDS if kw in query_lower]
    return len(matched) > 0, matched


def requires_structured_list(query: str) -> bool:
    """Detect queries that expect list- or ranking-style answers."""
    q = query.lower()
    return any(trigger in q for trigger in STRUCTURED_TRIGGERS)
