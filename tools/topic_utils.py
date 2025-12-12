"""Shared topic detection utilities for query routing."""

from typing import Optional

from api.logging_setup import get_logger

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
        "machine",
        "learning",
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
    "sports": {
        "premier",
        "league",
        "table",
        "standing",
        "score",
        "match",
        "football",
        "soccer",
        "nba",
        "mlb",
        "nfl",
        "nhl",
        "fifa",
        "uefa",
        "champions",
        "world cup",
        "fixtures",
        "results",
    },
}


def detect_query_topic(query: str) -> Optional[str]:
    """Detect the likely topic category of a query."""
    logger = get_logger(__name__)
    query_lower = query.lower()
    query_words = set(query_lower.split())

    best_topic = None
    best_overlap = 0.0

    for topic, keywords in TOPIC_KEYWORDS.items():
        overlap = len(query_words & keywords)
        for keyword in keywords:
            if keyword in query_lower and len(keyword) > 3:
                overlap += 0.5

        if overlap > best_overlap:
            best_overlap = overlap
            best_topic = topic

    if best_topic and best_overlap >= 1:
        logger.info(
            "topic_detected",
            extra={
                "topic": best_topic,
                "query": query,
                "overlap": best_overlap,
            },
        )
        return best_topic
    return None
