"""Query classification for hybrid RAG/direct LLM routing.

Classifies queries into types that determine how answers are generated:
- factual: Strict RAG with citations (news, research, technical)
- recommendation: LLM knowledge + RAG context (suggestions, opinions)
- creative: Primarily LLM knowledge (brainstorming, explanations)
"""

import re
from typing import Optional

from api.logging_setup import get_logger

# Query type constants
QUERY_TYPE_FACTUAL = "factual"
QUERY_TYPE_RECOMMENDATION = "recommendation"
QUERY_TYPE_CREATIVE = "creative"

# Patterns that indicate recommendation/subjective queries
RECOMMENDATION_PATTERNS = {
    # Direct recommendation requests
    r"\brecommend\w*\b",
    r"\bsuggest\w*\b",
    r"\bshould\s+i\b",
    r"\bwhat\s+(?:anime|movie|show|game|book|series)\s+(?:should|to|can)\b",
    r"\bbest\s+(?:anime|movie|show|game|book|series)\b",
    r"\btop\s+\d*\s*(?:anime|movie|show|game|book|series)\b",
    r"\blike\s+(?:attack on titan|naruto|one piece|demon slayer)\b",
    r"\bsimilar\s+to\b",
    r"\balternatives?\s+to\b",
    r"\bwhat\s+(?:are|is)\s+(?:some|good|the best)\b",
    # Opinion/preference queries
    r"\bfavorite\b",
    r"\bworth\s+(?:watching|reading|playing|buying)\b",
    r"\bgood\s+(?:anime|movie|show|game|book|series)\b",
    r"\bwhat\s+do\s+you\s+think\b",
    r"\byour\s+(?:opinion|thoughts)\b",
}

# Patterns that indicate factual/research queries
FACTUAL_PATTERNS = {
    # News and current events
    r"\bnews\b",
    r"\bwhat\s+happened\b",
    r"\bwhen\s+(?:did|was|is)\b",
    r"\bwho\s+(?:is|was|won|lost)\b",
    r"\bhow\s+(?:much|many|long)\b",
    r"\bstatistics?\b",
    r"\bdata\b",
    r"\bfacts?\s+about\b",
    # Technical/research
    r"\bhow\s+(?:does|do|to)\b",
    r"\bwhat\s+(?:is|are)\s+the\s+(?:difference|cause|reason|effect)\b",
    r"\bexplain\b",
    r"\bdefin(?:e|ition)\b",
    r"\bresearch\b",
    r"\bstudy\b",
    r"\bscientific\b",
    # Verification
    r"\bis\s+it\s+true\b",
    r"\bverify\b",
    r"\bfact\s*check\b",
    # Specific lookups
    r"\bprice\s+of\b",
    r"\brelease\s+date\b",
    r"\bschedule\b",
    r"\bstandings\b",
    r"\bresults?\b",
    r"\bscore\b",
}

# Patterns that indicate creative/open-ended queries
CREATIVE_PATTERNS = {
    r"\bwrite\s+(?:a|me)\b",
    r"\bcreate\b",
    r"\bgenerate\b",
    r"\bbrainstorm\b",
    r"\bideas?\s+for\b",
    r"\bhelp\s+me\s+(?:understand|think)\b",
    r"\bwhat\s+if\b",
    r"\bimagine\b",
}

# Entertainment topics that often need recommendations
ENTERTAINMENT_KEYWORDS = {
    "anime",
    "manga",
    "movie",
    "movies",
    "film",
    "films",
    "show",
    "shows",
    "series",
    "tv",
    "game",
    "games",
    "book",
    "books",
    "novel",
    "novels",
    "music",
    "song",
    "songs",
    "album",
    "podcast",
}


def _matches_patterns(query: str, patterns: set) -> bool:
    """Check if query matches any pattern in the set."""
    query_lower = query.lower()
    for pattern in patterns:
        if re.search(pattern, query_lower):
            return True
    return False


def _has_entertainment_topic(query: str) -> bool:
    """Check if query contains entertainment-related keywords."""
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in ENTERTAINMENT_KEYWORDS)


def classify_query(
    query: str,
    run_id: Optional[str] = None,
) -> str:
    """Classify a query into factual, recommendation, or creative type.

    Args:
        query: The user's search query.
        run_id: Optional run ID for logging.

    Returns:
        One of: "factual", "recommendation", "creative"
    """
    logger = get_logger(__name__, run_id=run_id)

    query_lower = query.lower().strip()

    # Check creative patterns first (explicit content creation requests)
    if _matches_patterns(query, CREATIVE_PATTERNS):
        logger.info(
            "query_classified",
            extra={"query_type": QUERY_TYPE_CREATIVE, "reason": "creative_pattern"},
        )
        return QUERY_TYPE_CREATIVE

    # Check recommendation patterns
    if _matches_patterns(query, RECOMMENDATION_PATTERNS):
        logger.info(
            "query_classified",
            extra={
                "query_type": QUERY_TYPE_RECOMMENDATION,
                "reason": "recommendation_pattern",
            },
        )
        return QUERY_TYPE_RECOMMENDATION

    # Check factual patterns
    if _matches_patterns(query, FACTUAL_PATTERNS):
        logger.info(
            "query_classified",
            extra={"query_type": QUERY_TYPE_FACTUAL, "reason": "factual_pattern"},
        )
        return QUERY_TYPE_FACTUAL

    # Heuristic: Entertainment topic without explicit factual markers
    # tends to be recommendation-seeking
    if _has_entertainment_topic(query_lower):
        # Check if it's asking for specific info (release date, cast, etc.)
        specific_info_patterns = {
            r"\brelease\b",
            r"\bcast\b",
            r"\bdirector\b",
            r"\bplot\b",
            r"\bending\b",
            r"\bepisode\s+\d+\b",
            r"\bseason\s+\d+\b",
            r"\bcharacter\b",
            r"\bauthor\b",
            r"\bcreator\b",
        }
        if _matches_patterns(query, specific_info_patterns):
            logger.info(
                "query_classified",
                extra={
                    "query_type": QUERY_TYPE_FACTUAL,
                    "reason": "entertainment_specific_info",
                },
            )
            return QUERY_TYPE_FACTUAL

        # Generic entertainment query -> likely wants recommendations
        logger.info(
            "query_classified",
            extra={
                "query_type": QUERY_TYPE_RECOMMENDATION,
                "reason": "entertainment_generic",
            },
        )
        return QUERY_TYPE_RECOMMENDATION

    # Default to factual for research-oriented tool
    logger.info(
        "query_classified",
        extra={"query_type": QUERY_TYPE_FACTUAL, "reason": "default"},
    )
    return QUERY_TYPE_FACTUAL


def get_query_type_description(query_type: str) -> str:
    """Get a human-readable description of query type behavior."""
    descriptions = {
        QUERY_TYPE_FACTUAL: "Research mode: Answers based strictly on retrieved sources with citations.",
        QUERY_TYPE_RECOMMENDATION: "Recommendation mode: Combining web sources with AI knowledge for suggestions.",
        QUERY_TYPE_CREATIVE: "Creative mode: AI-generated response with optional source support.",
    }
    return descriptions.get(query_type, "Unknown mode")


__all__ = [
    "classify_query",
    "get_query_type_description",
    "QUERY_TYPE_FACTUAL",
    "QUERY_TYPE_RECOMMENDATION",
    "QUERY_TYPE_CREATIVE",
]
