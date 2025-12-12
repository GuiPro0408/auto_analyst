"""Gemini Google Search Grounding integration for web-augmented responses.

Uses the new google-genai SDK with google_search tool (Gemini 2.0+).
See: https://ai.google.dev/gemini-api/docs/google-search
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from api.config import GEMINI_API_KEYS, GEMINI_MODEL
from api.key_rotator import APIKeyRotator, get_default_rotator
from api.logging_setup import get_logger

# Re-export for backward compatibility with tests
GEMINI_API_KEY = GEMINI_API_KEYS[0] if GEMINI_API_KEYS else ""

# Retry configuration
MAX_RETRIES_PER_KEY = 1
BASE_BACKOFF_SECONDS = 1.0
MAX_BACKOFF_SECONDS = 8.0


@dataclass
class GroundingSource:
    """A source extracted from Gemini grounding metadata."""

    url: str
    title: str
    snippet: str = ""


@dataclass
class GroundingResult:
    """Result from a grounded Gemini query."""

    answer: str
    sources: List[GroundingSource] = field(default_factory=list)
    success: bool = True
    error: Optional[str] = None
    web_search_queries: List[str] = field(default_factory=list)


def _extract_grounding_sources(
    grounding_metadata: Any,
    run_id: Optional[str] = None,
) -> tuple[List[GroundingSource], List[str]]:
    """Extract sources and search queries from Gemini grounding metadata.

    New google-genai SDK returns grounding_metadata with:
    - grounding_chunks: list of {web: {uri, title}}
    - web_search_queries: list of search queries used
    - grounding_supports: list of support objects with segments

    Args:
        grounding_metadata: The grounding_metadata from Gemini response.
        run_id: Optional run ID for logging correlation.

    Returns:
        Tuple of (sources list, web_search_queries list).
    """
    logger = get_logger(__name__, run_id=run_id)
    sources: List[GroundingSource] = []
    web_search_queries: List[str] = []

    if not grounding_metadata:
        logger.debug("grounding_no_metadata")
        return sources, web_search_queries

    # Extract grounding chunks (citations) - new SDK structure
    grounding_chunks = getattr(grounding_metadata, "grounding_chunks", None) or []
    for chunk in grounding_chunks:
        web = getattr(chunk, "web", None)
        if web:
            url = getattr(web, "uri", "") or ""
            title = getattr(web, "title", "") or ""
            if url:
                sources.append(GroundingSource(url=url, title=title, snippet=""))
                logger.debug(
                    "grounding_source_extracted",
                    extra={"url": url, "title": title[:50] if title else ""},
                )

    # Extract web search queries used by Gemini
    search_queries = getattr(grounding_metadata, "web_search_queries", None) or []
    web_search_queries = list(search_queries)

    # Also try to get search entry point for additional context
    search_entry = getattr(grounding_metadata, "search_entry_point", None)
    if search_entry:
        rendered_content = getattr(search_entry, "rendered_content", None)
        if rendered_content:
            logger.debug(
                "grounding_search_entry_found",
                extra={"content_length": len(rendered_content)},
            )

    logger.debug(
        "grounding_extraction_complete",
        extra={
            "sources_count": len(sources),
            "search_queries": web_search_queries,
        },
    )
    return sources, web_search_queries


def query_with_grounding(
    query: str,
    run_id: Optional[str] = None,
    model_name: Optional[str] = None,
    key_rotator: Optional[APIKeyRotator] = None,
) -> GroundingResult:
    """Query Gemini with Google Search grounding enabled.

    Uses the new google-genai SDK with google_search tool (Gemini 2.0+).
    Implements exponential backoff on rate limit errors.

    Args:
        query: The user's question.
        run_id: Optional run ID for logging correlation.
        model_name: Optional Gemini model override.
        key_rotator: Optional APIKeyRotator instance (uses shared default if None).

    Returns:
        GroundingResult with answer, sources, and metadata.
    """
    logger = get_logger(__name__, run_id=run_id)
    rotator = key_rotator or get_default_rotator(GEMINI_API_KEYS)
    logger.info(
        "grounding_query_start",
        extra={
            "query": query[:100],
            "total_api_keys": rotator.total_keys,
        },
    )

    if rotator.total_keys == 0:
        logger.error("grounding_no_api_key")
        return GroundingResult(
            answer="",
            success=False,
            error="GOOGLE_API_KEY or GOOGLE_API_KEYS not configured",
        )

    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        logger.error("grounding_import_failed", extra={"error": str(exc)})
        return GroundingResult(
            answer="",
            success=False,
            error="google-genai package not installed. Run: pip install google-genai",
        )

    # Get current API key from rotator
    current_api_key = rotator.current_key
    client = genai.Client(api_key=current_api_key)

    # Configure Google Search grounding tool (new API for Gemini 2.0+)
    try:
        google_search_tool = types.Tool(google_search=types.GoogleSearch())
        logger.debug("grounding_tool_created")
    except Exception as exc:
        logger.error("grounding_tool_creation_failed", extra={"error": str(exc)})
        return GroundingResult(
            answer="",
            success=False,
            error=f"Failed to create grounding tool: {exc}",
        )

    last_error: Optional[Exception] = None
    backoff = BASE_BACKOFF_SECONDS
    target_model = model_name or GEMINI_MODEL
    total_attempts = MAX_RETRIES_PER_KEY * max(rotator.total_keys, 1)
    attempt = 0

    while attempt < total_attempts:
        attempt += 1
        current_api_key = rotator.current_key
        try:
            # Recreate client with current key (may have rotated)
            client = genai.Client(api_key=current_api_key)
            logger.debug(
                "grounding_attempt",
                extra={
                    "attempt": attempt,
                    "total_attempts": total_attempts,
                    "model": target_model,
                    "api_key_suffix": (
                        current_api_key[-8:] if current_api_key else "none"
                    ),
                    "available_keys": rotator.available_keys,
                },
            )

            # Use the new Client API with GenerateContentConfig
            response = client.models.generate_content(
                model=target_model,
                contents=query,
                config=types.GenerateContentConfig(
                    tools=[google_search_tool],
                    temperature=0.4,
                    max_output_tokens=1024,
                ),
            )

            # Extract answer text - new SDK structure
            answer_text = ""
            if response.candidates:
                candidate = response.candidates[0]
                if candidate.content and candidate.content.parts:
                    answer_text = "".join(
                        part.text
                        for part in candidate.content.parts
                        if hasattr(part, "text") and part.text
                    )

                # Extract grounding metadata
                grounding_metadata = getattr(candidate, "grounding_metadata", None)
                sources, web_queries = _extract_grounding_sources(
                    grounding_metadata, run_id=run_id
                )

                # Success - reset rate limit tracking
                rotator.reset()
                logger.info(
                    "grounding_query_success",
                    extra={
                        "answer_length": len(answer_text),
                        "sources_count": len(sources),
                        "web_queries": web_queries,
                        "attempt": attempt,
                        "api_key_suffix": (
                            current_api_key[-8:] if current_api_key else "none"
                        ),
                    },
                )

                return GroundingResult(
                    answer=answer_text.strip(),
                    sources=sources,
                    success=True,
                    web_search_queries=web_queries,
                )

            # No candidates returned
            logger.warning("grounding_no_candidates")
            return GroundingResult(
                answer="",
                success=False,
                error="No response candidates from Gemini",
            )

        except Exception as exc:
            error_str = str(exc)
            error_type = type(exc).__name__

            # Check for rate limit errors using shared rotator method
            if rotator.is_rate_limit_error(exc):
                last_error = exc
                # Try rotating to another API key
                has_more_keys = rotator.mark_rate_limited(current_api_key or "")
                logger.warning(
                    "grounding_rate_limited",
                    extra={
                        "attempt": attempt,
                        "backoff_seconds": backoff,
                        "error": error_str,
                        "api_key_suffix": (
                            current_api_key[-8:] if current_api_key else "none"
                        ),
                        "rotated_to_new_key": has_more_keys,
                        "available_keys": rotator.available_keys,
                    },
                )
                if not has_more_keys:
                    logger.warning(
                        "grounding_rate_limited_exhausted_keys",
                        extra={"attempt": attempt, "total_attempts": total_attempts},
                    )
                    return GroundingResult(
                        answer="",
                        success=False,
                        error="All Gemini API keys rate limited; skipping grounding.",
                    )
                if attempt < total_attempts:
                    time.sleep(0.5)
                continue

            # Check for transient errors using shared rotator method
            if rotator.is_transient_error(exc):
                last_error = exc
                logger.warning(
                    "grounding_transient_error",
                    extra={
                        "attempt": attempt,
                        "backoff_seconds": backoff,
                        "error": error_str,
                        "error_type": error_type,
                    },
                )
                if attempt < total_attempts:
                    time.sleep(backoff)
                    backoff = min(backoff * 2, MAX_BACKOFF_SECONDS)
                continue

            # Non-retriable error
            logger.exception(
                "grounding_unexpected_error",
                extra={"attempt": attempt, "error": error_str},
            )
            return GroundingResult(
                answer="",
                success=False,
                error=f"Unexpected error: {exc}",
            )

    # All retries exhausted
    error_msg = f"Grounding failed after {attempt} attempts across {rotator.total_keys} API key(s): {last_error}"
    logger.error(
        "grounding_all_retries_exhausted",
        extra={
            "error": str(last_error),
            "attempts": attempt,
            "total_keys": rotator.total_keys,
        },
    )
    return GroundingResult(
        answer="",
        success=False,
        error=error_msg,
    )


def grounding_sources_to_chunks(
    sources: List[GroundingSource],
    run_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Convert grounding sources to chunk-compatible dicts for the pipeline.

    Args:
        sources: List of GroundingSource from grounding result.
        run_id: Optional run ID for logging.

    Returns:
        List of dicts compatible with Chunk dataclass construction.
    """
    logger = get_logger(__name__, run_id=run_id)
    chunks = []

    for idx, source in enumerate(sources):
        chunk_id = f"grounding_{idx}_{hash(source.url) % 10000}"
        text = source.title if source.title else source.url

        chunks.append(
            {
                "id": chunk_id,
                "text": text,
                "metadata": {
                    "url": source.url,
                    "title": source.title or "Web Source",
                    "source": "gemini_grounding",
                    "media_type": "text",
                },
            }
        )

    logger.debug(
        "grounding_chunks_created",
        extra={"chunks_count": len(chunks)},
    )
    return chunks
