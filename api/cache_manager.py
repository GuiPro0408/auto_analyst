"""Cache management utilities extracted from graph.py."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

from api.cache import (
    CACHE_SCHEMA_VERSION,
    QueryCache,
    decode_research_state,
    encode_research_state,
)
from api.logging_setup import get_logger

if TYPE_CHECKING:
    from api.state import ResearchState


class CacheManager:
    """Manages query caching for research pipeline results."""

    def __init__(
        self,
        db_path: Path,
        ttl_seconds: int,
        run_id: Optional[str] = None,
    ) -> None:
        """Initialize the cache manager.

        Args:
            db_path: Path to the SQLite cache database.
            ttl_seconds: Time-to-live for cache entries in seconds.
            run_id: Optional run ID for logging correlation.
        """
        self._cache = QueryCache(db_path, ttl_seconds)
        self._run_id = run_id
        self._logger = get_logger(__name__, run_id=run_id)

    def get_cached_result(
        self,
        query: str,
        skip_grounded: bool = True,
    ) -> Optional["ResearchState"]:
        """Attempt to retrieve a cached result for the query.

        Args:
            query: The search query to look up.
            skip_grounded: Whether to skip grounded cached results.

        Returns:
            Cached ResearchState if valid cache hit, None otherwise.
        """
        try:
            cached_payload = self._cache.get(query)
        except Exception as exc:
            self._logger.warning("cache_get_failed", extra={"error": str(exc)})
            return None

        if not cached_payload:
            return None

        cached_state = decode_research_state(cached_payload)

        # Check if cached result should be skipped
        is_fallback = cached_state.draft_answer.startswith(
            "No sufficient context"
        ) or cached_state.verified_answer.startswith("No context retrieved")

        is_grounded = any(
            getattr(sr, "source", None) == "gemini_grounding"
            for sr in cached_state.search_results
        )

        if cached_state.time_sensitive:
            self._logger.info(
                "cache_skip_time_sensitive",
                extra={"query": query},
            )
            return None

        if is_fallback:
            self._logger.info(
                "cache_skip_low_context",
                extra={"query": query},
            )
            return None

        if skip_grounded and is_grounded:
            self._logger.info(
                "cache_skip_grounded",
                extra={"query": query},
            )
            return None

        self._logger.info(
            "cache_hit",
            extra={"query": query, "run_id": cached_state.run_id},
        )

        # Add cache notice to warnings
        cached_state.warnings = [
            *cached_state.warnings,
            "Response served from cache; rerun with a rephrase to refresh.",
        ]

        return cached_state

    def save_result(self, query: str, state: "ResearchState") -> None:
        """Save a research result to cache.

        Args:
            query: The search query.
            state: The ResearchState to cache.
        """
        if state.time_sensitive:
            self._logger.debug(
                "cache_skip_save_time_sensitive",
                extra={"query": query},
            )
            return

        if state.errors:
            self._logger.debug(
                "cache_skip_save_errors",
                extra={"query": query, "error_count": len(state.errors)},
            )
            return

        payload = {
            "version": CACHE_SCHEMA_VERSION,
            "state": encode_research_state(state),
        }

        try:
            self._cache.set(query, payload)
            self._logger.debug("cache_saved", extra={"query": query})
        except Exception as exc:
            self._logger.warning("cache_set_failed", extra={"error": str(exc)})

    def purge(self) -> None:
        """Clear all cached entries."""
        self._cache.purge()
        self._logger.info("cache_purged")
