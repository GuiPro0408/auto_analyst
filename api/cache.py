"""Persistent query caching for pipeline results."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from api.logging_setup import get_logger
from api.state import (
    Chunk,
    ConversationTurn,
    Document,
    ResearchState,
    SearchQuery,
    SearchResult,
)

CACHE_SCHEMA_VERSION = 1


def _now() -> float:
    return time.time()


def _as_chunk(data: Dict[str, Any]) -> Chunk:
    return Chunk(
        id=data.get("id", ""),
        text=data.get("text", ""),
        metadata=data.get("metadata", {}),
    )


def _as_document(data: Dict[str, Any]) -> Document:
    return Document(
        url=data.get("url", ""),
        title=data.get("title", ""),
        content=data.get("content", ""),
        media_type=data.get("media_type", "text"),
        metadata=data.get("metadata", {}),
    )


def _as_search_result(data: Dict[str, Any]) -> SearchResult:
    return SearchResult(
        url=data.get("url", ""),
        title=data.get("title", ""),
        snippet=data.get("snippet", ""),
        source=data.get("source", "web"),
        content=data.get("content", ""),
    )


def _as_search_query(data: Dict[str, Any]) -> SearchQuery:
    return SearchQuery(
        text=data.get("text", ""),
        rationale=data.get("rationale", ""),
        topic=data.get("topic", ""),
        preferred_domains=data.get("preferred_domains", []),
    )


def _as_conversation_turn(data: Dict[str, Any]) -> ConversationTurn:
    return ConversationTurn.from_dict(data)


def encode_research_state(state: ResearchState) -> Dict[str, Any]:
    """Convert ResearchState dataclass into a JSON-serializable dict."""
    return asdict(state)


def decode_research_state(payload: Dict[str, Any]) -> ResearchState:
    """Rehydrate ResearchState from cached payload."""
    state_payload = payload.get("state", payload)
    return ResearchState(
        query=state_payload.get("query", ""),
        run_id=state_payload.get("run_id", ""),
        plan=[_as_search_query(item) for item in state_payload.get("plan", [])],
        search_results=[
            _as_search_result(item) for item in state_payload.get("search_results", [])
        ],
        documents=[_as_document(item) for item in state_payload.get("documents", [])],
        chunks=[_as_chunk(item) for item in state_payload.get("chunks", [])],
        retrieved=[_as_chunk(item) for item in state_payload.get("retrieved", [])],
        retrieval_scores=state_payload.get("retrieval_scores", []),
        draft_answer=state_payload.get("draft_answer", ""),
        verified_answer=state_payload.get("verified_answer", ""),
        citations=state_payload.get("citations", []),
        errors=state_payload.get("errors", []),
        warnings=state_payload.get("warnings", []),
        adaptive_iterations=state_payload.get("adaptive_iterations", 0),
        qc_passes=state_payload.get("qc_passes", 0),
        qc_notes=state_payload.get("qc_notes", []),
        time_sensitive=state_payload.get("time_sensitive", False),
        conversation_history=[
            _as_conversation_turn(item)
            for item in state_payload.get("conversation_history", [])
        ],
        grounded_answer=state_payload.get("grounded_answer", ""),
        grounded_sources=[
            _as_chunk(item) for item in state_payload.get("grounded_sources", [])
        ],
    )


class QueryCache:
    """Simple SQLite-backed cache for query results with LRU eviction."""

    def __init__(
        self,
        db_path: Path,
        ttl_seconds: int = 3600,
        max_entries: Optional[int] = None,
    ) -> None:
        """Initialize the cache.

        Args:
            db_path: Path to SQLite database file.
            ttl_seconds: Time-to-live for cache entries (0 = no expiry).
            max_entries: Maximum number of entries to keep (None = unlimited).
        """
        from api.config import CACHE_MAX_ENTRIES

        self.db_path = Path(db_path)
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries if max_entries is not None else CACHE_MAX_ENTRIES
        self.logger = get_logger(__name__)
        self._lock = threading.Lock()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS query_cache (
                    cache_key TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    accessed_at REAL NOT NULL
                )
                """
            )
            # Add accessed_at column if it doesn't exist (migration)
            try:
                conn.execute("SELECT accessed_at FROM query_cache LIMIT 1")
            except sqlite3.OperationalError:
                conn.execute(
                    "ALTER TABLE query_cache ADD COLUMN accessed_at REAL DEFAULT 0"
                )
            conn.commit()

    @staticmethod
    def _hash_query(query: str) -> str:
        normalized = query.strip().lower().encode("utf-8")
        return hashlib.sha256(normalized).hexdigest()

    def _evict_if_needed(self, conn: sqlite3.Connection) -> None:
        """Evict oldest entries if cache exceeds max size."""
        if self.max_entries <= 0:
            return

        count = conn.execute("SELECT COUNT(*) FROM query_cache").fetchone()[0]
        if count >= self.max_entries:
            # Remove oldest 10% of entries by access time (LRU)
            to_remove = max(1, int(self.max_entries * 0.1))
            conn.execute(
                """
                DELETE FROM query_cache WHERE cache_key IN (
                    SELECT cache_key FROM query_cache
                    ORDER BY accessed_at ASC
                    LIMIT ?
                )
                """,
                (to_remove,),
            )
            self.logger.debug(
                "cache_evicted",
                extra={"evicted": to_remove, "total_before": count},
            )

    def get(self, query: str) -> Optional[Dict[str, Any]]:
        cache_key = self._hash_query(query)
        with self._lock, sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT payload, created_at FROM query_cache WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
            if not row:
                return None
            payload_raw, created_at = row
            if self.ttl_seconds > 0 and (_now() - created_at) > self.ttl_seconds:
                conn.execute(
                    "DELETE FROM query_cache WHERE cache_key = ?", (cache_key,)
                )
                conn.commit()
                self.logger.debug("query_cache_expired", extra={"cache_key": cache_key})
                return None

            # Update access time for LRU tracking
            conn.execute(
                "UPDATE query_cache SET accessed_at = ? WHERE cache_key = ?",
                (_now(), cache_key),
            )
            conn.commit()

        try:
            payload = json.loads(payload_raw)
        except json.JSONDecodeError:
            self.logger.warning("query_cache_corrupt", extra={"cache_key": cache_key})
            self.delete(query)
            return None

        if payload.get("version") != CACHE_SCHEMA_VERSION:
            self.logger.debug(
                "query_cache_version_mismatch",
                extra={"cache_key": cache_key, "version": payload.get("version")},
            )
            self.delete(query)
            return None

        return payload

    def set(self, query: str, payload: Dict[str, Any]) -> None:
        cache_key = self._hash_query(query)
        encoded = json.dumps(payload)
        now = _now()
        with self._lock, sqlite3.connect(self.db_path) as conn:
            self._evict_if_needed(conn)
            conn.execute(
                "REPLACE INTO query_cache (cache_key, payload, created_at, accessed_at) VALUES (?, ?, ?, ?)",
                (cache_key, encoded, now, now),
            )
            conn.commit()
        self.logger.debug("query_cache_set", extra={"cache_key": cache_key})

    def delete(self, query: str) -> None:
        cache_key = self._hash_query(query)
        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM query_cache WHERE cache_key = ?", (cache_key,))
            conn.commit()
        self.logger.debug("query_cache_delete", extra={"cache_key": cache_key})

    def purge(self) -> None:
        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM query_cache")
            conn.commit()
        self.logger.info("query_cache_purged")

    def keys(self) -> List[str]:
        with self._lock, sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("SELECT cache_key FROM query_cache").fetchall()
        return [row[0] for row in rows]

    def size(self) -> int:
        """Return current number of entries in cache."""
        with self._lock, sqlite3.connect(self.db_path) as conn:
            return conn.execute("SELECT COUNT(*) FROM query_cache").fetchone()[0]
