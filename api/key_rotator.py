"""Shared API key rotation for rate limit handling."""

from __future__ import annotations

import threading
from typing import List, Optional, Set

from api.logging_setup import get_logger


class APIKeyRotator:
    """Manages rotation through multiple API keys on rate limits.

    Thread-safe implementation that tracks rate-limited keys and
    rotates to available ones automatically.
    """

    def __init__(self, api_keys: Optional[List[str]] = None) -> None:
        """Initialize the rotator with a list of API keys.

        Args:
            api_keys: List of API keys to rotate through. Empty list is allowed.
        """
        self._keys: List[str] = list(api_keys) if api_keys else []
        self._current_index: int = 0
        self._rate_limited_keys: Set[str] = set()
        self._lock = threading.Lock()
        self._logger = get_logger(__name__)

    @property
    def current_key(self) -> Optional[str]:
        """Get current API key, skipping rate-limited ones.

        Returns:
            Current available API key, or None if no keys configured or all exhausted.
        """
        with self._lock:
            if not self._keys:
                return None

            # Try to find a non-rate-limited key
            for _ in range(len(self._keys)):
                key = self._keys[self._current_index]
                if key not in self._rate_limited_keys:
                    return key
                self._current_index = (self._current_index + 1) % len(self._keys)

            # All keys rate limited - return None to signal exhaustion
            self._logger.warning(
                "api_key_rotator_all_exhausted",
                extra={
                    "total_keys": len(self._keys),
                    "rate_limited_keys": len(self._rate_limited_keys),
                },
            )
            return None

    def mark_rate_limited(self, key: str) -> bool:
        """Mark key as rate limited and rotate to next.

        Args:
            key: The API key to mark as rate limited.

        Returns:
            True if more keys are available, False if all keys exhausted.
        """
        with self._lock:
            self._rate_limited_keys.add(key)
            self._current_index = (
                (self._current_index + 1) % len(self._keys) if self._keys else 0
            )
            has_more = len(self._rate_limited_keys) < len(self._keys)
            self._logger.warning(
                "api_key_rate_limited",
                extra={
                    "key_suffix": key[-8:] if key else "none",
                    "rate_limited_count": len(self._rate_limited_keys),
                    "total_keys": len(self._keys),
                    "has_more_keys": has_more,
                },
            )
            return has_more

    def reset(self) -> None:
        """Reset rate limit tracking (e.g., after successful request)."""
        with self._lock:
            if self._rate_limited_keys:
                self._logger.debug(
                    "api_key_rotator_reset",
                    extra={"cleared_keys": len(self._rate_limited_keys)},
                )
            self._rate_limited_keys.clear()

    @property
    def total_keys(self) -> int:
        """Total number of configured API keys."""
        return len(self._keys)

    @property
    def available_keys(self) -> int:
        """Number of keys not currently rate-limited."""
        with self._lock:
            return len(self._keys) - len(self._rate_limited_keys)

    @property
    def is_exhausted(self) -> bool:
        """Check if all API keys are currently rate-limited."""
        with self._lock:
            return len(self._keys) > 0 and len(self._rate_limited_keys) >= len(
                self._keys
            )

    def is_rate_limit_error(self, error: Exception) -> bool:
        """Check if an exception indicates a rate limit error.

        Args:
            error: The exception to check.

        Returns:
            True if the error appears to be rate-limit related.
        """
        error_str = str(error).lower()
        return any(
            term in error_str
            for term in (
                "429",
                "resource_exhausted",
                "rate limit",
                "quota",
                "too many requests",
            )
        )

    def is_transient_error(self, error: Exception) -> bool:
        """Check if an exception is a transient/retriable error.

        Args:
            error: The exception to check.

        Returns:
            True if the error appears to be transient.
        """
        error_str = str(error).lower()
        return any(
            term in error_str
            for term in (
                "unavailable",
                "deadline",
                "timeout",
                "connection",
                "temporary",
            )
        )


# Singleton instance - lazy initialized
_default_rotator: Optional[APIKeyRotator] = None
_rotator_lock = threading.Lock()


def get_default_rotator(api_keys: Optional[List[str]] = None) -> APIKeyRotator:
    """Get or create the default API key rotator.

    Args:
        api_keys: API keys to use if creating a new rotator.
                  Ignored if rotator already exists.

    Returns:
        The shared APIKeyRotator instance.
    """
    global _default_rotator
    with _rotator_lock:
        if _default_rotator is None:
            from api.config import GEMINI_API_KEYS

            keys = api_keys if api_keys is not None else GEMINI_API_KEYS
            _default_rotator = APIKeyRotator(keys)
        return _default_rotator


def reset_default_rotator() -> None:
    """Reset the default rotator (mainly for testing)."""
    global _default_rotator
    with _rotator_lock:
        _default_rotator = None
