"""Shared LLM backend detection utilities.

These functions are used across multiple modules to check the current LLM backend
and adjust behavior accordingly (e.g., skip expensive operations for local/groq).
"""

from api.config import LLM_BACKEND


def is_local_backend() -> bool:
    """Check if using local LLM backend (llama.cpp)."""
    return LLM_BACKEND.lower() in {"local", "llama_cpp", "llamacpp"}


def is_limited_backend() -> bool:
    """Check if backend has limits that require optimizations.

    Returns True for:
    - Local backends (slow inference)
    - Groq (rate limits, smaller context)
    """
    return LLM_BACKEND.lower() in {"local", "llama_cpp", "llamacpp", "groq"}
