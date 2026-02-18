"""OpenAI-compatible LLM wrapper for Auto-Analyst."""

from time import monotonic, perf_counter, sleep
from typing import Any, Dict, List, Optional

import httpx

from api.config import GROQ_AUTH_COOLDOWN_SECONDS
from api.logging_setup import get_logger


class OpenAICompatibleLLM:
    """Callable wrapper for OpenAI-compatible APIs (Groq, etc.).

    Provides a standardized interface to various cloud LLM providers that
    follow the OpenAI chat completions API format. Supports fallback to
    another LLM on rate limit errors.
    """

    def __init__(
        self,
        model_name: str,
        api_key: str,
        generation_kwargs: Dict[str, Any],
        base_url: str,
        timeout: int = 60,
        provider_name: str = "openai_compatible",
        fallback_llm: Optional[Any] = None,
        max_retries: int = 2,
        retry_delay: float = 1.0,
        auth_cooldown_seconds: int = GROQ_AUTH_COOLDOWN_SECONDS,
    ) -> None:
        if not api_key:
            raise ValueError(f"{provider_name} API key is not set.")

        self.logger = get_logger(__name__)
        self._model_name = model_name
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._provider_name = provider_name
        self._generation_kwargs = self._map_generation_kwargs(generation_kwargs)
        self._fallback_llm = fallback_llm
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._auth_cooldown_seconds = max(0, auth_cooldown_seconds)
        self._disabled_until = 0.0
        self._disable_reason: Optional[str] = None

        self.logger.info(
            f"{self._provider_name}_llm_initialized",
            extra={
                "model_name": model_name,
                "generation_kwargs": self._generation_kwargs,
                "has_fallback": fallback_llm is not None,
                "auth_cooldown_seconds": self._auth_cooldown_seconds,
            },
        )

    def set_fallback(self, fallback_llm: Any) -> None:
        """Set a fallback LLM for rate limit handling."""
        self._fallback_llm = fallback_llm
        self.logger.info(
            f"{self._provider_name}_fallback_configured",
            extra={"fallback_type": type(fallback_llm).__name__},
        )

    @staticmethod
    def _map_generation_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Map generic generation kwargs to OpenAI-compatible format."""
        mapped = {}
        if "max_new_tokens" in kwargs:
            mapped["max_tokens"] = kwargs["max_new_tokens"]
        if "temperature" in kwargs:
            mapped["temperature"] = kwargs["temperature"]
        if "top_p" in kwargs:
            mapped["top_p"] = kwargs["top_p"]
        return mapped

    def _is_rate_limit_error(self, exc: Exception) -> bool:
        """Check if exception is a rate limit error."""
        status_code = self._status_code_from_exception(exc)
        if status_code == 429:
            return True
        error_str = str(exc).lower()
        return any(
            term in error_str
            for term in ("429", "rate limit", "too many requests", "quota")
        )

    @staticmethod
    def _status_code_from_exception(exc: Exception) -> Optional[int]:
        if isinstance(exc, httpx.HTTPStatusError) and exc.response is not None:
            return exc.response.status_code
        return None

    def _is_auth_permission_error(self, exc: Exception) -> bool:
        """Check if exception indicates auth/permission/model-access failure."""
        status_code = self._status_code_from_exception(exc)
        if status_code in {401, 403}:
            return True
        if status_code == 404:
            error_str = str(exc).lower()
            return any(
                term in error_str
                for term in ("model", "not found", "does not exist", "unknown model")
            )

        error_str = str(exc).lower()
        return any(
            term in error_str
            for term in ("forbidden", "unauthorized", "permission denied")
        )

    def _cooldown_remaining_seconds(self) -> float:
        return max(0.0, self._disabled_until - monotonic())

    def _call_fallback(
        self, prompt: str, cause: Optional[Exception] = None
    ) -> List[Dict[str, str]]:
        """Call the fallback LLM."""
        if self._fallback_llm is None:
            raise RuntimeError(
                f"{self._provider_name} API call failed and no fallback configured: {cause}"
            )

        self.logger.warning(
            f"{self._provider_name}_using_fallback",
            extra={
                "reason": str(cause)[:200] if cause else "unknown",
                "fallback_type": type(self._fallback_llm).__name__,
            },
        )
        return self._fallback_llm(prompt)

    def __call__(self, prompt: str) -> List[Dict[str, str]]:
        start = perf_counter()

        if self._disabled_until > monotonic():
            remaining = self._cooldown_remaining_seconds()
            self.logger.info(
                f"{self._provider_name}_disabled_cooldown_active",
                extra={
                    "remaining_seconds": round(remaining, 2),
                    "reason": self._disable_reason or "auth_or_permission_failure",
                    "has_fallback": self._fallback_llm is not None,
                },
            )
            if self._fallback_llm is not None:
                return self._call_fallback(
                    prompt,
                    RuntimeError(
                        f"{self._provider_name} temporarily disabled ({remaining:.1f}s remaining)"
                    ),
                )
            raise RuntimeError(
                f"{self._provider_name} temporarily disabled due to auth/permission errors: "
                f"{self._disable_reason}"
            )

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self._model_name,
            "messages": [{"role": "user", "content": prompt}],
            **self._generation_kwargs,
        }

        last_error: Optional[Exception] = None

        for attempt in range(self._max_retries):
            try:
                with httpx.Client(timeout=self._timeout) as client:
                    response = client.post(
                        f"{self._base_url}/chat/completions",
                        headers=headers,
                        json=payload,
                    )
                    response.raise_for_status()
                    data = response.json()

                text = data["choices"][0]["message"]["content"]
                duration_ms = (perf_counter() - start) * 1000

                self.logger.info(
                    f"{self._provider_name}_generate_complete",
                    extra={
                        "model_name": self._model_name,
                        "duration_ms": duration_ms,
                        "output_length": len(text) if text else 0,
                    },
                )
                return [{"generated_text": text}]

            except Exception as exc:
                last_error = exc
                status_code = self._status_code_from_exception(exc)

                if self._is_auth_permission_error(exc):
                    self._disable_reason = str(exc)[:200]
                    self._disabled_until = monotonic() + self._auth_cooldown_seconds
                    self.logger.error(
                        f"{self._provider_name}_auth_permission_failure",
                        extra={
                            "status_code": status_code,
                            "cooldown_seconds": self._auth_cooldown_seconds,
                            "has_fallback": self._fallback_llm is not None,
                            "model_name": self._model_name,
                            "error": str(exc)[:200],
                        },
                    )
                    if self._fallback_llm is not None:
                        return self._call_fallback(prompt, exc)
                    raise RuntimeError(
                        f"{self._provider_name} authentication/permission failure: {exc}"
                    ) from exc

                if self._is_rate_limit_error(exc):
                    self.logger.warning(
                        f"{self._provider_name}_rate_limited",
                        extra={
                            "attempt": attempt + 1,
                            "max_retries": self._max_retries,
                            "has_fallback": self._fallback_llm is not None,
                        },
                    )

                    # If we have a fallback, use it immediately on rate limit
                    if self._fallback_llm is not None:
                        return self._call_fallback(prompt, exc)

                    # Otherwise retry with delay
                    if attempt < self._max_retries - 1:
                        sleep(self._retry_delay * (attempt + 1))
                        continue

                # Non-rate-limit error or final attempt
                self.logger.exception(
                    f"{self._provider_name}_generate_failed",
                    extra={"model_name": self._model_name, "error": str(exc)},
                )

                # Try fallback for any error on final attempt
                if self._fallback_llm is not None:
                    return self._call_fallback(prompt, exc)

                raise RuntimeError(
                    f"{self._provider_name} API call failed: {exc}"
                ) from exc

        # All retries exhausted
        if self._fallback_llm is not None:
            return self._call_fallback(prompt, last_error)

        raise RuntimeError(
            f"{self._provider_name} API call failed after {self._max_retries} attempts: {last_error}"
        )
