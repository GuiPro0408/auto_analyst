"""Model loading helpers for free/open-source LLMs and embedding models."""

from functools import lru_cache
from time import perf_counter, sleep
from typing import Any, Dict, List, Optional

from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer

from api.config import (
    DEFAULT_EMBED_MODEL,
    DEFAULT_LLM_MODEL,
    GEMINI_API_KEYS,
    GEMINI_MODEL,
    GENERATION_KWARGS,
    GROQ_API_KEY,
    GROQ_MODEL,
    HUGGINGFACE_API_TOKEN,
    HUGGINGFACE_INFERENCE_MODEL,
    LLM_BACKEND,
    LLM_BACKOFF_SECONDS,
)
from api.key_rotator import APIKeyRotator, get_default_rotator
from api.logging_setup import get_logger
from tools.openai_compatible_llm import OpenAICompatibleLLM

# Re-export for backward compatibility with tests
GEMINI_API_KEY = GEMINI_API_KEYS[0] if GEMINI_API_KEYS else ""


class GeminiLLM:
    """Callable wrapper for Google Gemini models matching HF pipeline output.

    Uses the shared APIKeyRotator for rate limit handling across API keys.
    """

    def __init__(
        self,
        model_name: str,
        generation_kwargs: Dict[str, Any],
        api_key: Optional[str] = None,
        api_keys: Optional[List[str]] = None,
        key_rotator: Optional[APIKeyRotator] = None,
        genai_module: Optional[Any] = None,
        fallback_llm: Optional[Any] = None,
    ) -> None:
        self.logger = get_logger(__name__)
        self._fallback_llm = fallback_llm
        self._using_fallback = False

        # Build key rotator from provided keys or use default
        # Filter out empty strings from key lists
        if key_rotator is not None:
            self._key_rotator = key_rotator
        elif api_keys:
            valid_keys = [k for k in api_keys if k]
            self._key_rotator = APIKeyRotator(valid_keys)
        elif api_key:
            self._key_rotator = APIKeyRotator([api_key] if api_key else [])
        else:
            self._key_rotator = get_default_rotator()

        if self._key_rotator.total_keys == 0:
            raise ValueError(
                "GOOGLE_API_KEY is not set. Please add it to your environment or .env file."
            )

        if genai_module is None:
            try:
                import google.generativeai as genai  # pylint: disable=import-error
            except ImportError as exc:  # pragma: no cover - dependency issue
                raise ImportError(
                    "google-generativeai is required for the Gemini backend. Install it via requirements.txt."
                ) from exc
            genai_module = genai

        self._genai = genai_module
        self._model_name = model_name
        self._generation_kwargs = generation_kwargs
        self._generation_config = self._map_generation_kwargs(generation_kwargs)
        self._configure_model()
        self.logger.info(
            "gemini_llm_initialized",
            extra={
                "model_name": model_name,
                "generation_config": self._generation_config,
                "has_fallback": fallback_llm is not None,
                "api_key_count": self._key_rotator.total_keys,
            },
        )

    def _configure_model(self) -> None:
        """Configure Gemini with current API key from rotator."""
        current_key = self._key_rotator.current_key
        if not current_key:
            raise ValueError("No API keys available")
        self._genai.configure(api_key=current_key)
        self._model = self._genai.GenerativeModel(self._model_name)
        self.logger.debug(
            "gemini_api_key_configured",
            extra={
                "model_name": self._model_name,
                "key_suffix": current_key[-8:] if current_key else "none",
            },
        )

    @staticmethod
    def _map_generation_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
        config: Dict[str, Any] = {}
        if "max_new_tokens" in kwargs:
            config["max_output_tokens"] = kwargs["max_new_tokens"]
        if "temperature" in kwargs:
            config["temperature"] = kwargs["temperature"]
        # Gemini does not support do_sample flag directly; ignore but keep log clarity.
        return config

    def set_fallback(self, fallback_llm: Any) -> None:
        """Set a fallback LLM to use when rate limited."""
        self._fallback_llm = fallback_llm
        self.logger.info(
            "gemini_fallback_set",
            extra={"fallback_type": type(fallback_llm).__name__},
        )

    def _call_fallback(
        self, prompt: str, cause: Optional[Exception] = None
    ) -> List[Dict[str, str]]:
        if self._fallback_llm is None:
            raise RuntimeError("No fallback LLM configured.")

        try:
            return self._fallback_llm(prompt)
        except Exception as exc:  # pragma: no cover - remote dependency
            self.logger.exception(
                "gemini_fallback_failed",
                extra={
                    "cause": str(cause)[:200] if cause else None,
                    "fallback_error": str(exc)[:200],
                },
            )
            return [
                {
                    "generated_text": (
                        "Unable to generate a response because both primary and fallback LLMs failed. "
                        "Please retry shortly or configure additional providers."
                    )
                }
            ]

    def __call__(self, prompt: str) -> List[Dict[str, str]]:
        # If we've already switched to fallback, use it directly
        if self._using_fallback and self._fallback_llm is not None:
            self.logger.info(
                "gemini_using_fallback",
                extra={"reason": "previously_rate_limited"},
            )
            return self._call_fallback(prompt)

        # Check if all keys are exhausted before attempting
        if self._key_rotator.is_exhausted:
            if self._fallback_llm is not None:
                self.logger.warning(
                    "gemini_all_keys_exhausted_using_fallback",
                    extra={"total_keys": self._key_rotator.total_keys},
                )
                self._using_fallback = True
                return self._call_fallback(prompt)
            else:
                self.logger.error(
                    "gemini_all_keys_exhausted_no_fallback",
                    extra={"total_keys": self._key_rotator.total_keys},
                )
                return [
                    {
                        "generated_text": (
                            "All Gemini API keys are rate limited and no fallback LLM is configured. "
                            "Please wait a moment and retry."
                        )
                    }
                ]

        attempts = 0
        max_attempts = self._key_rotator.total_keys
        last_error: Optional[Exception] = None

        while attempts < max_attempts:
            start = perf_counter()
            current_key = self._key_rotator.current_key

            # Check for key exhaustion mid-loop
            if current_key is None:
                if self._fallback_llm is not None:
                    self.logger.warning(
                        "gemini_keys_exhausted_mid_loop_using_fallback",
                        extra={"attempts": attempts},
                    )
                    self._using_fallback = True
                    return self._call_fallback(prompt, last_error)
                break

            try:
                self._configure_model()  # Ensure we're using current key
                response = self._model.generate_content(
                    prompt,
                    generation_config=self._generation_config,
                )
            except Exception as exc:  # pragma: no cover - relies on remote API
                last_error = exc
                if self._key_rotator.is_rate_limit_error(exc):
                    has_more = self._key_rotator.mark_rate_limited(current_key or "")
                    if has_more:
                        attempts += 1
                        continue

                if self._fallback_llm is not None:
                    self.logger.warning(
                        "gemini_error_using_fallback",
                        extra={
                            "model_name": self._model_name,
                            "error": str(exc)[:200],
                            "fallback_type": type(self._fallback_llm).__name__,
                        },
                    )
                    self._using_fallback = True
                    return self._call_fallback(prompt, exc)

                self.logger.exception(
                    "gemini_generate_failed",
                    extra={"model_name": self._model_name},
                )
                raise RuntimeError(f"Gemini API call failed: {exc}") from exc

            # Success - reset rate limit tracking
            self._key_rotator.reset()

            text = getattr(response, "text", "")
            if not text and getattr(response, "candidates", None):
                parts: List[str] = []
                for candidate in response.candidates:
                    content = getattr(candidate, "content", None)
                    content_parts = getattr(content, "parts", None) if content else None
                    if not content_parts:
                        continue
                    for part in content_parts:
                        parts.append(getattr(part, "text", ""))
                text = "".join(parts).strip()

            if not text:
                self.logger.warning(
                    "gemini_response_empty",
                    extra={"model_name": self._model_name},
                )
                text = ""

            duration_ms = (perf_counter() - start) * 1000
            self.logger.info(
                "gemini_generate_complete",
                extra={
                    "model_name": self._model_name,
                    "duration_ms": duration_ms,
                    "output_length": len(text),
                    "available_keys": self._key_rotator.available_keys,
                },
            )
            return [{"generated_text": text}]

        if self._fallback_llm is not None:
            self.logger.warning(
                "gemini_all_api_keys_exhausted_falling_back",
                extra={
                    "model_name": self._model_name,
                    "attempted_keys": self._key_rotator.total_keys,
                    "last_error": str(last_error)[:200] if last_error else None,
                },
            )
            self._using_fallback = True
            return self._call_fallback(prompt, last_error)

        raise RuntimeError(
            "Gemini API call failed and no fallback is configured. Last error: "
            f"{last_error}"
        )


@lru_cache(maxsize=2)
def load_embedding_model(
    model_name: str = DEFAULT_EMBED_MODEL, device: str = "cpu"
) -> SentenceTransformer:
    """Load a sentence-transformers embedding model."""
    logger = get_logger(__name__)
    logger.info(
        "load_embedding_model_start", extra={"model_name": model_name, "device": device}
    )
    start = perf_counter()
    model = SentenceTransformer(model_name, device=device)
    duration_ms = (perf_counter() - start) * 1000
    logger.info(
        "load_embedding_model_complete",
        extra={"model_name": model_name, "device": device, "duration_ms": duration_ms},
    )
    return model


class HuggingFaceInferenceLLM:
    """Callable wrapper around HuggingFace Inference API."""

    def __init__(
        self,
        model_name: str,
        api_token: str,
        generation_kwargs: Dict[str, Any],
        client: Optional[InferenceClient] = None,
        retries: int = 2,
        backoff_seconds: float = LLM_BACKOFF_SECONDS,
    ) -> None:
        if not api_token:
            raise ValueError(
                "HUGGINGFACE_API_TOKEN is not set. Please configure it in your environment or .env file."
            )

        self.logger = get_logger(__name__)
        self._model_name = model_name
        self._generation_kwargs = self._map_generation_kwargs(generation_kwargs)
        self._client = client or InferenceClient(model=model_name, token=api_token)
        self._retries = max(1, retries)
        self._backoff_seconds = max(0.0, backoff_seconds)
        self.logger.info(
            "hf_inference_llm_initialized",
            extra={
                "model_name": model_name,
                "generation_kwargs": self._generation_kwargs,
                "retries": self._retries,
                "backoff_seconds": self._backoff_seconds,
            },
        )

    @staticmethod
    def _map_generation_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Map generation kwargs to HuggingFace chat_completion format."""
        mapped = {}
        if "max_new_tokens" in kwargs:
            mapped["max_tokens"] = kwargs["max_new_tokens"]
        if "temperature" in kwargs:
            mapped["temperature"] = kwargs["temperature"]
        # do_sample is not used in chat_completion
        return mapped

    def __call__(self, prompt: str) -> List[Dict[str, str]]:
        last_error: Optional[Exception] = None
        for attempt in range(self._retries):
            start = perf_counter()
            try:
                # Use chat_completion which has better provider support
                response = self._client.chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    **self._generation_kwargs,
                )
                text = response.choices[0].message.content
                duration_ms = (perf_counter() - start) * 1000
                if text is None:
                    text = ""
                self.logger.info(
                    "hf_inference_generate_complete",
                    extra={
                        "model_name": self._model_name,
                        "duration_ms": duration_ms,
                        "output_length": len(text),
                        "attempt": attempt + 1,
                    },
                )
                return [{"generated_text": text}]
            except Exception as exc:  # pragma: no cover - remote dependency
                last_error = exc
                if attempt < self._retries - 1:
                    self.logger.warning(
                        "hf_inference_retry",
                        extra={
                            "model_name": self._model_name,
                            "attempt": attempt + 1,
                            "error": str(exc)[:200],
                        },
                    )
                    if self._backoff_seconds:
                        sleep(self._backoff_seconds)
                    continue

                self.logger.exception(
                    "hf_inference_generate_failed",
                    extra={
                        "model_name": self._model_name,
                        "attempts": self._retries,
                    },
                )
                raise RuntimeError(
                    f"HuggingFace Inference API call failed after {self._retries} attempts: {exc}"
                ) from exc

        # This should be unreachable, but satisfies type checker
        raise RuntimeError(
            f"HuggingFace Inference API call failed after {self._retries} attempts: {last_error}"
        )


@lru_cache(maxsize=2)
def load_llm(model_name: str = DEFAULT_LLM_MODEL):
    """Load an LLM via cloud API (Gemini, Groq, or HuggingFace Inference).

    Cloud providers are used exclusively to avoid local GPU/CPU inference costs.
    Priority order for fallback: Gemini -> Groq -> HuggingFace.
    """
    logger = get_logger(__name__)

    backend = LLM_BACKEND.lower()

    # Prepare fallbacks
    groq_llm = None
    if GROQ_API_KEY:
        try:
            groq_llm = OpenAICompatibleLLM(
                model_name=(
                    GROQ_MODEL if model_name == DEFAULT_LLM_MODEL else model_name
                ),
                api_key=GROQ_API_KEY,
                generation_kwargs=GENERATION_KWARGS,
                base_url="https://api.groq.com/openai/v1",
                provider_name="groq",
            )
            logger.info("load_llm_groq_ready")
        except Exception as e:
            logger.warning("load_llm_groq_init_failed", extra={"error": str(e)})

    hf_fallback = None
    if HUGGINGFACE_API_TOKEN:
        try:
            hf_model = (
                HUGGINGFACE_INFERENCE_MODEL
                if model_name == DEFAULT_LLM_MODEL
                else model_name
            )
            hf_fallback = HuggingFaceInferenceLLM(
                model_name=hf_model,
                api_token=HUGGINGFACE_API_TOKEN,
                generation_kwargs=GENERATION_KWARGS,
            )
            logger.info("load_llm_hf_ready")
        except Exception as exc:
            logger.warning("load_llm_hf_init_failed", extra={"error": str(exc)})

    # Chain fallbacks: Gemini -> Groq -> HF
    primary_fallback = groq_llm or hf_fallback

    if backend == "gemini":
        configured_model = (
            GEMINI_MODEL if model_name == DEFAULT_LLM_MODEL else model_name
        )
        logger.info(
            "load_llm_gemini_start",
            extra={
                "model_name": configured_model,
                "generation_kwargs": GENERATION_KWARGS,
            },
        )
        try:
            key_rotator = get_default_rotator(GEMINI_API_KEYS)
            return GeminiLLM(
                model_name=configured_model,
                generation_kwargs=GENERATION_KWARGS,
                key_rotator=key_rotator,
                fallback_llm=primary_fallback,
            )
        except ImportError as exc:
            logger.warning("load_llm_gemini_dependency_missing")
            if primary_fallback:
                return primary_fallback
            raise

    if backend == "groq":
        if not groq_llm:
            raise ValueError(
                "GROQ_API_KEY not configured but requested as primary backend."
            )
        # Configure HuggingFace as fallback for Groq rate limits
        if hf_fallback:
            groq_llm.set_fallback(hf_fallback)
            logger.info("load_llm_groq_with_hf_fallback")
        return groq_llm

    if backend in {"huggingface", "hf", "hf_inference"}:
        if not hf_fallback:
            raise ValueError(
                "HUGGINGFACE_API_TOKEN not configured but requested as primary backend."
            )
        return hf_fallback

    raise ValueError(f"Unsupported LLM_BACKEND: {backend}")
