"""Model loading helpers for free/open-source LLMs and embedding models."""

from functools import lru_cache
from time import perf_counter
from typing import Any, Dict, List, Optional

from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer

from api.config import (
    DEFAULT_EMBED_MODEL,
    DEFAULT_LLM_MODEL,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GENERATION_KWARGS,
    HUGGINGFACE_API_TOKEN,
    HUGGINGFACE_INFERENCE_MODEL,
    LLM_BACKEND,
)
from api.logging_setup import get_logger


class GeminiLLM:
    """Callable wrapper for Google Gemini models matching HF pipeline output."""

    def __init__(
        self,
        model_name: str,
        api_key: str,
        generation_kwargs: Dict[str, Any],
        genai_module: Optional[Any] = None,
    ) -> None:
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY is not set. Please add it to your environment or .env file."
            )

        self.logger = get_logger(__name__)

        if genai_module is None:
            try:
                import google.generativeai as genai  # pylint: disable=import-error
            except ImportError as exc:  # pragma: no cover - dependency issue
                raise ImportError(
                    "google-generativeai is required for the Gemini backend. Install it via requirements.txt."
                ) from exc
            genai_module = genai

        genai_module.configure(api_key=api_key)
        self._genai = genai_module
        self._model = genai_module.GenerativeModel(model_name)
        self._model_name = model_name
        self._generation_config = self._map_generation_kwargs(generation_kwargs)
        self.logger.info(
            "gemini_llm_initialized",
            extra={
                "model_name": model_name,
                "generation_config": self._generation_config,
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

    def __call__(self, prompt: str) -> List[Dict[str, str]]:
        start = perf_counter()
        try:
            response = self._model.generate_content(
                prompt,
                generation_config=self._generation_config,
            )
        except Exception as exc:  # pragma: no cover - relies on remote API
            self.logger.exception(
                "gemini_generate_failed",
                extra={"model_name": self._model_name},
            )
            raise RuntimeError(f"Gemini API call failed: {exc}") from exc

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
            },
        )
        return [{"generated_text": text}]


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
    ) -> None:
        if not api_token:
            raise ValueError(
                "HUGGINGFACE_API_TOKEN is not set. Please configure it in your environment or .env file."
            )

        self.logger = get_logger(__name__)
        self._model_name = model_name
        self._generation_kwargs = self._map_generation_kwargs(generation_kwargs)
        self._client = client or InferenceClient(model=model_name, token=api_token)
        self.logger.info(
            "hf_inference_llm_initialized",
            extra={
                "model_name": model_name,
                "generation_kwargs": self._generation_kwargs,
            },
        )

    @staticmethod
    def _map_generation_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
        mapped = {}
        if "max_new_tokens" in kwargs:
            mapped["max_new_tokens"] = kwargs["max_new_tokens"]
        if "temperature" in kwargs:
            mapped["temperature"] = kwargs["temperature"]
        if "do_sample" in kwargs:
            mapped["do_sample"] = kwargs["do_sample"]
        return mapped

    def __call__(self, prompt: str) -> List[Dict[str, str]]:
        start = perf_counter()
        try:
            response = self._client.text_generation(
                prompt,
                **self._generation_kwargs,
            )
        except Exception as exc:  # pragma: no cover - remote dependency
            self.logger.exception(
                "hf_inference_generate_failed",
                extra={"model_name": self._model_name},
            )
            raise RuntimeError(f"HuggingFace Inference API call failed: {exc}") from exc

        text: Optional[str] = None
        if hasattr(response, "generated_text"):
            text = getattr(response, "generated_text")
        elif isinstance(response, dict):
            text = response.get("generated_text")
        elif isinstance(response, str):
            text = response

        if text is None:
            text = str(response)

        duration_ms = (perf_counter() - start) * 1000
        self.logger.info(
            "hf_inference_generate_complete",
            extra={
                "model_name": self._model_name,
                "duration_ms": duration_ms,
                "output_length": len(text),
            },
        )
        return [{"generated_text": text}]


@lru_cache(maxsize=2)
def load_llm(model_name: str = DEFAULT_LLM_MODEL, device_map: str = "auto"):
    """Load an instruct-tuned causal LM as a text-generation pipeline.

    Cloud providers are used exclusively to avoid local GPU/CPU inference costs.
    """
    logger = get_logger(__name__)

    backend = LLM_BACKEND.lower()
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
            return GeminiLLM(
                model_name=configured_model,
                api_key=GEMINI_API_KEY,
                generation_kwargs=GENERATION_KWARGS,
            )
        except ImportError as exc:
            logger.warning(
                "load_llm_gemini_dependency_missing",
                extra={"error": str(exc)},
            )
            if HUGGINGFACE_API_TOKEN:
                fallback_model = (
                    HUGGINGFACE_INFERENCE_MODEL
                    if model_name == DEFAULT_LLM_MODEL
                    else model_name
                )
                logger.info(
                    "load_llm_fallback_hf",
                    extra={
                        "fallback_model": fallback_model,
                        "reason": "gemini_dependency_missing",
                    },
                )
                return HuggingFaceInferenceLLM(
                    model_name=fallback_model,
                    api_token=HUGGINGFACE_API_TOKEN,
                    generation_kwargs=GENERATION_KWARGS,
                )
            raise RuntimeError(
                "Gemini backend requires the 'google-generativeai' package. Install it via requirements.txt "
                "or set AUTO_ANALYST_LLM_BACKEND=huggingface."
            ) from exc
    if backend in {"huggingface", "hf", "hf_inference", "huggingface_inference"}:
        configured_model = (
            HUGGINGFACE_INFERENCE_MODEL
            if model_name == DEFAULT_LLM_MODEL
            else model_name
        )
        logger.info(
            "load_llm_hf_inference_start",
            extra={
                "model_name": configured_model,
                "generation_kwargs": GENERATION_KWARGS,
            },
        )
        return HuggingFaceInferenceLLM(
            model_name=configured_model,
            api_token=HUGGINGFACE_API_TOKEN,
            generation_kwargs=GENERATION_KWARGS,
        )

    raise ValueError(
        "Unsupported AUTO_ANALYST_LLM_BACKEND. Supported values: 'gemini', 'huggingface'."
    )
