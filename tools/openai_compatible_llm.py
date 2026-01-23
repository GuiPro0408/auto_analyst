"""OpenAI-compatible LLM wrapper for Auto-Analyst."""

import json
from time import perf_counter
from typing import Any, Dict, List, Optional

import httpx

from api.logging_setup import get_logger


class OpenAICompatibleLLM:
    """Callable wrapper for OpenAI-compatible APIs (Groq, etc.).
    
    Provides a standardized interface to various cloud LLM providers that 
    follow the OpenAI chat completions API format.
    """

    def __init__(
        self,
        model_name: str,
        api_key: str,
        generation_kwargs: Dict[str, Any],
        base_url: str,
        timeout: int = 60,
        provider_name: str = "openai_compatible",
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
        
        self.logger.info(
            f"{self._provider_name}_llm_initialized",
            extra={
                "model_name": model_name,
                "generation_kwargs": self._generation_kwargs,
            },
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

    def __call__(self, prompt: str) -> List[Dict[str, str]]:
        start = perf_counter()
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": self._model_name,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            **self._generation_kwargs
        }
        
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
            self.logger.exception(
                f"{self._provider_name}_generate_failed",
                extra={"model_name": self._model_name, "error": str(exc)},
            )
            raise RuntimeError(f"{self._provider_name} API call failed: {exc}") from exc
