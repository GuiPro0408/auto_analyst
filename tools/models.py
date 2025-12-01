"""Model loading helpers for free/open-source LLMs and embedding models."""

import os
from functools import lru_cache
from time import perf_counter

from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from api.config import (
    DEFAULT_EMBED_MODEL,
    DEFAULT_LLM_MODEL,
    GENERATION_KWARGS,
    MIN_RECOMMENDED_MEMORY_GB,
)
from api.logging_setup import get_logger


def _get_available_memory_gb() -> float:
    """Get available system memory in GB."""
    try:
        import psutil

        mem = psutil.virtual_memory()
        return mem.available / (1024**3)
    except ImportError:
        # psutil not installed, try reading /proc/meminfo on Linux
        try:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        # Value is in kB
                        kb = int(line.split()[1])
                        return kb / (1024**2)
        except (FileNotFoundError, ValueError, IndexError):
            pass
    return -1  # Unknown


def _check_memory_requirements(model_name: str) -> None:
    """Log a warning if available memory is below recommended threshold."""
    logger = get_logger(__name__)

    # Model memory requirements (approximate, in GB)
    # These are full precision weights; actual usage varies with batch size
    MODEL_MEMORY_REQUIREMENTS = {
        "qwen2.5-1.5b": 4,
        "qwen2.5-0.5b": 2,
        "qwen2.5-3b": 7,
        "qwen2.5-7b": 16,
        "gemma-2-2b": 6,
        "phi-3-mini": 8,
        "tinyllama": 3,
        "mistral-7b": 16,
        "llama-3.1-8b": 18,
    }

    available_gb = _get_available_memory_gb()
    if available_gb < 0:
        logger.debug(
            "memory_check_skipped",
            extra={"reason": "unable_to_detect_memory"},
        )
        return

    # Determine required memory based on model name
    model_lower = model_name.lower()
    required_gb = MIN_RECOMMENDED_MEMORY_GB  # Default fallback

    for pattern, mem_gb in MODEL_MEMORY_REQUIREMENTS.items():
        if pattern in model_lower:
            required_gb = mem_gb
            break

    if available_gb < required_gb:
        # Suggest appropriate alternative based on available memory
        if available_gb >= 6:
            suggestion = "Qwen/Qwen2.5-1.5B-Instruct or google/gemma-2-2b-it"
        elif available_gb >= 4:
            suggestion = "Qwen/Qwen2.5-1.5B-Instruct or Qwen/Qwen2.5-0.5B-Instruct"
        else:
            suggestion = (
                "Qwen/Qwen2.5-0.5B-Instruct or TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            )

        logger.warning(
            "low_memory_warning",
            extra={
                "available_gb": round(available_gb, 2),
                "required_gb": required_gb,
                "model_name": model_name,
                "suggestion": f"Consider using: {suggestion}",
            },
        )
        print(
            f"\n⚠️  WARNING: Low memory detected ({available_gb:.1f}GB available, "
            f"~{required_gb}GB needed for {model_name}).\n"
            f"   Consider using a smaller model:\n"
            f"   export AUTO_ANALYST_LLM='{suggestion.split(' or ')[0]}'\n"
        )
    else:
        logger.info(
            "memory_check_passed",
            extra={
                "available_gb": round(available_gb, 2),
                "required_gb": required_gb,
                "model_name": model_name,
            },
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


@lru_cache(maxsize=2)
def load_llm(model_name: str = DEFAULT_LLM_MODEL, device_map: str = "auto"):
    """Load an instruct-tuned causal LM as a text-generation pipeline."""
    logger = get_logger(__name__)
    logger.info(
        "load_llm_start",
        extra={"model_name": model_name, "device_map": device_map},
    )

    # Check memory requirements before loading
    _check_memory_requirements(model_name)

    start = perf_counter()
    logger.debug("load_llm_tokenizer", extra={"model_name": model_name})
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    logger.debug("load_llm_model", extra={"model_name": model_name})
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map=device_map, trust_remote_code=False
    )
    gen_kwargs = GENERATION_KWARGS.copy()
    logger.debug(
        "load_llm_pipeline",
        extra={"gen_kwargs": gen_kwargs},
    )
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, **gen_kwargs)
    duration_ms = (perf_counter() - start) * 1000
    logger.info(
        "load_llm_complete",
        extra={"model_name": model_name, "duration_ms": duration_ms},
    )
    return pipe
