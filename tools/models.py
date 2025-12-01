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

    # Only check for large models (Mistral, Llama, etc.)
    large_model_patterns = ["mistral", "llama", "falcon", "mpt", "phi-3"]
    is_large_model = any(
        pattern in model_name.lower() for pattern in large_model_patterns
    )

    if not is_large_model:
        return

    available_gb = _get_available_memory_gb()
    if available_gb < 0:
        logger.debug(
            "memory_check_skipped",
            extra={"reason": "unable_to_detect_memory"},
        )
        return

    if available_gb < MIN_RECOMMENDED_MEMORY_GB:
        logger.warning(
            "low_memory_warning",
            extra={
                "available_gb": round(available_gb, 2),
                "recommended_gb": MIN_RECOMMENDED_MEMORY_GB,
                "model_name": model_name,
                "suggestion": (
                    f"Consider setting AUTO_ANALYST_LLM to a smaller model like "
                    f"'TinyLlama/TinyLlama-1.1B-Chat-v1.0' or 'microsoft/phi-2'"
                ),
            },
        )
        print(
            f"\n⚠️  WARNING: Low memory detected ({available_gb:.1f}GB available, "
            f"{MIN_RECOMMENDED_MEMORY_GB}GB recommended for {model_name}).\n"
            f"   Consider using a smaller model by setting:\n"
            f"   export AUTO_ANALYST_LLM='TinyLlama/TinyLlama-1.1B-Chat-v1.0'\n"
        )
    else:
        logger.info(
            "memory_check_passed",
            extra={
                "available_gb": round(available_gb, 2),
                "recommended_gb": MIN_RECOMMENDED_MEMORY_GB,
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
