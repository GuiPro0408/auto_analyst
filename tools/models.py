"""Model loading helpers for free/open-source LLMs and embedding models."""

from functools import lru_cache
from time import perf_counter

from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from api.config import DEFAULT_EMBED_MODEL, DEFAULT_LLM_MODEL, GENERATION_KWARGS
from api.logging_setup import get_logger


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
