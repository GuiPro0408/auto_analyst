"""Model loading helpers for free/open-source LLMs and embedding models."""

from functools import lru_cache

from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from api.config import DEFAULT_EMBED_MODEL, DEFAULT_LLM_MODEL, GENERATION_KWARGS


@lru_cache(maxsize=2)
def load_embedding_model(
    model_name: str = DEFAULT_EMBED_MODEL, device: str = "cpu"
) -> SentenceTransformer:
    """Load a sentence-transformers embedding model."""
    return SentenceTransformer(model_name, device=device)


@lru_cache(maxsize=2)
def load_llm(model_name: str = DEFAULT_LLM_MODEL, device_map: str = "auto"):
    """Load an instruct-tuned causal LM as a text-generation pipeline."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map=device_map, trust_remote_code=False
    )
    gen_kwargs = GENERATION_KWARGS.copy()
    return pipeline("text-generation", model=model, tokenizer=tokenizer, **gen_kwargs)
