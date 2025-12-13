"""Cross-encoder based reranking for retrieved chunks."""

import os
from functools import lru_cache
from typing import List, Optional, Sequence, Tuple

from sentence_transformers import CrossEncoder

from api.config import ENABLE_RERANKER, RERANK_MODEL_NAME
from api.logging_setup import get_logger
from api.state import Chunk

# Force CPU for reranker to avoid CUDA compatibility issues with older GPUs
RERANKER_DEVICE = os.getenv("AUTO_ANALYST_RERANKER_DEVICE", "cpu")


@lru_cache(maxsize=1)
def load_reranker(model_name: str = RERANK_MODEL_NAME) -> CrossEncoder:
    logger = get_logger(__name__)
    logger.info(
        "reranker_load_start", extra={"model_name": model_name, "device": RERANKER_DEVICE}
    )
    model = CrossEncoder(model_name, device=RERANKER_DEVICE)
    logger.info("reranker_load_complete", extra={"model_name": model_name})
    return model


def rerank_chunks(
    query: str,
    chunks: Sequence[Chunk],
    top_k: Optional[int] = None,
    model_name: str = RERANK_MODEL_NAME,
    model: Optional[CrossEncoder] = None,
    run_id: Optional[str] = None,
) -> Tuple[List[Chunk], List[float]]:
    """Rerank retrieved chunks using a cross-encoder model."""

    logger = get_logger(__name__, run_id=run_id)

    if not ENABLE_RERANKER or not chunks:
        return list(chunks), []

    try:
        reranker = model or load_reranker(model_name)
    except (OSError, ImportError, RuntimeError, ValueError) as exc:  # pragma: no cover - network/dependency issues
        logger.warning("reranker_load_failed", extra={"error": str(exc)})
        return list(chunks), []

    pairs = [(query, chunk.text or "") for chunk in chunks]
    logger.debug(
        "reranker_predict_start",
        extra={"pairs": len(pairs), "model_name": model_name},
    )
    scores = reranker.predict(pairs)
    logger.debug(
        "reranker_predict_complete",
        extra={"scores_len": len(scores)},
    )

    chunk_scores = list(zip(chunks, scores))
    chunk_scores.sort(key=lambda item: item[1], reverse=True)
    if top_k:
        chunk_scores = chunk_scores[:top_k]
    reranked_chunks = [chunk for chunk, _ in chunk_scores]
    reranked_scores = [float(score) for _, score in chunk_scores]
    logger.info(
        "reranker_applied",
        extra={"kept": len(reranked_chunks), "top_k": top_k},
    )
    return reranked_chunks, reranked_scores
