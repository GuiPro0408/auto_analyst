"""Answer generation and verification helpers."""

import re
from typing import Dict, List, Tuple

from api.logging_setup import get_logger
from api.state import Chunk


def _format_context(chunks: List[Chunk]) -> str:
    lines = []
    for idx, chunk in enumerate(chunks):
        meta = chunk.metadata or {}
        title = meta.get("title") or "Source"
        url = meta.get("url") or ""
        lines.append(f"[{idx + 1}] {title} ({url}) :: {chunk.text}")
    return "\n".join(lines)


def _is_coherent(
    text: str, min_word_length: int = 3, max_repetition_ratio: float = 0.3
) -> bool:
    """Check if generated text appears coherent and not gibberish."""
    if not text or len(text.strip()) < 10:
        return False

    # Check for excessive repetition of short patterns
    words = text.lower().split()

    # Very short responses (less than 5 words) are OK if they're not gibberish
    # This allows test mocks and concise answers through
    if len(words) < 5:
        # Just check for basic gibberish indicators in short text
        alnum_count = sum(c.isalnum() or c.isspace() for c in text)
        alnum_ratio = alnum_count / len(text) if text else 0
        return alnum_ratio > 0.7

    # Count word frequencies
    word_counts: Dict[str, int] = {}
    for word in words:
        cleaned = re.sub(r"[^a-z]", "", word)
        if len(cleaned) >= 2:
            word_counts[cleaned] = word_counts.get(cleaned, 0) + 1

    if not word_counts:
        return False

    # Check if any single word is repeated too frequently (lowered threshold)
    total_meaningful = sum(word_counts.values())
    max_count = max(word_counts.values())
    if total_meaningful > 5 and max_count / total_meaningful > max_repetition_ratio:
        return False

    # Check for patterns like "word word word" or "... ... ..."
    pattern_matches = re.findall(r"(\b\w+\b)(?:\s+\1){2,}", text.lower())
    if len(pattern_matches) > 1:
        return False

    # Check for excessive ellipsis or dots (gibberish indicator)
    dot_count = text.count("...") + text.count("..") + text.count(". . .")
    if dot_count > 5:
        return False

    # Check for excessive non-alphanumeric characters (gibberish indicator)
    alnum_count = sum(c.isalnum() or c.isspace() for c in text)
    alnum_ratio = alnum_count / len(text) if text else 0
    if alnum_ratio < 0.75:
        return False

    # Check for repeated fragments (like "Bun" appearing many times)
    # If any 3+ letter word appears more than 5 times, likely gibberish
    for word, count in word_counts.items():
        if len(word) >= 3 and count > 5:
            return False

    # Check for nonsensical number/letter combinations
    nonsense_patterns = re.findall(r"[a-z]+\d+[a-z]*|\d+[a-z]+\d*", text.lower())
    if len(nonsense_patterns) > 3:
        return False

    return True


def build_citations(chunks: List[Chunk]) -> List[Dict[str, str]]:
    citations = []
    for idx, chunk in enumerate(chunks):
        meta = chunk.metadata or {}
        citations.append(
            {
                "marker": f"[{idx + 1}]",
                "url": meta.get("url", ""),
                "title": meta.get("title", "Source"),
                "media_type": meta.get("media_type", "text"),
            }
        )
    return citations


def _extract_meaningful_words(text: str) -> set:
    """Extract meaningful keywords from text, excluding stopwords."""
    stopwords = {
        "what",
        "which",
        "when",
        "where",
        "who",
        "how",
        "is",
        "are",
        "the",
        "a",
        "an",
        "for",
        "this",
        "that",
        "do",
        "does",
        "can",
        "will",
        "to",
        "of",
        "in",
        "on",
        "at",
        "by",
        "with",
        "about",
        "from",
        "and",
        "or",
        "should",
        "use",
        "using",
        "i",
        "you",
        "we",
        "they",
        "it",
        "my",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "having",
        "next",
        "upcoming",
        "coming",
        "new",
        "latest",
        "recent",
        "current",
    }
    words = set(re.findall(r"\b[a-z]{3,}\b", text.lower()))
    return {w for w in words if w not in stopwords}


def generate_answer(
    llm, query: str, retrieved: List[Chunk]
) -> Tuple[str, List[Dict[str, str]]]:
    logger = get_logger(__name__)
    logger.info(
        "generate_answer_start",
        extra={"query": query, "retrieved_chunks": len(retrieved)},
    )
    if not retrieved:
        logger.warning("generate_answer_no_context", extra={"query": query})
        return "No context retrieved to answer the question.", []

    # Check if retrieved chunks are relevant to the query
    # Use meaningful keyword extraction for better matching
    query_keywords = _extract_meaningful_words(query)
    logger.debug(
        "generate_answer_keywords",
        extra={"query_keywords": list(query_keywords)[:10]},
    )
    relevant_chunks = []

    for chunk in retrieved:
        chunk_text = chunk.text or ""
        chunk_keywords = _extract_meaningful_words(chunk_text)

        # Check for keyword overlap
        overlap = query_keywords & chunk_keywords

        # If there's overlap OR query has no extractable keywords, include the chunk
        # This handles cases where the query is very short or generic
        if overlap or not query_keywords:
            relevant_chunks.append(chunk)
            logger.debug(
                "generate_chunk_relevant",
                extra={"chunk_id": chunk.id, "overlap_count": len(overlap)},
            )
        else:
            logger.debug(
                "generate_chunk_filtered",
                extra={"chunk_id": chunk.id, "reason": "no_keyword_overlap"},
            )

    logger.info(
        "generate_chunks_filtered",
        extra={"input_chunks": len(retrieved), "relevant_chunks": len(relevant_chunks)},
    )

    # If no relevant chunks found after filtering, use all chunks but warn in the answer
    if not relevant_chunks and retrieved:
        # Fall back to using all retrieved chunks if filtering is too aggressive
        logger.warning(
            "generate_fallback_to_all_chunks",
            extra={
                "reason": "filtering_too_aggressive",
                "original_count": len(retrieved),
            },
        )
        relevant_chunks = retrieved

    if not relevant_chunks:
        logger.warning("generate_no_relevant_chunks", extra={"query": query})
        return (
            "I could not find relevant information to answer your question. "
            "The search results did not contain content directly related to your query. "
            "Please try rephrasing your question or being more specific.",
            [],
        )

    context_block = _format_context(relevant_chunks)
    logger.debug(
        "generate_context_prepared",
        extra={
            "context_length": len(context_block),
            "chunks_used": len(relevant_chunks),
        },
    )
    prompt = (
        "You are an evidence-based research assistant. Using only the context provided, "
        "write a concise answer to the user question. Cite supporting evidence inline using [n] "
        "where n matches the numbered context entries. Do not fabricate details.\n\n"
        f"User question: {query}\n\nContext:\n{context_block}\n\nAnswer:"
    )
    logger.debug("generate_llm_call", extra={"prompt_length": len(prompt)})
    output = llm(prompt)[0]["generated_text"]
    logger.debug(
        "generate_llm_response",
        extra={"output_length": len(output), "output_preview": output[:200]},
    )
    # Strip prompt echo if present
    answer = (
        output.split("Answer:", 1)[-1].strip()
        if "Answer:" in output
        else output.strip()
    )

    # Check if the generated answer is coherent
    is_coherent = _is_coherent(answer)
    logger.debug(
        "generate_coherence_check",
        extra={"is_coherent": is_coherent, "answer_length": len(answer)},
    )
    if not is_coherent:
        # Provide a structured fallback using the retrieved information
        logger.warning(
            "generate_incoherent_answer",
            extra={"answer_preview": answer[:100], "falling_back": True},
        )
        answer = _generate_fallback_answer(query, relevant_chunks)

    citations = build_citations(relevant_chunks)
    logger.info(
        "generate_answer_complete",
        extra={"answer_length": len(answer), "citations": len(citations)},
    )
    return answer, citations


def _generate_fallback_answer(query: str, chunks: List[Chunk]) -> str:
    """Generate a simple extractive answer when LLM output is incoherent."""
    if not chunks:
        return "Unable to generate a coherent answer from the available context."

    # Extract the most relevant snippets
    snippets = []
    for idx, chunk in enumerate(chunks[:3]):
        meta = chunk.metadata or {}
        title = meta.get("title", "Source")
        # Take first 200 chars of each chunk as a summary
        text = (chunk.text or "")[:300].strip()
        if text:
            snippets.append(f"[{idx + 1}] From '{title}': {text}...")

    if not snippets:
        return "Unable to generate a coherent answer from the available context."

    return (
        f"Based on the search results for '{query}':\n\n"
        + "\n\n".join(snippets)
        + "\n\nNote: The AI was unable to synthesize a full answer. "
        "Please review the sources above for more details."
    )


def verify_answer(llm, draft: str, query: str, retrieved: List[Chunk]) -> str:
    logger = get_logger(__name__)
    logger.info(
        "verify_answer_start",
        extra={
            "query": query,
            "draft_length": len(draft),
            "retrieved_chunks": len(retrieved),
        },
    )
    if not retrieved:
        logger.warning("verify_no_context", extra={"returning": "draft"})
        return draft

    # If draft is already a fallback message, don't try to verify it
    if "Unable to generate" in draft or "AI was unable to synthesize" in draft:
        logger.debug("verify_skip_fallback_message")
        return draft

    # If draft is incoherent, return it as-is (it's already a fallback)
    if not _is_coherent(draft):
        logger.warning("verify_draft_incoherent", extra={"generating_fallback": True})
        return _generate_fallback_answer(query, retrieved)

    context_block = _format_context(retrieved)
    logger.debug(
        "verify_context_prepared",
        extra={"context_length": len(context_block)},
    )
    prompt = (
        "You are a fact-checking verifier. Review the draft answer against the provided context. "
        "Remove or correct any statements that are not directly supported. Preserve inline citations [n] "
        "only when the claim is supported by the corresponding context entry. Keep the answer concise.\n\n"
        f"User question: {query}\n\nContext:\n{context_block}\n\nDraft answer:\n{draft}\n\nVerified answer:"
    )
    logger.debug("verify_llm_call", extra={"prompt_length": len(prompt)})
    output = llm(prompt)[0]["generated_text"]
    logger.debug(
        "verify_llm_response",
        extra={"output_length": len(output), "output_preview": output[:200]},
    )
    verified = (
        output.split("Verified answer:", 1)[-1].strip()
        if "Verified answer:" in output
        else output.strip()
    )

    # Check if verification made things worse
    is_coherent = _is_coherent(verified)
    logger.debug(
        "verify_coherence_check",
        extra={"is_coherent": is_coherent, "verified_length": len(verified)},
    )
    if not is_coherent:
        logger.warning("verify_result_incoherent", extra={"returning": "draft"})
        return draft  # Return original draft if verification produced gibberish

    logger.info(
        "verify_answer_complete",
        extra={"verified_length": len(verified)},
    )
    return verified
