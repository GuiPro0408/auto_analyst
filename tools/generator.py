"""Answer generation and verification helpers."""

import re
from typing import Dict, List, Optional, Tuple

from api.backend_utils import is_limited_backend, is_local_backend
from api.config import LLM_BACKEND
from api.logging_setup import get_logger
from api.state import Chunk
from tools.text_utils import STOPWORDS, extract_keywords, requires_structured_list

# Query type constants (imported from query_classifier if needed elsewhere)
QUERY_TYPE_FACTUAL = "factual"
QUERY_TYPE_RECOMMENDATION = "recommendation"
QUERY_TYPE_CREATIVE = "creative"

# =============================================================================
# PROMPT TEMPLATES BY QUERY TYPE
# =============================================================================

# Compact prompt for local/CPU LLM to reduce token overhead
PROMPT_LOCAL_COMPACT = """Answer the question using only the context below. Be concise but complete. Cite sources with [n].

Question: {query}

Context:
{context_block}

Answer:"""

PROMPT_FACTUAL = """You are a highly capable, evidence-based research assistant. Using the context provided, \
write a sophisticated, well-structured, and comprehensive answer to the user question.

Guidelines:
- Aim for a professional, analytical tone
- Organize your response with clear sections using **bold headers**
- Provide deep, detailed explanations rather than surface-level summaries
- Use bullet points or numbered lists to present complex information clearly
{list_instruction}\
- Support every claim with inline citations [n] using the context entries
- Be specific: include dates, names, figures, and technical details from the sources
- If sources present conflicting information, acknowledge both perspectives with their respective citations
- If the context is insufficient to fully answer the question, clearly state what aspects cannot be addressed and why
- Do not fabricate information; stay strictly within the context provided

{context_instruction}User question: {query}

Context:
{context_block}

Answer:"""

PROMPT_RECOMMENDATION = """You are an expert advisor providing curated recommendations. \
Examine the search results thoroughly and combine them with your extensive latent knowledge \
to provide high-quality, insightful suggestions.

Guidelines:
- Provide specific, actionable recommendations with descriptive detail
- For each recommendation, explain the reasoning and why it's a good fit
- Use **bold headers** for each recommendation section
- Utilize bullet points for features, pros/cons, or additional details
{list_instruction}\
- Use [n] citations whenever a detail or specific item is found in the search results
- Add your own expert perspective to add value beyond the raw search data
- If sources present conflicting opinions or reviews, acknowledge both perspectives
- If the context lacks sufficient options or details, clearly note what additional information would help
- Ensure the response is engaging, detailed, and worth reading

{context_instruction}User question: {query}

Reference context (use as primary evidence):
{context_block}

Detailed Recommendations:"""

PROMPT_CREATIVE = """You are a helpful assistant answering a creative or open-ended question. \
Use the provided context for reference and inspiration, but feel free to provide \
your own insights and explanations.

Guidelines:
- Be informative and engaging
- Structure your response clearly
{list_instruction}\
- Reference sources with [n] when using specific information from them
- You may expand beyond the provided context when helpful
- If sources offer different viewpoints, explore the nuances of each
- If the context is limited, acknowledge it while still providing your best response

{context_instruction}User question: {query}

Reference context:
{context_block}

Response:"""

# Map query types to their prompts
PROMPT_TEMPLATES = {
    QUERY_TYPE_FACTUAL: PROMPT_FACTUAL,
    QUERY_TYPE_RECOMMENDATION: PROMPT_RECOMMENDATION,
    QUERY_TYPE_CREATIVE: PROMPT_CREATIVE,
}

# Response delimiters for stripping prompt echoes
RESPONSE_DELIMITERS = {
    QUERY_TYPE_FACTUAL: "Answer:",
    QUERY_TYPE_RECOMMENDATION: "Detailed Recommendations:",
    QUERY_TYPE_CREATIVE: "Response:",
}


def _format_context(chunks: List[Chunk]) -> str:
    lines = []
    for idx, chunk in enumerate(chunks):
        meta = chunk.metadata or {}
        title = meta.get("title") or "Source"
        url = meta.get("url") or ""
        lines.append(f"[{idx + 1}] {title} ({url}) :: {chunk.text}")
    return "\n".join(lines)


def _remap_citation_markers(answer: str, index_map: Dict[int, int]) -> str:
    """Remap inline [n] markers after citation de-duplication."""

    def _replace(match: re.Match[str]) -> str:
        old_idx = int(match.group(1))
        new_idx = index_map.get(old_idx)
        return f"[{new_idx}]" if new_idx else match.group(0)

    return re.sub(r"\[(\d+)\]", _replace, answer)


def build_citations(
    chunks: List[Chunk], answer: Optional[str] = None
) -> Tuple[str, List[Dict[str, str]]]:
    citations: List[Dict[str, str]] = []
    index_map: Dict[int, int] = {}
    seen_urls: Dict[str, int] = {}

    for idx, chunk in enumerate(chunks):
        meta = chunk.metadata or {}
        raw_url = meta.get("url", "")
        normalized_url = raw_url.split("#")[0] if raw_url else raw_url
        if normalized_url and normalized_url in seen_urls:
            index_map[idx + 1] = seen_urls[normalized_url]
            continue

        marker = len(citations) + 1
        index_map[idx + 1] = marker
        if normalized_url:
            seen_urls[normalized_url] = marker
        citations.append(
            {
                "marker": f"[{marker}]",
                "url": normalized_url,
                "title": meta.get("title", "Source"),
                "media_type": meta.get("media_type", "text"),
            }
        )

    remapped_answer = answer or ""
    if answer is not None:
        remapped_answer = _remap_citation_markers(answer, index_map)

    return remapped_answer, citations


def generate_answer(
    llm,
    query: str,
    retrieved: List[Chunk],
    conversation_context: Optional[str] = None,
    query_type: str = QUERY_TYPE_FACTUAL,
) -> Tuple[str, List[Dict[str, str]]]:
    logger = get_logger(__name__)
    logger.info(
        "generate_answer_start",
        extra={
            "query": query,
            "retrieved_chunks": len(retrieved),
            "conversation_context": bool(conversation_context),
            "query_type": query_type,
        },
    )
    if not retrieved:
        logger.warning("generate_answer_no_context", extra={"query": query})
        return "No context retrieved to answer the question.", []

    # Check if retrieved chunks are relevant to the query
    query_keywords = extract_keywords(query, stopwords=STOPWORDS)
    logger.debug(
        "generate_query_keywords",
        extra={"query_keywords": list(query_keywords)[:10]},
    )

    relevant_chunks: List[Chunk] = []
    for chunk in retrieved:
        chunk_text = chunk.text or ""
        chunk_keywords = extract_keywords(chunk_text, stopwords=STOPWORDS)

        # Check for keyword overlap
        overlap = query_keywords & chunk_keywords

        # If there's overlap OR query has no extractable keywords, include the chunk
        # This handles cases where the query is very short or generic
        if overlap or not query_keywords:
            relevant_chunks.append(chunk)

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

    # Trim context for backends with token limits
    if is_limited_backend():
        max_chunks = 6 if not is_local_backend() else 4
        max_chunk_chars = 800 if not is_local_backend() else 1200
        if len(relevant_chunks) > max_chunks:
            logger.info(
                "generate_limited_trim_chunks",
                extra={
                    "original_chunks": len(relevant_chunks),
                    "kept_chunks": max_chunks,
                    "backend": LLM_BACKEND,
                },
            )
        trimmed_chunks: List[Chunk] = []
        for chunk in relevant_chunks[:max_chunks]:
            text = chunk.text or ""
            if len(text) > max_chunk_chars:
                text = text[:max_chunk_chars]
            trimmed_chunks.append(
                Chunk(
                    id=chunk.id,
                    text=text,
                    metadata=chunk.metadata,
                )
            )
        relevant_chunks = trimmed_chunks

    context_block = _format_context(relevant_chunks)
    logger.debug(
        "generate_context_prepared",
        extra={
            "context_length": len(context_block),
            "chunks_used": len(relevant_chunks),
        },
    )

    # Use compact prompt for local backend to reduce token overhead
    if is_local_backend():
        prompt = PROMPT_LOCAL_COMPACT.format(
            query=query,
            context_block=context_block,
        )
        response_delimiter = "Answer:"
        logger.debug(
            "generate_using_compact_prompt",
            extra={"prompt_length": len(prompt)},
        )
    else:
        context_instruction = ""
        if conversation_context:
            trimmed_context = " ".join(conversation_context.split())
            if len(trimmed_context) > 800:
                trimmed_context = trimmed_context[-800:]
            context_instruction = (
                "Prior conversation summary (use for continuity when relevant):\n"
                f"{trimmed_context}\n\n"
            )

        list_instruction = ""
        if requires_structured_list(query):
            list_instruction = (
                "- Present the answer as a bullet or numbered list of items. "
                "For each item include the name/title and any available date/status, "
                "with an inline citation [n] for that item.\n"
            )

        # Select prompt template based on query type
        prompt_template = PROMPT_TEMPLATES.get(query_type, PROMPT_FACTUAL)
        response_delimiter = RESPONSE_DELIMITERS.get(query_type, "Answer:")

        prompt = prompt_template.format(
            list_instruction=list_instruction,
            context_instruction=context_instruction,
            query=query,
            context_block=context_block,
        )

    logger.debug(
        "generate_llm_call",
        extra={"prompt_length": len(prompt), "query_type": query_type},
    )
    output = llm(prompt)[0]["generated_text"]
    logger.debug(
        "generate_llm_response",
        extra={"output_length": len(output)},
    )
    # Strip prompt echo if present
    answer = (
        output.split(response_delimiter, 1)[-1].strip()
        if response_delimiter in output
        else output.strip()
    )

    # NOTE: Coherence check disabled - Tavily provides high-quality context,
    # and the fallback templates produce worse output ("See sources below").
    # The LLM output with good context is preferable even if slightly imperfect.
    # If quality issues arise, improve the prompt rather than falling back.
    logger.debug(
        "generate_coherence_check_skipped",
        extra={"answer_length": len(answer)},
    )

    # Strip LLM-generated references section (UI displays citations separately)
    answer = _strip_references_section(answer)

    answer, citations = build_citations(relevant_chunks, answer)
    logger.info(
        "generate_answer_complete",
        extra={"answer_length": len(answer), "citations": len(citations)},
    )
    return answer, citations


def generate_answer_stream(
    llm,
    query: str,
    retrieved: List[Chunk],
    conversation_context: Optional[str] = None,
    query_type: str = QUERY_TYPE_FACTUAL,
):
    """Stream answer generation token by token.

    Yields:
        Tuple[str, bool, List[Dict[str, str]]]: (partial_text, is_complete, citations)
        - partial_text: accumulated answer text so far
        - is_complete: True on final yield
        - citations: populated only on final yield
    """
    from typing import Generator

    logger = get_logger(__name__)
    logger.info(
        "generate_answer_stream_start",
        extra={
            "query": query,
            "retrieved_chunks": len(retrieved),
            "conversation_context": bool(conversation_context),
            "query_type": query_type,
        },
    )

    if not retrieved:
        logger.warning("generate_answer_stream_no_context", extra={"query": query})
        yield ("No context retrieved to answer the question.", True, [])
        return

    # Check if LLM supports streaming
    if not hasattr(llm, "stream"):
        logger.info("generate_answer_stream_fallback_to_sync")
        answer, citations = generate_answer(
            llm, query, retrieved, conversation_context, query_type
        )
        yield (answer, True, citations)
        return

    # Prepare context (same as generate_answer)
    query_keywords = extract_keywords(query, stopwords=STOPWORDS)
    relevant_chunks: List[Chunk] = []
    for chunk in retrieved:
        chunk_text = chunk.text or ""
        chunk_keywords = extract_keywords(chunk_text, stopwords=STOPWORDS)
        overlap = query_keywords & chunk_keywords
        if overlap or not query_keywords:
            relevant_chunks.append(chunk)

    if not relevant_chunks and retrieved:
        relevant_chunks = retrieved

    if not relevant_chunks:
        yield (
            "I could not find relevant information to answer your question.",
            True,
            [],
        )
        return

    # Trim context for limited backends
    if is_limited_backend():
        max_chunks = 6 if not is_local_backend() else 4
        max_chunk_chars = 800 if not is_local_backend() else 1200
        trimmed_chunks: List[Chunk] = []
        for chunk in relevant_chunks[:max_chunks]:
            text = chunk.text or ""
            if len(text) > max_chunk_chars:
                text = text[:max_chunk_chars]
            trimmed_chunks.append(
                Chunk(id=chunk.id, text=text, metadata=chunk.metadata)
            )
        relevant_chunks = trimmed_chunks

    context_block = _format_context(relevant_chunks)

    # Build prompt (same logic as generate_answer)
    if is_local_backend():
        prompt = PROMPT_LOCAL_COMPACT.format(query=query, context_block=context_block)
        response_delimiter = "Answer:"
    else:
        context_instruction = ""
        if conversation_context:
            trimmed_context = " ".join(conversation_context.split())
            if len(trimmed_context) > 800:
                trimmed_context = trimmed_context[-800:]
            context_instruction = (
                "Prior conversation summary (use for continuity when relevant):\n"
                f"{trimmed_context}\n\n"
            )

        list_instruction = ""
        if requires_structured_list(query):
            list_instruction = (
                "- Present the answer as a bullet or numbered list of items. "
                "For each item include the name/title and any available date/status, "
                "with an inline citation [n] for that item.\n"
            )

        prompt_template = PROMPT_TEMPLATES.get(query_type, PROMPT_FACTUAL)
        response_delimiter = RESPONSE_DELIMITERS.get(query_type, "Answer:")

        prompt = prompt_template.format(
            list_instruction=list_instruction,
            context_instruction=context_instruction,
            query=query,
            context_block=context_block,
        )

    # Stream tokens
    accumulated = ""
    found_delimiter = False

    for token in llm.stream(prompt):
        accumulated += token

        # Strip prompt echo once we find delimiter
        if not found_delimiter and response_delimiter in accumulated:
            found_delimiter = True
            accumulated = accumulated.split(response_delimiter, 1)[-1].lstrip()

        # Yield partial (without citations until complete)
        yield (accumulated, False, [])

    # Final processing
    final_answer = _strip_references_section(accumulated)
    final_answer, citations = build_citations(relevant_chunks, final_answer)

    logger.info(
        "generate_answer_stream_complete",
        extra={"answer_length": len(final_answer), "citations": len(citations)},
    )
    yield (final_answer, True, citations)


def verify_answer_stream(
    llm,
    draft: str,
    query: str,
    retrieved: List[Chunk],
    conversation_context: Optional[str] = None,
):
    """Stream answer verification token by token.

    Yields:
        Tuple[str, bool]: (partial_text, is_complete)
        - partial_text: accumulated verified text so far
        - is_complete: True on final yield
    """
    logger = get_logger(__name__)
    logger.info(
        "verify_answer_stream_start",
        extra={
            "draft_length": len(draft),
            "retrieved_chunks": len(retrieved),
        },
    )

    # Check if LLM supports streaming
    if not hasattr(llm, "stream"):
        logger.info("verify_answer_stream_fallback_to_sync")
        verified = verify_answer(llm, draft, query, retrieved, conversation_context)
        yield (verified, True)
        return

    if not retrieved:
        yield (draft, True)
        return

    context_block = _format_context(retrieved)

    context_instruction = ""
    if conversation_context:
        trimmed_context = " ".join(conversation_context.split())
        if len(trimmed_context) > 800:
            trimmed_context = trimmed_context[-800:]
        context_instruction = (
            "Prior conversation summary (use for continuity when relevant):\n"
            f"{trimmed_context}\n\n"
        )

    prompt = (
        "You are a fact-checking verifier. Review the draft answer against the provided context. "
        "Remove or correct any statements that are not directly supported by the context. "
        "Preserve the structure, formatting (headers, bullet points, lists), and level of detail from the draft. "
        "Keep inline citations [n] only when the claim is supported by the corresponding context entry. "
        "Do not shorten or oversimplify the answer - maintain comprehensive coverage.\n\n"
        f"{context_instruction}"
        f"User question: {query}\n\n"
        f"Context:\n{context_block}\n\n"
        f"Draft answer:\n{draft}\n\n"
        "Verified answer:"
    )

    # Stream tokens
    accumulated = ""
    found_delimiter = False
    response_delimiter = "Verified answer:"

    for token in llm.stream(prompt):
        accumulated += token

        # Strip prompt echo once we find delimiter
        if not found_delimiter and response_delimiter in accumulated:
            found_delimiter = True
            accumulated = accumulated.split(response_delimiter, 1)[-1].lstrip()

        yield (accumulated, False)

    # Final cleanup
    final_answer = accumulated.strip()
    logger.info(
        "verify_answer_stream_complete",
        extra={"verified_length": len(final_answer)},
    )
    yield (final_answer, True)


def _strip_references_section(text: str) -> str:
    """Remove LLM-generated References/Sources sections from the answer.

    The UI displays citations separately, so we strip any references the LLM
    adds to avoid duplication.
    """
    # Common patterns for references sections (order matters - more specific first)
    patterns = [
        # Markdown headers: ### References, ## Sources, etc.
        r"\n+#{1,6}\s*References\s*\n[\s\S]*$",
        r"\n+#{1,6}\s*Sources\s*\n[\s\S]*$",
        r"\n+#{1,6}\s*Citations\s*\n[\s\S]*$",
        # Bold or plain text: **References:** or References:
        r"\n+\*?\*?References:?\*?\*?\s*\n[\s\S]*$",
        r"\n+\*?\*?Sources:?\*?\*?\s*\n[\s\S]*$",
        r"\n+\*?\*?Citations:?\*?\*?\s*\n[\s\S]*$",
        # Horizontal rule followed by citations: --- \n [1]...
        r"\n+---+\s*\n\s*\[\d+\][\s\S]*$",
    ]

    result = text
    for pattern in patterns:
        result = re.sub(pattern, "", result, flags=re.IGNORECASE)

    return result.rstrip()


def verify_answer(
    llm,
    draft: str,
    query: str,
    retrieved: List[Chunk],
    conversation_context: Optional[str] = None,
) -> str:
    logger = get_logger(__name__)
    logger.info(
        "verify_answer_start",
        extra={
            "query": query,
            "draft_length": len(draft),
            "retrieved_chunks": len(retrieved),
            "conversation_context": bool(conversation_context),
        },
    )

    if is_limited_backend():
        logger.info("verify_answer_skipped_limited_backend")
        return draft
    if not retrieved:
        logger.warning("verify_no_context", extra={"returning": "draft"})
        return draft

    # If draft is already a fallback message, don't try to verify it
    if "Unable to generate" in draft or "AI was unable to synthesize" in draft:
        logger.debug("verify_skip_fallback_message")
        return draft

    # NOTE: Draft coherence check disabled - if draft got through generate_answer,
    # we trust it and proceed with verification.
    logger.debug("verify_draft_coherence_skipped")

    context_block = _format_context(retrieved)
    logger.debug(
        "verify_context_prepared",
        extra={"context_length": len(context_block)},
    )
    context_instruction = ""
    if conversation_context:
        trimmed_context = " ".join(conversation_context.split())
        if len(trimmed_context) > 800:
            trimmed_context = trimmed_context[-800:]
        context_instruction = (
            "Prior conversation summary (maintain continuity where appropriate):\n"
            f"{trimmed_context}\n\n"
        )

    prompt = (
        "You are a fact-checking verifier. Review the draft answer against the provided context. "
        "Remove or correct any statements that are not directly supported by the context. "
        "Preserve the structure, formatting (headers, bullet points, lists), and level of detail from the draft. "
        "Keep inline citations [n] only when the claim is supported by the corresponding context entry. "
        "Do not shorten or oversimplify the answer - maintain comprehensive coverage.\n\n"
        f"{context_instruction}User question: {query}\n\nContext:\n{context_block}\n\nDraft answer:\n{draft}\n\nVerified answer:"
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

    # NOTE: Verification coherence check disabled - LLM verification with good
    # Tavily context should produce usable output. If not, we still prefer
    # the verified output over reverting to draft (which may have citation issues).
    logger.debug(
        "verify_coherence_check_skipped",
        extra={"verified_length": len(verified)},
    )

    logger.info(
        "verify_answer_complete",
        extra={"verified_length": len(verified)},
    )
    return verified
