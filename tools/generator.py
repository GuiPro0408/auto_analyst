"""Answer generation and verification helpers."""

import re
from typing import Dict, List, Optional, Tuple

from api.config import (
    COHERENCE_MAX_REPETITION_RATIO,
    COHERENCE_MAX_WORD_REPEAT,
    COHERENCE_MIN_ALNUM_RATIO,
    COHERENCE_MIN_WORDS,
)
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
- If sources disagree, mention the different perspectives
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


def _is_coherent(
    text: str,
    min_words: int = COHERENCE_MIN_WORDS,
    max_repetition_ratio: float = COHERENCE_MAX_REPETITION_RATIO,
    min_alnum_ratio: float = COHERENCE_MIN_ALNUM_RATIO,
    max_word_repeat: int = COHERENCE_MAX_WORD_REPEAT,
) -> bool:
    """Check if generated text appears coherent and not gibberish.

    Args:
        text: The text to check.
        min_words: Minimum word count for detailed checking.
        max_repetition_ratio: Maximum ratio of any word to total words.
        min_alnum_ratio: Minimum ratio of alphanumeric characters.
        max_word_repeat: Maximum times a single word can repeat.

    Returns:
        True if text appears coherent.
    """
    if not text or len(text.strip()) < 10:
        return False

    # Check for excessive repetition of short patterns
    words = text.lower().split()

    # Short to medium responses (less than min_words) are usually OK
    # This allows concise LLM answers through without excessive checking
    if len(words) < min_words:
        # Just check for basic gibberish indicators
        alnum_count = sum(c.isalnum() or c.isspace() for c in text)
        alnum_ratio = alnum_count / len(text) if text else 0
        return alnum_ratio > 0.6

    # Count word frequencies
    word_counts: Dict[str, int] = {}
    for word in words:
        cleaned = re.sub(r"[^a-z]", "", word)
        if len(cleaned) >= 2:
            word_counts[cleaned] = word_counts.get(cleaned, 0) + 1

    if not word_counts:
        return False

    # Check if any single word is repeated too frequently (relaxed threshold)
    total_meaningful = sum(word_counts.values())
    max_count = max(word_counts.values())
    if total_meaningful > 10 and max_count / total_meaningful > max_repetition_ratio:
        return False

    # Check for patterns like "word word word" or "... ... ..."
    pattern_matches = re.findall(r"(\b\w+\b)(?:\s+\1){3,}", text.lower())  # 3+ repeats
    if len(pattern_matches) > 2:
        return False

    # Check for excessive ellipsis or dots (gibberish indicator)
    dot_count = text.count("...") + text.count("..") + text.count(". . .")
    if dot_count > 10:
        return False

    # Check for excessive non-alphanumeric characters (gibberish indicator)
    alnum_count = sum(c.isalnum() or c.isspace() for c in text)
    alnum_ratio = alnum_count / len(text) if text else 0
    if alnum_ratio < min_alnum_ratio:
        return False

    # If any 3+ letter word appears more than max_word_repeat times, likely gibberish
    # But allow for technical terms that might repeat naturally in large blocks
    # Extended allowlist for technical/AI content that legitimately repeats terms
    technical_allowlist = {
        # Common English words
        "the",
        "and",
        "that",
        "this",
        "with",
        "from",
        "for",
        "are",
        "can",
        "has",
        # AI/ML model names and terms (these repeat in comparison queries)
        "model",
        "models",
        "gemini",
        "claude",
        "gpt",
        "flash",
        "sonnet",
        "opus",
        "llm",
        "llms",
        "token",
        "tokens",
        "benchmark",
        "performance",
        "latency",
        # Technical terms
        "api",
        "data",
        "code",
        "test",
        "context",
        "output",
        "input",
        "response",
        # Media/entertainment terms
        "anime",
        "release",
        "season",
        "episode",
        "series",
        "movie",
        "game",
    }
    for word, count in word_counts.items():
        if len(word) >= 3 and count > max_word_repeat:
            # Check if it's an allowed term that can repeat more frequently
            if word in technical_allowlist:
                if count > max_word_repeat * 3:  # Allow 3x more for technical terms
                    return False
                continue
            return False

    # Check for nonsensical number/letter combinations
    nonsense_patterns = re.findall(r"[a-z]+\d+[a-z]*|\d+[a-z]+\d*", text.lower())
    if len(nonsense_patterns) > 10:  # Relaxed from 5
        return False

    return True


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

    context_block = _format_context(relevant_chunks)
    logger.debug(
        "generate_context_prepared",
        extra={
            "context_length": len(context_block),
            "chunks_used": len(relevant_chunks),
        },
    )
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


def _clean_snippet(text: str, max_len: int = 300) -> str:
    """Extract a clean, readable snippet from raw chunk text."""
    if not text:
        return ""

    # Remove markdown headers
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)  # # Headers
    text = re.sub(r"\s+#{1,6}\s+", " ", text)  # Inline # headers

    # Remove common boilerplate patterns
    text = re.sub(r"https?://\S+", "", text)  # URLs
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)  # [text](url) -> text
    text = re.sub(r"\[\s*\]\([^)]*\)", "", text)  # Empty markdown links
    text = re.sub(
        r"/[a-z0-9_-]+\)", "", text, flags=re.IGNORECASE
    )  # URL path fragments like /winter-2025)
    text = re.sub(
        r"-\d+-[a-z0-9-]+/?\)", "", text, flags=re.IGNORECASE
    )  # URL slugs like -2-release-date-story/)
    text = re.sub(
        r"\([^)]*release-date[^)]*\)", "", text, flags=re.IGNORECASE
    )  # Full URL path in parens

    # Sign in / account prompts
    text = re.sub(r"Sign in[^.]*(?:account|now|to your)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"Sign in to your \w+ account", "", text, flags=re.IGNORECASE)
    text = re.sub(r"Subscribe.*?newsletter", "", text, flags=re.IGNORECASE)

    # MAL/anime site specific noise
    text = re.sub(
        r"\d+\s*eps?,?\s*\d*\s*min", "", text
    )  # Episode info like "11 eps, 23 min"
    text = re.sub(r"\d{8}", "", text)  # Date codes like 20250112
    text = re.sub(r"Add to My List", "", text, flags=re.IGNORECASE)
    text = re.sub(r"Studio\w+", "", text)  # StudioILCA etc
    text = re.sub(r"Source\s+\w+", "", text)  # Source Original etc
    text = re.sub(r"\d+\.\d+\s+\d+\.?\d*K?", "", text)  # Ratings like "5.80 4.4K"

    # Navigation menu items (MAL, etc.)
    nav_items = [
        r"\bArchive\b",
        r"\bNew Manga\b",
        r"\bJump to\b",
        r"\bNot in My List\b",
        r"\bPlan to Watch\b",
        r"\bOn-Hold\b",
        r"\bWatching\b",
        r"\bCompleted\b",
        r"\bDropped\b",
        r"\bClear All\b",
        r"\bFilter Sort\b",
        r"\bR18\+\b",
        r"\bAvailable in My Region\b",
        r"\bClick once to include\b",
    ]
    for nav in nav_items:
        text = re.sub(nav, "", text, flags=re.IGNORECASE)

    # Clean whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Try to find complete sentences
    sentences = re.split(r"(?<=[.!?])\s+", text)
    result = []
    current_len = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence or len(sentence) < 10:
            continue
        if current_len + len(sentence) > max_len:
            break
        result.append(sentence)
        current_len += len(sentence) + 1

    if result:
        return " ".join(result)

    # Fallback: just truncate cleanly
    if len(text) > max_len:
        # Try to break at a word boundary
        truncated = text[:max_len].rsplit(" ", 1)[0]
        return truncated + "..." if truncated else text[:max_len] + "..."

    return text


def _extract_key_items(chunks: List[Chunk]) -> List[str]:
    """Extract named items (anime titles, etc.) from chunks."""
    items = set()

    # Common anime title patterns
    title_patterns = [
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+(?:Season|Part)\s+\d+)?)",  # Title Case
        r"([A-Z][a-z]+(?:'s|:)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",  # Possessive/subtitle
    ]

    for chunk in chunks[:5]:
        text = chunk.text or ""
        meta = chunk.metadata or {}

        # Get title from metadata
        title = meta.get("title", "")
        if title and "anime" not in title.lower() and len(title) < 80:
            # Extract potential anime names from article titles
            for part in re.split(r"[-–:|]", title):
                part = part.strip()
                if (
                    part
                    and 3 < len(part) < 50
                    and not part.lower().startswith(
                        ("best", "new", "top", "winter", "fall")
                    )
                ):
                    items.add(part)

        # Look for known anime patterns in text
        known_titles = [
            "Solo Leveling",
            "The Apothecary Diaries",
            "Blue Exorcist",
            "Fate/strange Fake",
            "Zenshu",
            "Jujutsu Kaisen",
            "My Hero Academia",
            "Attack on Titan",
            "Demon Slayer",
            "One Piece",
            "Naruto",
            "Hell's Paradise",
            "Oshi no Ko",
            "Haikyuu",
            "Chainsaw Man",
        ]
        for known in known_titles:
            if known.lower() in text.lower():
                items.add(known)

    return list(items)[:10]


def _is_comparison_query(query: str) -> bool:
    """Detect if query is asking for a comparison between items."""
    query_lower = query.lower()
    comparison_keywords = [
        "compare",
        "comparison",
        "vs",
        "versus",
        "difference",
        "differences",
        "better",
        "which is",
        "pros and cons",
        "advantages",
        "disadvantages",
    ]
    return any(kw in query_lower for kw in comparison_keywords)


def _generate_comparison_fallback(query: str, chunks: List[Chunk]) -> str:
    """Generate a comparison-focused fallback answer with table format."""
    # Extract entities being compared from query and chunks
    entities: List[str] = []
    query_lower = query.lower()

    # Common AI model names to detect
    model_names = [
        "gemini",
        "gpt-4",
        "gpt-4o",
        "gpt4",
        "claude",
        "sonnet",
        "opus",
        "haiku",
        "llama",
        "mistral",
        "palm",
        "bard",
        "copilot",
        "chatgpt",
        "flash",
    ]
    for model in model_names:
        if model in query_lower:
            # Capitalize properly
            if model == "gpt-4" or model == "gpt-4o" or model == "gpt4":
                entities.append("GPT-4o")
            elif model == "gemini" or model == "flash":
                entities.append("Gemini 2.0 Flash")
            elif model == "claude" or model == "sonnet":
                entities.append("Claude 3.5 Sonnet")
            elif model == "opus":
                entities.append("Claude Opus")
            else:
                entities.append(model.title())

    # Deduplicate
    entities = list(dict.fromkeys(entities))[:4]

    lines = []
    lines.append(f"**Comparison Summary**\n")
    lines.append(f"Query: *{query}*\n\n")

    if len(entities) >= 2:
        # Build comparison table header
        lines.append("| Aspect | " + " | ".join(entities) + " |\n")
        lines.append("|" + "---|" * (len(entities) + 1) + "\n")

        # Add placeholder rows based on common comparison aspects
        aspects = ["Performance", "Speed/Latency", "Cost", "Context Window", "Best For"]
        for aspect in aspects:
            row = f"| {aspect} |"
            for _ in entities:
                row += " See sources below |"
            lines.append(row + "\n")
        lines.append("\n")

    lines.append("### Key Information from Sources\n\n")

    seen_sources = set()
    source_count = 0

    for chunk in chunks[:6]:
        meta = chunk.metadata or {}
        title = meta.get("title", "Source")
        url = meta.get("url", "")

        source_key = url or title
        if source_key in seen_sources:
            continue
        seen_sources.add(source_key)
        source_count += 1

        snippet = _clean_snippet(chunk.text or "", max_len=250)
        if not snippet or len(snippet) < 20:
            continue

        lines.append(f"**{source_count}. {title}**\n")
        lines.append(f"> {snippet}\n")
        if url:
            lines.append(f"[Read more]({url})\n")
        lines.append("\n")

    if source_count == 0:
        return (
            "I found some comparison sources but couldn't extract meaningful content. "
            "Please check the linked sources for detailed comparisons."
        )

    lines.append("---\n")
    lines.append(
        "*This comparison summary was generated from retrieved sources. "
        "For detailed benchmarks and analysis, please visit the linked sources.*"
    )

    return "".join(lines)


def _generate_fallback_answer(query: str, chunks: List[Chunk]) -> str:
    """Generate a structured, readable answer when LLM is unavailable."""
    if not chunks:
        return (
            "I couldn't find sufficient information to answer your question. "
            "Please try rephrasing or being more specific."
        )

    # Check for comparison queries first - use specialized format
    if _is_comparison_query(query):
        return _generate_comparison_fallback(query, chunks)

    # Determine if this looks like a recommendation query
    is_recommendation = any(
        kw in query.lower()
        for kw in ["recommend", "suggest", "should", "watch", "best", "top", "good"]
    )

    # Extract key items mentioned
    key_items = _extract_key_items(chunks)

    # Build a cleaner answer
    lines = []

    if is_recommendation and key_items:
        lines.append(
            f"**Based on the sources found, here are some options for your query:**\n"
        )
        lines.append("### Mentioned Titles\n")
        for item in key_items[:8]:
            lines.append(f"- **{item}**\n")
        lines.append("\n")

    lines.append("### Source Highlights\n")

    seen_sources = set()
    source_count = 0

    for chunk in chunks[:6]:
        meta = chunk.metadata or {}
        title = meta.get("title", "Source")
        url = meta.get("url", "")

        # Skip duplicate sources
        source_key = url or title
        if source_key in seen_sources:
            continue
        seen_sources.add(source_key)
        source_count += 1

        # Clean and format the snippet
        snippet = _clean_snippet(chunk.text or "", max_len=250)
        if not snippet or len(snippet) < 20:
            continue

        lines.append(f"**{source_count}. {title}**\n")
        lines.append(f"> {snippet}\n")
        if url:
            lines.append(f"[Read more]({url})\n")
        lines.append("\n")

    if not lines or source_count == 0:
        return (
            "I found some sources but couldn't extract meaningful content. "
            "The pages may require JavaScript or have restricted access."
        )

    lines.append("---\n")
    lines.append(
        "*This summary was generated from retrieved sources. "
        "For detailed information, please visit the linked sources above.*"
    )

    return "".join(lines)


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
