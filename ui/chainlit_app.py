"""Chainlit UI for Auto-Analyst.

Production chat interface with streaming support.

Run from the project root:
    chainlit run ui/chainlit_app.py -w
"""

import sys
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Generator, List
from urllib.parse import urlparse

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chainlit as cl  # type: ignore[import-not-found]

# Initialize logging BEFORE Chainlit's logging takes over
# This ensures our file handler is added
from api.logging_setup import configure_logging, get_logger

configure_logging()  # Force logging setup with file handler
logger = get_logger(__name__)

# =============================================================================
# DATA PERSISTENCE (Chat History)
# =============================================================================
try:
    from ui.data_layer import get_data_layer

    cl.data_layer = get_data_layer()
    logger.info("data_layer_initialized", extra={"backend": "sqlite"})
except ImportError as e:
    logger.warning("data_layer_unavailable", extra={"error": str(e)})
except Exception as e:
    logger.warning("data_layer_init_failed", extra={"error": str(e)})

from api.config import (
    CONVERSATION_MEMORY_TURNS,
    DEFAULT_EMBED_MODEL,
    DEFAULT_LLM_MODEL,
    TOP_K_RESULTS,
)
from api.graph import run_research_streaming
from api.memory import trim_history
from api.state import ConversationTurn
from tools.models import load_llm
from tools.retriever import build_vector_store

logger.info("chainlit_app_initialized")

# =============================================================================
# AVATAR CONFIGURATION
# =============================================================================
ASSISTANT_AVATAR = "/public/assistant.svg"
USER_AVATAR = "/public/user.svg"


def build_source_card(marker: str, title: str, url: str, snippet: str = "") -> str:
    """Build a rich source card with favicon and snippet.

    Args:
        marker: The citation marker (e.g., "[1]").
        title: The title of the source.
        url: The URL of the source.
        snippet: An optional snippet of text from the source.

    Returns:
        A markdown string representing the source card.
    """
    domain = ""
    if url:
        try:
            parsed = urlparse(url)
            domain = parsed.netloc or ""
        except Exception:
            pass

    # Build card HTML-like markdown
    card_lines = [
        f"**{marker}** ",
        f"[**{title}**]({url})" if url else f"**{title}**",
    ]

    if domain:
        card_lines.append(f"  \n🌐 `{domain}`")

    if snippet:
        # Truncate snippet to ~100 chars
        short_snippet = snippet[:100] + "..." if len(snippet) > 100 else snippet
        card_lines.append(f"  \n_{short_snippet}_")

    return "".join(card_lines)


# =============================================================================
# CHAT SETTINGS
# =============================================================================
@cl.set_chat_profiles
async def chat_profiles():
    """Define available chat profiles."""
    return [
        cl.ChatProfile(
            name="Research Assistant",
            markdown_description="Autonomous research with RAG-powered answers",
            icon="🔬",
        ),
    ]


@cl.on_settings_update
async def settings_update(settings):
    """Handle settings changes from the UI."""
    cl.user_session.set("top_k", settings.get("top_k", TOP_K_RESULTS))
    cl.user_session.set("llm_model", settings.get("llm_model", DEFAULT_LLM_MODEL))
    cl.user_session.set("embed_model", settings.get("embed_model", DEFAULT_EMBED_MODEL))


# =============================================================================
# CHAT START
# =============================================================================
@cl.on_chat_start
async def on_chat_start():
    """Initialize the chat session."""
    # Set default settings
    cl.user_session.set("top_k", TOP_K_RESULTS)
    cl.user_session.set("llm_model", DEFAULT_LLM_MODEL)
    cl.user_session.set("embed_model", DEFAULT_EMBED_MODEL)
    cl.user_session.set("conversation_history", [])

    # Configure chat settings panel
    settings = await cl.ChatSettings(
        [
            cl.input_widget.Slider(
                id="top_k",
                label="Number of sources",
                initial=TOP_K_RESULTS,
                min=2,
                max=10,
                step=1,
            ),
            cl.input_widget.TextInput(
                id="llm_model",
                label="LLM Model",
                initial=DEFAULT_LLM_MODEL,
            ),
            cl.input_widget.TextInput(
                id="embed_model",
                label="Embedding Model",
                initial=DEFAULT_EMBED_MODEL,
            ),
        ]
    ).send()

    # Welcome message with custom avatar
    await cl.Message(
        content=(
            "# 🔬 Auto-Analyst\n\n"
            "I'm your autonomous research assistant powered by RAG. "
            "Ask me any research question and I'll:\n\n"
            "1. **Plan** targeted search queries\n"
            "2. **Search** multiple sources\n"
            "3. **Fetch** and process documents\n"
            "4. **Generate** a comprehensive answer with citations\n"
            "5. **Verify** facts against sources\n\n"
            "What would you like to research?"
        ),
        author="Auto-Analyst",
    ).send()


# =============================================================================
# MESSAGE HANDLER
# =============================================================================
@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming user messages."""
    query = message.content.strip()
    if not query:
        await cl.Message(
            content="Please enter a research question.",
            author="Auto-Analyst",
        ).send()
        return

    # Get settings from session (with type casts for Pylance)
    top_k: int = cl.user_session.get("top_k") or TOP_K_RESULTS
    llm_model: str = cl.user_session.get("llm_model") or DEFAULT_LLM_MODEL
    embed_model: str = cl.user_session.get("embed_model") or DEFAULT_EMBED_MODEL

    # Get conversation history
    history_dicts: List[Dict[str, Any]] = (
        cl.user_session.get("conversation_history") or []
    )
    history_turns = [ConversationTurn.from_dict(h) for h in history_dicts]
    history_turns = trim_history(history_turns, CONVERSATION_MEMORY_TURNS)

    # Initialize LLM and vector store
    llm = load_llm(model_name=llm_model)
    from uuid import uuid4
    from time import perf_counter

    run_id = str(uuid4())
    start_time = perf_counter()
    store = build_vector_store(model_name=embed_model, run_id=run_id)

    # Show "Researching..." animation message
    researching_msg = cl.Message(
        content="🔍 **Researching...**\n\n_Analyzing your query and searching for relevant sources..._",
        author="Auto-Analyst",
    )
    await researching_msg.send()

    # Create the response message for streaming (will replace researching message)
    response_msg = cl.Message(content="", author="Auto-Analyst")

    # Track streaming state
    generate_buffer = ""
    verify_buffer = ""
    final_result = None
    response_started = False  # Track if we've started the response

    # Step tracking
    steps = {}

    async def update_step(
        node: str, status: str, label: str, data: Dict[str, Any] | None = None
    ) -> None:
        """Update or create a step in the UI."""
        if node not in steps:
            steps[node] = cl.Step(name=label, type="tool")
            await steps[node].__aenter__()

        if status == "complete":
            step = steps[node]
            if data:
                output_parts = []
                if "task_count" in data:
                    output_parts.append(f"Created {data['task_count']} search tasks")
                if "result_count" in data:
                    output_parts.append(f"Found {data['result_count']} results")
                if "document_count" in data:
                    output_parts.append(
                        f"Fetched {data['document_count']} documents, "
                        f"created {data.get('chunk_count', 0)} chunks"
                    )
                if "retrieved_count" in data:
                    output_parts.append(
                        f"Retrieved {data['retrieved_count']} relevant chunks"
                    )
                if "answer_length" in data:
                    output_parts.append(f"Generated {data['answer_length']} chars")
                if "verified_length" in data:
                    output_parts.append(f"Verified {data['verified_length']} chars")
                if output_parts:
                    step.output = " | ".join(output_parts)
            await step.__aexit__(None, None, None)

    # Stream through the research pipeline
    event: Dict[str, Any]
    async for event in async_generator_wrapper(
        run_research_streaming(
            query,
            llm=llm,
            vector_store=store,
            embed_model=embed_model,
            top_k=top_k,
            conversation_history=history_turns,
        )
    ):
        event_type = event.get("type")

        if event_type == "step":
            node = event.get("node", "")
            status = event.get("status", "")
            label = event.get("label", node)
            data = event.get("data", {})

            # Update researching message with current step
            if status == "start" and node in ("search", "fetch", "retrieve"):
                step_labels = {
                    "search": "🔍 **Searching** the web...",
                    "fetch": "📄 **Fetching** documents...",
                    "retrieve": "🧠 **Analyzing** content...",
                }
                researching_msg.content = (
                    f"{step_labels.get(node, '🔍 **Researching...**')}"
                )
                await researching_msg.update()

            if status == "start":
                await update_step(node, status, label)
            elif status == "complete":
                await update_step(node, status, label, data)

        elif event_type == "token":
            text = event.get("text", "")
            phase = event.get("phase", "generate")

            # When we get the first token, remove researching message and start response
            if not response_started:
                response_started = True
                await researching_msg.remove()
                await response_msg.send()

            if phase == "generate":
                generate_buffer = text
                # Update the response message with the streaming text
                await response_msg.stream_token(
                    text[len(response_msg.content) :]
                    if len(text) > len(response_msg.content)
                    else ""
                )
            elif phase == "verify":
                verify_buffer = text
                # Clear and show verification
                if verify_buffer != generate_buffer:
                    # During verification, update the message
                    new_content = verify_buffer
                    if len(new_content) > len(response_msg.content):
                        await response_msg.stream_token(
                            new_content[len(response_msg.content) :]
                        )

        elif event_type == "complete":
            result = event.get("result")
            if result is not None:
                final_result = result
            from_cache = event.get("from_cache", False)
            if from_cache and final_result is not None:
                # For cached results, remove researching and show the answer
                if not response_started:
                    await researching_msg.remove()
                    await response_msg.send()
                answer = final_result.verified_answer or final_result.draft_answer
                response_msg.content = answer
                await response_msg.update()

    # Calculate elapsed time
    elapsed_time = perf_counter() - start_time

    # Finalize the message with the verified answer
    if final_result:
        answer = final_result.verified_answer or final_result.draft_answer

        # Update final message content
        response_msg.content = answer
        await response_msg.update()

        # Build source cards with favicons
        if final_result.citations:
            # Get snippets from search results if available
            snippets: Dict[str, str] = {}
            if hasattr(final_result, "search_results"):
                for sr in final_result.search_results:
                    if hasattr(sr, "url") and hasattr(sr, "snippet"):
                        snippets[sr.url] = sr.snippet

            # Build rich source cards
            source_cards: List[str] = []
            for cite in final_result.citations:
                marker = cite.get("marker", "")
                title = cite.get("title", "Source")
                url = cite.get("url", "")
                snippet = snippets.get(url, "")

                if url:
                    card = build_source_card(marker, title, url, snippet)
                    source_cards.append(card)
                else:
                    source_cards.append(f"**{marker}** {title}")

            if source_cards:
                # Send sources as rich cards
                sources_header = f"### 📚 Sources ({len(source_cards)})\n"
                sources_header += f"_Research completed in {elapsed_time:.1f}s_\n\n"
                sources_content = sources_header + "\n\n---\n\n".join(source_cards)

                await cl.Message(
                    content=sources_content,
                    author="Auto-Analyst",
                ).send()

        # Update conversation history
        new_turn = ConversationTurn(
            query=query,
            answer=answer,
            citations=final_result.citations,
        )
        history_dicts.append(new_turn.to_dict())
        cl.user_session.set("conversation_history", history_dicts)


async def async_generator_wrapper(
    sync_gen: "Generator[Dict[str, Any], None, None]",
) -> "AsyncGenerator[Dict[str, Any], None]":
    """Wrap a sync generator to work with async iteration."""
    import asyncio

    loop = asyncio.get_event_loop()

    def get_next() -> tuple[Dict[str, Any] | None, bool]:
        try:
            return next(sync_gen), False
        except StopIteration:
            return None, True

    while True:
        item, done = await loop.run_in_executor(None, get_next)
        if done:
            break
        if item is not None:
            yield item


# =============================================================================
# RESUME HANDLER
# =============================================================================
@cl.on_chat_resume
async def on_chat_resume(thread):
    """Resume a previous chat session."""
    # Restore conversation history from thread
    history = []
    for message in thread.get("steps", []):
        if message.get("type") == "user_message":
            history.append({"role": "user", "content": message.get("output", "")})
        elif message.get("type") == "assistant_message":
            history.append({"role": "assistant", "content": message.get("output", "")})

    # Convert to ConversationTurn format
    conversation_history = []
    i = 0
    while i < len(history) - 1:
        if (
            history[i].get("role") == "user"
            and history[i + 1].get("role") == "assistant"
        ):
            conversation_history.append(
                {
                    "query": history[i].get("content", ""),
                    "answer": history[i + 1].get("content", ""),
                    "citations": [],
                }
            )
            i += 2
        else:
            i += 1

    cl.user_session.set("conversation_history", conversation_history)
    cl.user_session.set("top_k", TOP_K_RESULTS)
    cl.user_session.set("llm_model", DEFAULT_LLM_MODEL)
    cl.user_session.set("embed_model", DEFAULT_EMBED_MODEL)
