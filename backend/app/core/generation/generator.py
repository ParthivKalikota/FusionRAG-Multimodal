"""
Generator — builds citation-aware prompts and streams LLM responses.

Pipeline:
  top 5 RankedChunks → numbered prompt [1]-[5] → LLM stream
  → inline [N] markers extracted → citation objects returned
"""
import re
from typing import AsyncIterator

from app.factory import get_llm_provider
from app.providers.base.reranker_provider import RankedChunk

# ── System prompts ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are a helpful, precise assistant. Answer the user's question using ONLY \
the context provided below.

Rules:
1. Cite sources inline using [1], [2], etc. exactly as you use each piece of information.
2. Only cite a source if you genuinely used it — do not cite unused sources.
3. If the context does not contain sufficient information to answer, say so clearly.
4. Be concise and factual. Avoid repeating information unnecessarily.
"""

# Used when no documents are uploaded / no relevant chunks found
CHAT_SYSTEM_PROMPT = """\
You are a helpful, knowledgeable assistant for a Multimodal RAG application. \
You can answer general questions, explain concepts, and help users understand \
how to use the system. Be friendly, concise, and accurate.
"""


# ── Location formatter ─────────────────────────────────────────────────────────
def _format_location(chunk_type: str, modal: dict) -> str:
    """Build a human-readable location string from modal_metadata."""
    if chunk_type == "text":
        page = modal.get("page_number")
        heading = modal.get("section_heading")
        parts = []
        if page:
            parts.append(f"Page {page}")
        if heading:
            parts.append(heading)
        return " · ".join(parts) if parts else "text"

    elif chunk_type == "table":
        page = modal.get("page_number")
        idx = modal.get("table_index")
        caption = modal.get("caption")
        parts = []
        if idx:
            parts.append(f"Table {idx}")
        if page:
            parts.append(f"Page {page}")
        if caption:
            parts.append(caption)
        return " · ".join(parts) if parts else "table"

    elif chunk_type == "image":
        page = modal.get("page_number")
        caption = modal.get("caption")
        parts = []
        if caption:
            parts.append(caption)
        elif page:
            parts.append(f"Page {page} image")
        return " · ".join(parts) if parts else "image"

    elif chunk_type == "audio":
        ts = modal.get("timestamp_display")
        return f"Audio {ts}" if ts else "audio"

    return chunk_type


def build_messages(query: str, chunks: list[RankedChunk]) -> list[dict]:
    """
    Build the OpenAI-format messages list with numbered context chunks.

    Each chunk is numbered [1]–[N] so the LLM can cite inline.
    Format:
        [1] {content}
            (Source: filename · location)
    """
    context_parts = []
    for i, chunk in enumerate(chunks, start=1):
        modal = chunk.metadata.get("modal_metadata", {})
        source = chunk.metadata.get("source_file", "unknown")
        location = _format_location(chunk.chunk_type, modal)

        content = (chunk.content_text or "").strip()
        # Trim very long chunks to avoid blowing the context window
        if len(content) > 1500:
            content = content[:1500] + "…"

        context_parts.append(
            f"[{i}] {content}\n"
            f"    (Source: {source} · {location})"
        )

    context = "\n\n".join(context_parts)

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Context:\n{context}\n\n"
                f"Question: {query}"
            ),
        },
    ]


def extract_cited_chunks(
    answer: str,
    chunks: list[RankedChunk],
) -> list[dict]:
    """
    Parse inline [N] citation markers from the LLM answer.
    Returns only the chunks that were actually cited.
    Falls back to all chunks if the LLM cited nothing.
    """
    cited_indices = set(int(n) for n in re.findall(r"\[(\d+)\]", answer))

    citations = [
        {
            "rank": i,
            "relevance_score": round(chunks[i - 1].relevance_score, 4),
            "chunk_type": chunks[i - 1].chunk_type,
            "source_file": chunks[i - 1].metadata.get("source_file", ""),
            "snippet": (chunks[i - 1].content_text or "")[:300],
            "modal_metadata": chunks[i - 1].metadata.get("modal_metadata", {}),
            "image_path": chunks[i - 1].image_path,
        }
        for i in sorted(cited_indices)
        if 1 <= i <= len(chunks)
    ]

    return citations


async def stream_answer(
    query: str,
    chunks: list[RankedChunk],
) -> AsyncIterator[str]:
    """
    Stream the LLM answer token by token using RAG context.
    Yields individual text tokens from the LLM.
    """
    llm = get_llm_provider()
    messages = build_messages(query, chunks)
    async for token in llm.stream(messages):
        yield token


async def stream_chat(query: str) -> AsyncIterator[str]:
    """
    Stream a direct conversational LLM response — no RAG context.
    Used for greetings, general questions, and when no documents are uploaded.
    Yields individual text tokens from the LLM.
    """
    llm = get_llm_provider()
    messages = [
        {"role": "system", "content": CHAT_SYSTEM_PROMPT},
        {"role": "user",   "content": query},
    ]
    async for token in llm.stream(messages):
        yield token
