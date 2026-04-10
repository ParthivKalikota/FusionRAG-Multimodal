"""
Query route — SSE streaming endpoint for RAG queries.

POST /query
  Body: { "query": "...", "document_ids": [...] (optional) }

SSE event stream format:
  data: {"type": "token",   "content": "Revenue"}
  data: {"type": "token",   "content": " grew..."}
  ...
  data: {"type": "done",    "citations": [...]}
  data: {"type": "error",   "message": "..."}   ← on failure
"""
import json
import uuid
from typing import AsyncGenerator, Optional

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db
from app.core.generation.generator import extract_cited_chunks, stream_answer, stream_chat
from app.core.retrieval.hybrid_search import hybrid_search
from app.core.retrieval.reranker import rerank

router = APIRouter()


# ── Request / response schemas ────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    document_ids: Optional[list[uuid.UUID]] = None   # filter to specific docs


# ── SSE helpers ───────────────────────────────────────────────────────────────

def _sse(data: dict) -> str:
    """Format a dict as a single SSE data line."""
    return f"data: {json.dumps(data)}\n\n"


async def _query_stream(
    query: str,
    document_ids: Optional[list[uuid.UUID]],
    session: AsyncSession,
) -> AsyncGenerator[str, None]:
    """
    Full RAG pipeline as an SSE generator:
      hybrid search → rerank → LLM stream → citations
    """
    try:
        # ── Step 1: Hybrid search → top 30 ──────────────────────────────
        candidates = await hybrid_search(
            session=session,
            query=query,
            document_ids=document_ids,
        )

        # ── Conversational fallback ──────────────────────────────────────
        # No relevant documents found — answer as a general assistant
        if not candidates:
            async for token in stream_chat(query):
                yield _sse({"type": "token", "content": token})
            yield _sse({"type": "done", "citations": []})
            return

        # ── Step 2: Rerank → top 5 ───────────────────────────────────────
        top_chunks = await rerank(query=query, candidates=candidates)

        if not top_chunks:
            yield _sse({"type": "error", "message": "Reranking returned no results."})
            return

        # ── Step 3: Stream LLM answer token by token ─────────────────────
        full_answer_parts: list[str] = []

        async for token in stream_answer(query=query, chunks=top_chunks):
            full_answer_parts.append(token)
            yield _sse({"type": "token", "content": token})

        # ── Step 4: Extract cited chunks from complete answer ────────────
        full_answer = "".join(full_answer_parts)
        citations = extract_cited_chunks(full_answer, top_chunks)

        # ── Step 5: Send final done event with citations ─────────────────
        yield _sse({"type": "done", "citations": citations})

    except Exception as e:
        yield _sse({"type": "error", "message": str(e)})


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.post("/query")
async def query(
    request: Request,
    body: QueryRequest,
    session: AsyncSession = Depends(get_db),
):
    """
    Submit a query and receive a streaming SSE response.

    The stream consists of:
    - Multiple `token` events (the answer, token by token)
    - A final `done` event with `citations` (only actually-cited chunks)
    - A `error` event if something goes wrong

    Inline [N] citation markers in the token stream link to the citations list.
    """
    if not body.query.strip():
        return StreamingResponse(
            iter([_sse({"type": "error", "message": "Query cannot be empty."})]),
            media_type="text/event-stream",
        )

    return StreamingResponse(
        _query_stream(
            query=body.query.strip(),
            document_ids=body.document_ids,
            session=session,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",   # disable Nginx buffering if proxied
        },
    )
