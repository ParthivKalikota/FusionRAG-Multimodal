"""
Hybrid Search — combines vector similarity + full-text search via RRF fusion.

Pipeline:
  query → CLIP embed → vector search (top K)
                    → FTS search    (top K)
                    → RRF fusion    → top 30 RankedChunks
"""
import uuid
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core.database.pgvector_ops import (
    SearchResult,
    fts_search,
    vector_search,
)
from app.core.embedding.clip_embedder import embedder
from app.providers.base.reranker_provider import RankedChunk

settings = get_settings()

# RRF constant — higher k reduces the impact of high-ranked results
RRF_K = 60


def _to_ranked_chunk(chunk, rrf_score: float, rank: int) -> RankedChunk:
    """Convert a DB Chunk ORM object into a RankedChunk."""
    return RankedChunk(
        chunk_id=str(chunk.id),
        content_text=chunk.content_text or "",
        chunk_type=chunk.chunk_type,
        metadata=chunk.chunk_metadata or {},   # renamed from 'metadata' (SQLAlchemy reserved)
        image_path=chunk.image_path,
        relevance_score=rrf_score,
        original_rank=rank,
    )


def _rrf_fusion(
    vector_results: list[SearchResult],
    fts_results: list[SearchResult],
    top_k: int,
) -> list[RankedChunk]:
    """
    Reciprocal Rank Fusion of two result lists.

    score(d) = Σ  1 / (k + rank_i)   for each list i that contains d

    Deduplicates by chunk ID and returns top_k results sorted by score.
    """
    rrf_scores: dict[str, float] = {}
    chunk_map: dict[str, object] = {}  # chunk_id → Chunk ORM object

    # Accumulate vector search ranks
    for result in vector_results:
        cid = str(result.chunk.id)
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (RRF_K + result.rank)
        chunk_map[cid] = result.chunk

    # Accumulate FTS ranks
    for result in fts_results:
        cid = str(result.chunk.id)
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (RRF_K + result.rank)
        chunk_map[cid] = result.chunk

    # Sort by combined RRF score descending
    sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)

    return [
        _to_ranked_chunk(
            chunk=chunk_map[cid],
            rrf_score=rrf_scores[cid],
            rank=rank,
        )
        for rank, cid in enumerate(sorted_ids[:top_k], start=1)
    ]


async def hybrid_search(
    session: AsyncSession,
    query: str,
    top_k: int | None = None,
    document_ids: Optional[list[uuid.UUID]] = None,
) -> list[RankedChunk]:
    """
    Run hybrid search (vector + FTS) and return RRF-fused top_k candidates.

    Args:
        session:      Async DB session.
        query:        User query string.
        top_k:        Number of candidates to return (default: settings.retrieval_top_k).
        document_ids: Optional filter — only search within these documents.

    Returns:
        List of RankedChunk sorted by RRF score descending, max top_k items.
    """
    if top_k is None:
        top_k = settings.retrieval_top_k

    # Embed the query with CLIP text encoder
    query_embedding = await embedder.embed_text(query)

    # Run sequentially — asyncio.gather cannot share a single AsyncSession
    vector_results = await vector_search(session, query_embedding, top_k, document_ids)
    fts_results = await fts_search(session, query, top_k, document_ids)

    # RRF fusion → deduplicated, score-sorted candidates
    candidates = _rrf_fusion(vector_results, fts_results, top_k)

    return candidates
