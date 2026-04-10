"""
Reranker — delegates to CrossEncoderReranker (always local, both modes).
Takes top 30 RRF candidates and returns top N after cross-encoder scoring.
"""
from app.config import get_settings
from app.factory import get_reranker
from app.providers.base.reranker_provider import RankedChunk

settings = get_settings()


async def rerank(
    query: str,
    candidates: list[RankedChunk],
    top_n: int | None = None,
) -> list[RankedChunk]:
    """
    Re-rank hybrid search candidates using the cross-encoder.

    Args:
        query:      The user's original query.
        candidates: Top-K chunks from hybrid search (RRF-fused).
        top_n:      Number of chunks to return (default: settings.rerank_top_n).

    Returns:
        top_n RankedChunk objects sorted by cross-encoder relevance score,
        with updated relevance_score values.
    """
    if top_n is None:
        top_n = settings.rerank_top_n

    if not candidates:
        return []

    reranker = get_reranker()
    return await reranker.rerank(query, candidates, top_n)
