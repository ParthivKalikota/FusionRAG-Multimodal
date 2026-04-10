import asyncio
from functools import lru_cache

from app.config import get_settings
from app.providers.base.reranker_provider import RankedChunk, RerankerProvider

settings = get_settings()


@lru_cache(maxsize=1)
def _load_cross_encoder():
    """Load and cache the cross-encoder model (only done once)."""
    from sentence_transformers import CrossEncoder
    print(f"[CrossEncoder] Loading model: {settings.reranker_model}")
    return CrossEncoder(settings.reranker_model)


class CrossEncoderReranker(RerankerProvider):
    """Reranker using ms-marco cross-encoder — always local, no API key.

    Used in BOTH cloud and local RAG_MODE.
    Model is loaded once and cached for the lifetime of the process.
    """

    async def rerank(
        self,
        query: str,
        candidates: list[RankedChunk],
        top_n: int,
    ) -> list[RankedChunk]:
        """Re-rank candidates using cross-encoder relevance scores.

        Runs in a thread pool to avoid blocking the event loop.
        """
        if not candidates:
            return []

        top_n = min(top_n, len(candidates))

        # Build (query, passage) pairs for the cross-encoder
        pairs = [(query, c.content_text or "") for c in candidates]

        loop = asyncio.get_running_loop()
        scores: list[float] = await loop.run_in_executor(
            None,
            self._score_sync,
            pairs,
        )

        # Attach scores and sort descending
        for chunk, score in zip(candidates, scores):
            chunk.relevance_score = float(score)

        ranked = sorted(candidates, key=lambda c: c.relevance_score, reverse=True)
        return ranked[:top_n]

    @staticmethod
    def _score_sync(pairs: list[tuple[str, str]]) -> list[float]:
        model = _load_cross_encoder()
        scores = model.predict(pairs, show_progress_bar=False)
        return scores.tolist()
