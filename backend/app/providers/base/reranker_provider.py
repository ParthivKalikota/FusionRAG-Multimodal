from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class RankedChunk:
    """A chunk with its reranker relevance score."""
    chunk_id: str
    content_text: str
    chunk_type: str
    metadata: dict
    image_path: str | None
    relevance_score: float
    original_rank: int  # rank before reranking (from hybrid search)


class RerankerProvider(ABC):
    """Abstract base class for reranker providers.

    Implementation: CrossEncoderReranker (always local, used in both modes)
    """

    @abstractmethod
    async def rerank(
        self,
        query: str,
        candidates: list[RankedChunk],
        top_n: int,
    ) -> list[RankedChunk]:
        """Re-rank candidate chunks by relevance to the query.

        Args:
            query:      The user's original question.
            candidates: Up to top_k chunks from hybrid search.
            top_n:      Number of chunks to return after reranking.

        Returns:
            List of RankedChunk sorted by relevance_score descending,
            limited to top_n items.
        """
        raise NotImplementedError
