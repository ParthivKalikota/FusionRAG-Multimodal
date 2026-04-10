from abc import ABC, abstractmethod
from typing import AsyncIterator


class LLMProvider(ABC):
    """Abstract base class for LLM providers.

    Implementations: OpenAILLM (cloud), OllamaLLM (local)
    """

    @abstractmethod
    async def stream(
        self,
        messages: list[dict],
    ) -> AsyncIterator[str]:
        """Stream response tokens for a list of chat messages.

        Args:
            messages: OpenAI-format message list, e.g.
                      [{"role": "system", ...}, {"role": "user", ...}]

        Yields:
            Individual text tokens as they are generated.
        """
        raise NotImplementedError

    @abstractmethod
    async def complete(
        self,
        messages: list[dict],
    ) -> str:
        """Non-streaming completion — returns the full response string.
        Used for structured outputs (e.g. citation extraction fallback).
        """
        raise NotImplementedError
