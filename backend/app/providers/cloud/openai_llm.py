from typing import AsyncIterator

from openai import AsyncOpenAI

from app.config import get_settings
from app.providers.base.llm_provider import LLMProvider

settings = get_settings()


class OpenAILLM(LLMProvider):
    """Cloud LLM provider using OpenAI GPT-4o with streaming."""

    def __init__(self) -> None:
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        self._model = settings.cloud_llm_model

    async def stream(self, messages: list[dict]) -> AsyncIterator[str]:
        """Stream GPT-4o tokens via OpenAI chat completions API."""
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            stream=True,
            temperature=0.2,      # lower temp for factual RAG answers
            max_tokens=2048,
        )
        async for chunk in response:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content

    async def complete(self, messages: list[dict]) -> str:
        """Non-streaming completion — used for structured outputs."""
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            stream=False,
            temperature=0.0,
            max_tokens=512,
        )
        return response.choices[0].message.content or ""
