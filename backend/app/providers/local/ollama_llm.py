from typing import AsyncIterator

import httpx

from app.config import get_settings
from app.providers.base.llm_provider import LLMProvider

settings = get_settings()


class OllamaLLM(LLMProvider):
    """Local LLM provider using Microsoft Phi-3 via Ollama."""

    def __init__(self) -> None:
        self._base_url = settings.ollama_base_url.rstrip("/")
        self._model = settings.local_llm_model  # "phi3"

    async def stream(self, messages: list[dict]) -> AsyncIterator[str]:
        """Stream tokens from Ollama chat API."""
        url = f"{self._base_url}/api/chat"
        payload = {
            "model": self._model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": 0.2,
                "num_predict": 2048,
            },
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream("POST", url, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    import json
                    try:
                        data = json.loads(line)
                        token = data.get("message", {}).get("content", "")
                        if token:
                            yield token
                        if data.get("done"):
                            break
                    except json.JSONDecodeError:
                        continue

    async def complete(self, messages: list[dict]) -> str:
        """Non-streaming completion via Ollama."""
        url = f"{self._base_url}/api/chat"
        payload = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_predict": 512,
            },
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            return data.get("message", {}).get("content", "")
