import asyncio
from pathlib import Path

from openai import AsyncOpenAI

from app.config import get_settings
from app.providers.base.stt_provider import STTProvider

settings = get_settings()


class OpenAIWhisper(STTProvider):
    """Cloud STT provider using OpenAI Whisper API."""

    def __init__(self) -> None:
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)

    async def transcribe(self, audio_path: str) -> str:
        """Transcribe audio via OpenAI Whisper API.

        Supports: mp3, mp4, mpeg, mpga, m4a, wav, webm (max 25MB).
        """
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        with open(path, "rb") as audio_file:
            transcript = await self._client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text",
            )

        return transcript if isinstance(transcript, str) else transcript.text
