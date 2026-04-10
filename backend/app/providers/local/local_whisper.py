import asyncio
from functools import lru_cache
from pathlib import Path

from app.config import get_settings
from app.providers.base.stt_provider import STTProvider

settings = get_settings()


@lru_cache(maxsize=1)
def _load_whisper_model():
    """Load and cache the Whisper model (expensive — only done once)."""
    import whisper
    print(f"[LocalWhisper] Loading model: {settings.local_whisper_model}")
    return whisper.load_model(settings.local_whisper_model)


class LocalWhisper(STTProvider):
    """Local STT provider using openai-whisper library.

    Model: 'small' (~240MB) — accuracy/speed sweet spot.
    Model is loaded once and cached for the lifetime of the process.
    """

    async def transcribe(self, audio_path: str) -> str:
        """Transcribe audio file using local Whisper model.

        Runs in a thread pool to avoid blocking the event loop.
        """
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Whisper is synchronous — run in executor to keep async safe
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self._transcribe_sync,
            str(path),
        )
        return result

    @staticmethod
    def _transcribe_sync(audio_path: str) -> str:
        model = _load_whisper_model()
        result = model.transcribe(
            audio_path,
            fp16=False,          # CPU-safe (no GPU required)
            verbose=False,
        )
        return result["text"].strip()
