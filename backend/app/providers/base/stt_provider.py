from abc import ABC, abstractmethod


class STTProvider(ABC):
    """Abstract base class for Speech-to-Text providers.

    Implementations: OpenAIWhisper (cloud), LocalWhisper (local)
    """

    @abstractmethod
    async def transcribe(self, audio_path: str) -> str:
        """Transcribe an audio file to text.

        Args:
            audio_path: Absolute path to the audio file
                        (MP3, WAV, M4A, OGG, FLAC, WebM).

        Returns:
            Full transcription as a plain string.
        """
        raise NotImplementedError
