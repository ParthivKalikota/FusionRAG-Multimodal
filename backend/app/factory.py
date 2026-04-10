"""
Provider Factory
================
Each component is independently configured via .env:

  LLM_PROVIDER=openai   → GPT-4o  (set OPENAI_API_KEY)
  LLM_PROVIDER=ollama   → Phi-3 via Ollama (fully offline)

  STT_PROVIDER=local    → openai-whisper small (no API key)
  STT_PROVIDER=openai   → OpenAI Whisper API

  Reranker              → always cross-encoder/ms-marco (local, no key)
  CLIP Embedder         → always HuggingFace CLIP (local, no key)

Default (hybrid) setup:
  LLM_PROVIDER=openai + STT_PROVIDER=local
  → GPT-4o answers + everything else local and free
"""
from functools import lru_cache

from app.config import get_settings
from app.providers.base.llm_provider import LLMProvider
from app.providers.base.reranker_provider import RerankerProvider
from app.providers.base.stt_provider import STTProvider

settings = get_settings()


@lru_cache(maxsize=1)
def get_llm_provider() -> LLMProvider:
    """
    openai → OpenAI GPT-4o (cloud, requires OPENAI_API_KEY)
    ollama → Microsoft Phi-3 via Ollama (fully local)
    """
    if settings.llm_provider == "openai":
        from app.providers.cloud.openai_llm import OpenAILLM
        return OpenAILLM()
    else:
        from app.providers.local.ollama_llm import OllamaLLM
        return OllamaLLM()


@lru_cache(maxsize=1)
def get_stt_provider() -> STTProvider:
    """
    local  → openai-whisper small (no API key, runs on CPU)
    openai → OpenAI Whisper API (requires OPENAI_API_KEY)
    """
    if settings.stt_provider == "local":
        from app.providers.local.local_whisper import LocalWhisper
        return LocalWhisper()
    else:
        from app.providers.cloud.openai_whisper import OpenAIWhisper
        return OpenAIWhisper()


@lru_cache(maxsize=1)
def get_reranker() -> RerankerProvider:
    """Always local cross-encoder — no API key, works in all modes."""
    from app.providers.local.cross_encoder import CrossEncoderReranker
    return CrossEncoderReranker()


def get_active_providers() -> dict[str, str]:
    """Summary of active providers — used by /health endpoint."""
    llm_name = (
        f"openai/{settings.cloud_llm_model}"
        if settings.llm_provider == "openai"
        else f"ollama/{settings.local_llm_model}"
    )
    stt_name = (
        f"whisper-{settings.local_whisper_model} (local)"
        if settings.stt_provider == "local"
        else "openai-whisper-api"
    )
    return {
        "llm":      llm_name,
        "stt":      stt_name,
        "reranker": f"{settings.reranker_model} (local)",
        "embedder": f"{settings.clip_model} (local)",
    }
