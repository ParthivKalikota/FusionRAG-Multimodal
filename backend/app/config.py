from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from typing import Literal

# Resolve .env from the project root (two levels up from this file)
# backend/app/config.py → backend/app/ → backend/ → MultiModal RAG/
_ENV_FILE = Path(__file__).resolve().parent.parent.parent / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM ──────────────────────────────────────────────────────────────
    # "openai" → GPT-4o via API | "ollama" → Phi-3 local
    llm_provider: Literal["openai", "ollama"] = "openai"
    openai_api_key: str = ""
    cloud_llm_model: str = "gpt-4o"
    local_llm_model: str = "phi3"
    ollama_base_url: str = "http://localhost:11434"

    # ── STT ───────────────────────────────────────────────────────────────
    # "local" → openai-whisper lib | "openai" → Whisper API
    stt_provider: Literal["local", "openai"] = "local"
    local_whisper_model: str = "small"      # tiny | base | small | medium

    # ── Reranker (always local cross-encoder, no API key needed) ──────────
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ── Retrieval ─────────────────────────────────────────────────────────
    retrieval_top_k: int = 30
    rerank_top_n: int = 5

    # ── Database ──────────────────────────────────────────────────────────
    database_url: str = "postgresql+asyncpg://postgres:password@localhost:5432/ragdb"

    # ── CLIP (always local) ───────────────────────────────────────────────
    clip_model: str = "openai/clip-vit-base-patch32"

    # ── Storage ───────────────────────────────────────────────────────────
    storage_dir: str = "storage/images"
    upload_dir: str = "uploads"

    # ── App ───────────────────────────────────────────────────────────────
    app_title: str = "Multimodal RAG API"
    app_version: str = "1.0.0"
    cors_origins: list[str] = ["http://localhost:3000"]


@lru_cache
def get_settings() -> Settings:
    return Settings()
