"""
FastAPI application entry point.

Startup:
  - Initialises PostgreSQL + pgvector extension + all ORM tables
  - Creates upload/storage directories

Shutdown:
  - Disposes DB connection pool

Routes:
  /upload    — file ingestion
  /query     — SSE streaming RAG query
  /health    — provider + DB status
"""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import query as query_router
from app.api.routes import upload as upload_router
from app.config import get_settings
from app.core.database.connection import close_db, init_db
from app.factory import get_active_providers

settings = get_settings()


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──────────────────────────────────────────────────────────
    # Ensure storage directories exist
    Path(settings.storage_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)

    # Initialise database (extension + tables + indexes)
    await init_db()
    providers = get_active_providers()
    print(f"[Startup] DB initialised.")
    print(f"[Startup] LLM: {providers['llm']}  |  STT: {providers['stt']}")
    print(f"[Startup] Reranker: {providers['reranker']}  |  Embedder: {providers['embedder']}")

    yield  # ← app is running

    # ── Shutdown ─────────────────────────────────────────────────────────
    await close_db()
    print("[Shutdown] DB connection pool closed.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.app_title,
    version=settings.app_version,
    description="Multimodal RAG — PDF, DOCX, Audio, Image ingestion with CLIP + PGVector",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────

app.include_router(upload_router.router, tags=["Ingestion"])
app.include_router(query_router.router, tags=["Query"])

# ── Static file serving (saved images) ───────────────────────────────────────
# Images are saved to storage_dir (e.g. storage/images) during ingestion.
# Use an absolute path anchored at this file so it works regardless of cwd.
# backend/app/main.py → backend/app/ → backend/ → backend/storage/
_BACKEND_ROOT = Path(__file__).resolve().parent.parent   # → backend/
_STORAGE_ROOT = _BACKEND_ROOT / Path(settings.storage_dir).parent  # → backend/storage/
_STORAGE_ROOT.mkdir(parents=True, exist_ok=True)

app.mount(
    "/storage",
    StaticFiles(directory=str(_STORAGE_ROOT)),
    name="storage",
)


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health():
    """Returns active provider configuration and service status."""
    return {
        "status": "ok",
        "providers": get_active_providers(),
    }
