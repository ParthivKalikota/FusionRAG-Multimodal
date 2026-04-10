"""
Ingestion Router — orchestrates the full ingestion pipeline:
  1. Detect file type
  2. Dispatch to correct parser → ParsedChunk list
  3. Embed each chunk with CLIP
  4. Insert into PGVector database
  5. Update document status
"""
import uuid
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database.pgvector_ops import (
    ChunkInsert,
    insert_chunks,
    update_document_status,
)
from app.core.embedding.clip_embedder import embedder
from app.core.ingestion.schemas import ParsedChunk

# Supported file types
SUPPORTED_TYPES = {
    # PDFs
    ".pdf": "pdf",
    # Documents
    ".docx": "docx",
    ".doc": "docx",
    ".txt": "txt",
    ".md": "txt",
    # Audio
    ".mp3": "audio",
    ".wav": "audio",
    ".m4a": "audio",
    ".ogg": "audio",
    ".flac": "audio",
    ".webm": "audio",
    # Images
    ".jpg": "image",
    ".jpeg": "image",
    ".png": "image",
    ".webp": "image",
    ".bmp": "image",
    ".tiff": "image",
}


def get_file_type(filename: str) -> str | None:
    """Return the canonical file type for a filename, or None if unsupported."""
    ext = Path(filename).suffix.lower()
    return SUPPORTED_TYPES.get(ext)


async def _parse_file(file_path: str, filename: str, file_type: str) -> list[ParsedChunk]:
    """Dispatch to the correct parser based on file type."""
    if file_type == "pdf":
        from app.core.ingestion.pdf_parser import parse_pdf
        return parse_pdf(file_path, filename)

    elif file_type == "docx":
        from app.core.ingestion.doc_parser import parse_docx
        return parse_docx(file_path, filename)

    elif file_type == "txt":
        from app.core.ingestion.doc_parser import parse_txt
        return parse_txt(file_path, filename)

    elif file_type == "audio":
        from app.core.ingestion.audio_parser import parse_audio
        return await parse_audio(file_path, filename)

    elif file_type == "image":
        from app.core.ingestion.image_parser import parse_image
        return parse_image(file_path, filename)

    else:
        raise ValueError(f"Unsupported file type: {file_type}")



async def ingest_file(
    session: AsyncSession,
    file_path: str,
    filename: str,
    document_id: uuid.UUID,
    context: str | None = None,
) -> int:
    """
    Full ingestion pipeline for a single file.

    Args:
        session:     Database session.
        file_path:   Absolute path to the uploaded file.
        filename:    Original filename (for metadata + routing).
        document_id: Pre-created document UUID in DB.
        context:     Optional user-provided context for the uploaded file.

    Returns:
        Number of chunks successfully inserted.

    Raises:
        ValueError: If the file type is not supported.
    """
    file_type = get_file_type(filename)
    if not file_type:
        raise ValueError(f"Unsupported file type for: {filename}")

    # ── Step 1: Parse ────────────────────────────────────────────────────
    parsed_chunks = await _parse_file(file_path, filename, file_type)

    if not parsed_chunks:
        await update_document_status(session, document_id, "done", total_chunks=0)
        return 0

    # ── Step 1.5: Inject Context ─────────────────────────────────────────
    if context:
        for chunk in parsed_chunks:
            if chunk.content_text:
                chunk.content_text = f"User Context: {context.strip()}\n\n{chunk.content_text}"
            else:
                chunk.content_text = context.strip()

    # ── Step 2: Embed (batch text, individual images) ───────────────────
    # Separate text-based and image chunks for efficiency
    text_chunks = [c for c in parsed_chunks if c.chunk_type != "image" or c.image is None]
    image_chunks = [c for c in parsed_chunks if c.chunk_type == "image" and c.image is not None]

    # Batch embed all text-based chunks
    text_contents = [c.content_text or "" for c in text_chunks]
    text_embeddings: list[list[float]] = []
    if text_contents:
        text_embeddings = await embedder.embed_texts_batch(text_contents)

    # Embed images individually
    image_embeddings: list[list[float]] = []
    for img_chunk in image_chunks:
        emb = await embedder.embed_image(img_chunk.image)
        image_embeddings.append(emb)

    # Rebuild ordered list matching parsed_chunks order
    text_iter = iter(text_embeddings)
    image_iter = iter(image_embeddings)
    all_embeddings: list[list[float]] = []
    for chunk in parsed_chunks:
        if chunk.chunk_type == "image" and chunk.image is not None:
            all_embeddings.append(next(image_iter))
        else:
            all_embeddings.append(next(text_iter))

    # ── Step 3: Build DB insert objects ─────────────────────────────────
    db_chunks = [
        ChunkInsert(
            document_id=document_id,
            chunk_type=chunk.chunk_type,
            content_text=chunk.content_text,
            image_path=chunk.image_path,
            embedding=embedding,
            metadata=chunk.metadata,
        )
        for chunk, embedding in zip(parsed_chunks, all_embeddings)
    ]

    # ── Step 4: Insert into DB ───────────────────────────────────────────
    await insert_chunks(session, db_chunks)

    # ── Step 5: Update document status ──────────────────────────────────
    await update_document_status(
        session,
        document_id,
        "done",
        total_chunks=len(db_chunks),
    )

    return len(db_chunks)
