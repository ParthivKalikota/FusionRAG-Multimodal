"""
Upload routes — file ingestion endpoint.

POST /upload           — Upload one or more files, trigger async ingestion
GET  /upload/status/{id} — Poll ingestion status
GET  /documents        — List all indexed documents
DELETE /documents/{id} — Delete a document and all its chunks
"""
import os
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, Form, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db
from app.config import get_settings
from app.core.database.pgvector_ops import (
    delete_document,
    get_document,
    insert_document,
    list_documents,
    update_document_status,
)
from app.core.ingestion.router import get_file_type, ingest_file

router = APIRouter()
settings = get_settings()


# ── Background ingestion task ─────────────────────────────────────────────────

async def _run_ingestion(
    file_path: str,
    filename: str,
    document_id: uuid.UUID,
    context: str | None = None,
) -> None:
    """Background task — runs ingestion pipeline and cleans up the temp file."""
    from app.core.database.connection import get_session

    async with get_session() as session:
        try:
            await ingest_file(session, file_path, filename, document_id, context)
        except Exception as e:
            async with get_session() as err_session:
                await update_document_status(err_session, document_id, "failed")
            print(f"[Ingestion ERROR] {filename}: {e}")
        finally:
            # Always clean up the temp upload file
            if os.path.exists(file_path):
                os.remove(file_path)


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/upload")
async def upload_files(
    files: list[UploadFile],
    background_tasks: BackgroundTasks,
    context: str | None = Form(None),
    session: AsyncSession = Depends(get_db),
):
    """
    Upload one or more files and trigger async ingestion.

    Supported: PDF, DOCX, TXT, MD, MP3, WAV, M4A, OGG, FLAC, WEBM, JPG, PNG, WEBP

    Returns immediately with document IDs — poll /upload/status/{id} for progress.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for file in files:
        filename = file.filename or "unknown"
        file_type = get_file_type(filename)

        if not file_type:
            results.append({
                "filename": filename,
                "status": "rejected",
                "error": f"Unsupported file type. Supported: pdf, docx, txt, mp3, wav, jpg, png",
            })
            continue

        # Save file to upload dir with a unique name to avoid collisions
        unique_name = f"{uuid.uuid4()}_{filename}"
        save_path = str(upload_dir / unique_name)

        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Create document DB record
        doc = await insert_document(
            session=session,
            filename=filename,
            file_type=file_type,
        )

        # Schedule ingestion as background task
        background_tasks.add_task(
            _run_ingestion,
            file_path=save_path,
            filename=filename,
            document_id=doc.id,
            context=context,
        )

        results.append({
            "document_id": str(doc.id),
            "filename": filename,
            "file_type": file_type,
            "status": "processing",
        })

    return {"documents": results}


@router.get("/upload/status/{document_id}")
async def get_upload_status(
    document_id: uuid.UUID,
    session: AsyncSession = Depends(get_db),
):
    """Poll the ingestion status of a document by its ID."""
    doc = await get_document(session, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    return {
        "document_id": str(doc.id),
        "filename": doc.filename,
        "status": doc.status,          # pending | processing | done | failed
        "total_chunks": doc.total_chunks,
        "created_at": doc.created_at.isoformat() if doc.created_at else None,
    }


@router.get("/documents")
async def list_all_documents(session: AsyncSession = Depends(get_db)):
    """List all indexed documents with their status and chunk counts."""
    docs = await list_documents(session)
    return {"documents": [doc.to_dict() for doc in docs]}


@router.delete("/documents/{document_id}")
async def remove_document(
    document_id: uuid.UUID,
    session: AsyncSession = Depends(get_db),
):
    """Delete a document and all its associated chunks and embeddings."""
    deleted = await delete_document(session, document_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found.")
    return {"message": f"Document {document_id} deleted successfully."}
