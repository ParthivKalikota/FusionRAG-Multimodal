import uuid
from dataclasses import dataclass
from typing import Optional

from sqlalchemy import func, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core.database.models import Chunk, Document

settings = get_settings()


@dataclass
class ChunkInsert:
    """Data transfer object for inserting a chunk."""
    document_id: uuid.UUID
    chunk_type: str          # text | table | image | audio
    content_text: Optional[str]
    image_path: Optional[str]
    embedding: list[float]   # 512-dim CLIP vector
    metadata: dict           # unified metadata schema


@dataclass
class SearchResult:
    """A retrieved chunk with its similarity score."""
    chunk: Chunk
    score: float
    rank: int


async def insert_document(
    session: AsyncSession,
    filename: str,
    file_type: str,
    title: Optional[str] = None,
    author: Optional[str] = None,
    language: Optional[str] = None,
    total_pages: Optional[int] = None,
) -> Document:
    """Create a new document record and return it."""
    doc = Document(
        filename=filename,
        file_type=file_type,
        status="processing",
        title=title,
        author=author,
        language=language,
        total_pages=total_pages,
    )
    session.add(doc)
    await session.flush()  # get the generated id without committing
    return doc


async def insert_chunks(
    session: AsyncSession,
    chunks: list[ChunkInsert],
) -> list[Chunk]:
    """Bulk insert chunks with embeddings and auto-generate ts_vector."""
    db_chunks = []
    for c in chunks:
        chunk = Chunk(
            document_id=c.document_id,
            chunk_type=c.chunk_type,
            content_text=c.content_text,
            image_path=c.image_path,
            embedding=c.embedding,
            chunk_metadata=c.metadata,   # 'metadata' is reserved in SQLAlchemy ORM
        )
        session.add(chunk)
        db_chunks.append(chunk)

    await session.flush()  # assign IDs

    # Update ts_vector for text-based chunks using func.to_tsvector
    for chunk in db_chunks:
        if chunk.content_text:
            await session.execute(
                update(Chunk)
                .where(Chunk.id == chunk.id)
                .values(
                    ts_vector=func.to_tsvector('english', chunk.content_text)
                )
            )

    return db_chunks


async def vector_search(
    session: AsyncSession,
    query_embedding: list[float],
    top_k: int,
    document_ids: Optional[list[uuid.UUID]] = None,
) -> list[SearchResult]:
    """Cosine similarity search using pgvector."""
    # Build base query
    stmt = (
        select(
            Chunk,
            (1 - Chunk.embedding.cosine_distance(query_embedding)).label("score"),
        )
        .where(Chunk.embedding.is_not(None))
        .order_by(Chunk.embedding.cosine_distance(query_embedding))
        .limit(top_k)
    )

    # Optional filter by document
    if document_ids:
        stmt = stmt.where(Chunk.document_id.in_(document_ids))

    result = await session.execute(stmt)
    rows = result.all()

    return [
        SearchResult(chunk=row.Chunk, score=float(row.score), rank=i + 1)
        for i, row in enumerate(rows)
    ]


async def fts_search(
    session: AsyncSession,
    query: str,
    top_k: int,
    document_ids: Optional[list[uuid.UUID]] = None,
) -> list[SearchResult]:
    """Full-text search using PostgreSQL tsvector + ts_rank.

    Uses plainto_tsquery which safely handles arbitrary user input
    (punctuation, stop words, multi-word phrases) without crashing.
    """
    # plainto_tsquery converts free text safely — no special chars needed
    stmt = text(
        """
        SELECT c.*, ts_rank(c.ts_vector, plainto_tsquery('english', :query)) AS score
        FROM chunks c
        WHERE c.ts_vector @@ plainto_tsquery('english', :query)
        {doc_filter}
        ORDER BY score DESC
        LIMIT :top_k
        """.format(
            doc_filter="AND c.document_id = ANY(:doc_ids::uuid[])" if document_ids else ""
        )
    )

    params: dict = {"query": query, "top_k": top_k}
    if document_ids:
        params["doc_ids"] = [str(d) for d in document_ids]

    rows = await session.execute(stmt, params)
    results = rows.fetchall()

    search_results = []
    for i, row in enumerate(results):
        # Re-fetch the ORM object so we get the full Chunk model
        chunk = await session.get(Chunk, row.id)
        if chunk:
            search_results.append(SearchResult(chunk=chunk, score=float(row.score), rank=i + 1))

    return search_results


async def update_document_status(
    session: AsyncSession,
    document_id: uuid.UUID,
    status: str,
    total_chunks: Optional[int] = None,
) -> None:
    """Update document processing status."""
    values: dict = {"status": status}
    if total_chunks is not None:
        values["total_chunks"] = total_chunks

    await session.execute(
        update(Document).where(Document.id == document_id).values(**values)
    )


async def get_document(
    session: AsyncSession,
    document_id: uuid.UUID,
) -> Optional[Document]:
    return await session.get(Document, document_id)


async def list_documents(session: AsyncSession) -> list[Document]:
    result = await session.execute(select(Document).order_by(Document.created_at.desc()))
    return list(result.scalars().all())


async def delete_document(session: AsyncSession, document_id: uuid.UUID) -> bool:
    doc = await session.get(Document, document_id)
    if not doc:
        return False
    await session.delete(doc)
    return True
