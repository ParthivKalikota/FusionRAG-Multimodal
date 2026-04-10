import uuid
from datetime import datetime
from typing import Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    filename: Mapped[str] = mapped_column(Text, nullable=False)
    file_type: Mapped[str] = mapped_column(Text, nullable=False)  # pdf|docx|txt|audio|image
    status: Mapped[str] = mapped_column(Text, default="pending")  # pending|processing|done|failed
    title: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    author: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    language: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    total_pages: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    total_chunks: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationship
    chunks: Mapped[list["Chunk"]] = relationship(
        "Chunk", back_populates="document", cascade="all, delete-orphan"
    )

    def to_dict(self) -> dict:
        return {
            "id": str(self.id),
            "filename": self.filename,
            "file_type": self.file_type,
            "status": self.status,
            "title": self.title,
            "author": self.author,
            "language": self.language,
            "total_pages": self.total_pages,
            "total_chunks": self.total_chunks,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    chunk_type: Mapped[str] = mapped_column(Text, nullable=False)  # text|table|image|audio
    content_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    image_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    embedding: Mapped[Optional[list[float]]] = mapped_column(Vector(512), nullable=True)
    # Unified metadata schema:
    # {
    #   chunk_index, chunk_type, source_file, language,
    #   word_count, char_start, char_end,
    #   modal_metadata: { ...modality-specific fields }
    # }
    chunk_metadata: Mapped[Optional[dict]] = mapped_column("metadata", JSONB, nullable=True)
    ts_vector: Mapped[Optional[str]] = mapped_column(TSVECTOR, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationship
    document: Mapped["Document"] = relationship("Document", back_populates="chunks")

    def to_dict(self) -> dict:
        return {
            "id": str(self.id),
            "document_id": str(self.document_id),
            "chunk_type": self.chunk_type,
            "content_text": self.content_text,
            "image_path": self.image_path,
            "metadata": self.chunk_metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    def to_citation(self, rank: int, relevance_score: float) -> dict:
        """Produce a citation object for the API response."""
        meta = self.chunk_metadata or {}
        snippet = (self.content_text or "")[:300]
        return {
            "rank": rank,
            "relevance_score": round(relevance_score, 4),
            "chunk_type": self.chunk_type,
            "source_file": meta.get("source_file", ""),
            "snippet": snippet,
            "modal_metadata": meta.get("modal_metadata", {}),
        }


# ── Indexes ────────────────────────────────────────────────────────────────────
# IVFFlat index for vector similarity search
Index(
    "ix_chunks_embedding_ivfflat",
    Chunk.embedding,
    postgresql_using="ivfflat",
    postgresql_ops={"embedding": "vector_cosine_ops"},
    postgresql_with={"lists": 100},
)

# GIN index for full-text search
Index(
    "ix_chunks_ts_vector_gin",
    Chunk.ts_vector,
    postgresql_using="gin",
)

# GIN index for JSONB metadata queries (modal_metadata fields)
Index(
    "ix_chunks_metadata_gin",
    Chunk.chunk_metadata,
    postgresql_using="gin",
)
