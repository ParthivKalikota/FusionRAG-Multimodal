"""
Document Parser — handles DOCX and plain text (TXT, MD) files.

Libraries:
  - python-docx : DOCX paragraph + table extraction
"""
from pathlib import Path

from app.core.ingestion.chunker import chunk_text
from app.core.ingestion.schemas import ParsedChunk


def _table_to_markdown(table) -> str:
    """Convert a python-docx Table object to markdown."""
    rows = []
    for row in table.rows:
        rows.append([cell.text.strip() for cell in row.cells])

    if not rows:
        return ""

    header = rows[0]
    separator = ["---"] * len(header)
    body = rows[1:]

    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    for row in body:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def parse_docx(file_path: str, filename: str) -> list[ParsedChunk]:
    """Parse a DOCX file into text and table ParsedChunks."""
    from docx import Document

    doc = Document(file_path)
    chunks: list[ParsedChunk] = []
    chunk_index = 0

    # Collect paragraphs into a running text block
    # Reset when we hit a heading to preserve section context
    current_section = None
    current_text_parts: list[str] = []

    def _flush_text(section: str | None) -> None:
        nonlocal chunk_index
        combined = "\n".join(current_text_parts).strip()
        if not combined:
            return
        text_chunks = chunk_text(
            text=combined,
            source_file=filename,
            chunk_type="text",
            section_heading=section,
            start_chunk_index=chunk_index,
        )
        for tc in text_chunks:
            chunks.append(ParsedChunk(
                chunk_type="text",
                content_text=tc["content_text"],
                image=None,
                image_path=None,
                metadata=tc["metadata"],
            ))
            chunk_index += 1
        current_text_parts.clear()

    for block in doc.element.body:
        from docx.oxml.ns import qn

        # Paragraph
        if block.tag == qn("w:p"):
            from docx.text.paragraph import Paragraph
            para = Paragraph(block, doc)
            style = para.style.name if para.style else ""
            text = para.text.strip()

            if not text:
                continue

            # Heading → flush current section and start new
            if style.startswith("Heading"):
                _flush_text(current_section)
                current_section = text
            else:
                current_text_parts.append(text)

        # Table
        elif block.tag == qn("w:tbl"):
            # Flush text before the table
            _flush_text(current_section)

            from docx.table import Table
            table = Table(block, doc)
            md_table = _table_to_markdown(table)
            if not md_table.strip():
                continue

            metadata = {
                "chunk_index": chunk_index,
                "chunk_type": "table",
                "source_file": filename,
                "language": "en",
                "word_count": len(md_table.split()),
                "char_start": None,
                "char_end": None,
                "modal_metadata": {
                    "row_count": len(table.rows),
                    "col_count": len(table.columns),
                    "caption": current_section,
                    "surrounding_text": None,
                },
            }
            chunks.append(ParsedChunk(
                chunk_type="table",
                content_text=md_table,
                image=None,
                image_path=None,
                metadata=metadata,
            ))
            chunk_index += 1

    # Flush any remaining text
    _flush_text(current_section)
    return chunks


def parse_txt(file_path: str, filename: str) -> list[ParsedChunk]:
    """Parse a plain text (TXT, MD) file into text ParsedChunks."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    text_chunks = chunk_text(
        text=text,
        source_file=filename,
        chunk_type="text",
    )

    return [
        ParsedChunk(
            chunk_type="text",
            content_text=tc["content_text"],
            image=None,
            image_path=None,
            metadata=tc["metadata"],
        )
        for tc in text_chunks
    ]
