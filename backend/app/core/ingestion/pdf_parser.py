"""
PDF Parser — extracts text blocks, tables, and images from PDF files.

Libraries:
  - pdfplumber : text extraction + table detection
  - pymupdf (fitz): image extraction from PDF pages
"""
import io
import os
from pathlib import Path
from typing import Optional

from PIL import Image

from app.config import get_settings
from app.core.ingestion.chunker import chunk_text
from app.core.ingestion.schemas import ParsedChunk

settings = get_settings()


def _table_to_markdown(table: list[list]) -> str:
    """Convert a pdfplumber table (list of rows) to a markdown string."""
    if not table or not table[0]:
        return ""

    # Replace None cells with empty string
    rows = [[str(cell) if cell is not None else "" for cell in row] for row in table]
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


def _detect_section_heading(text: str) -> Optional[str]:
    """Heuristic: first non-empty line if it's short (likely a heading)."""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if lines and len(lines[0]) < 100:
        return lines[0]
    return None


def parse_pdf(file_path: str, filename: str) -> list[ParsedChunk]:
    """Parse a PDF into text, table, and image ParsedChunks.

    Args:
        file_path: Absolute path to the PDF file.
        filename:  Original filename for metadata.

    Returns:
        List of ParsedChunk objects ready for embedding.
    """
    import pdfplumber
    import fitz  # pymupdf

    chunks: list[ParsedChunk] = []
    chunk_index = 0
    image_dir = Path(settings.storage_dir)
    image_dir.mkdir(parents=True, exist_ok=True)

    doc_stem = Path(filename).stem  # filename without extension

    with pdfplumber.open(file_path) as pdf:
        fitz_doc = fitz.open(file_path)

        for page_num, page in enumerate(pdf.pages, start=1):
            # ── Text extraction ──────────────────────────────────────────
            page_text = page.extract_text() or ""
            if page_text.strip():
                section = _detect_section_heading(page_text)
                text_chunks = chunk_text(
                    text=page_text,
                    source_file=filename,
                    chunk_type="text",
                    language="en",
                    page_number=page_num,
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

            # ── Table extraction ─────────────────────────────────────────
            tables = page.extract_tables() or []
            for table_idx, table in enumerate(tables):
                md_table = _table_to_markdown(table)
                if not md_table.strip():
                    continue

                # Grab text around the table for context
                surrounding = (page_text[:300] if page_text else "").replace("\n", " ")

                metadata = {
                    "chunk_index": chunk_index,
                    "chunk_type": "table",
                    "source_file": filename,
                    "language": "en",
                    "word_count": len(md_table.split()),
                    "char_start": None,
                    "char_end": None,
                    "modal_metadata": {
                        "page_number": page_num,
                        "table_index": table_idx + 1,
                        "row_count": len(table),
                        "col_count": len(table[0]) if table else 0,
                        "caption": None,
                        "surrounding_text": surrounding,
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

            # ── Image extraction (via pymupdf) ───────────────────────────
            fitz_page = fitz_doc[page_num - 1]
            image_list = fitz_page.get_images(full=True)

            for img_idx, img_info in enumerate(image_list):
                xref = img_info[0]
                try:
                    base_image = fitz_doc.extract_image(xref)
                    img_bytes = base_image["image"]
                    img_ext = base_image["ext"]  # png, jpeg, etc.

                    pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

                    # Skip tiny images (icons, decorations)
                    if pil_image.width < 100 or pil_image.height < 100:
                        continue

                    # Save image to storage
                    img_filename = f"{doc_stem}_p{page_num}_img{img_idx + 1}.png"
                    img_save_path = str(image_dir / img_filename)
                    pil_image.save(img_save_path, "PNG")

                    # Surrounding text as caption context
                    surrounding = (page_text[:300] if page_text else "").replace("\n", " ")

                    metadata = {
                        "chunk_index": chunk_index,
                        "chunk_type": "image",
                        "source_file": filename,
                        "language": "en",
                        "word_count": 0,
                        "char_start": None,
                        "char_end": None,
                        "modal_metadata": {
                            "page_number": page_num,
                            "image_index": img_idx + 1,
                            "width": pil_image.width,
                            "height": pil_image.height,
                            "format": "PNG",
                            "caption": None,
                            "surrounding_text": surrounding,
                            "image_path": img_save_path,
                        },
                    }
                    chunks.append(ParsedChunk(
                        chunk_type="image",
                        content_text=surrounding or None,
                        image=pil_image,
                        image_path=img_save_path,
                        metadata=metadata,
                    ))
                    chunk_index += 1

                except Exception:
                    continue  # skip unreadable images

        fitz_doc.close()

    return chunks
