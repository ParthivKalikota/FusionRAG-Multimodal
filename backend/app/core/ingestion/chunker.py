"""
Tiktoken-based sliding window chunker.
Splits text into overlapping chunks of ~512 tokens with 10% overlap.
"""
from dataclasses import dataclass
from typing import Optional

import tiktoken

# cl100k_base is used by GPT-4 and CLIP text tokenizer is similar in size
_ENCODING = tiktoken.get_encoding("cl100k_base")

CHUNK_TOKENS = 512
OVERLAP_TOKENS = 51   # ~10% overlap


def chunk_text(
    text: str,
    source_file: str,
    chunk_type: str = "text",
    language: str = "en",
    page_number: Optional[int] = None,
    section_heading: Optional[str] = None,
    start_chunk_index: int = 0,
) -> list[dict]:
    """
    Split text into overlapping token-based chunks.

    Returns a list of unified metadata dicts (without embedding).
    Each dict is ready to be wrapped into a ParsedChunk.
    """
    if not text or not text.strip():
        return []

    text = text.strip()
    tokens = _ENCODING.encode(text)
    chunks = []
    chunk_index = start_chunk_index
    pos = 0

    while pos < len(tokens):
        end = min(pos + CHUNK_TOKENS, len(tokens))
        chunk_tokens = tokens[pos:end]
        chunk_text_str = _ENCODING.decode(chunk_tokens)

        # Estimate character positions
        # (rough estimation based on token ratio)
        char_ratio = len(text) / max(len(tokens), 1)
        char_start = int(pos * char_ratio)
        char_end = int(end * char_ratio)

        word_count = len(chunk_text_str.split())

        modal_metadata: dict = {}
        if page_number is not None:
            modal_metadata["page_number"] = page_number
        if section_heading:
            modal_metadata["section_heading"] = section_heading
        modal_metadata["paragraph_index"] = chunk_index - start_chunk_index

        metadata = {
            "chunk_index": chunk_index,
            "chunk_type": chunk_type,
            "source_file": source_file,
            "language": language,
            "word_count": word_count,
            "char_start": char_start,
            "char_end": char_end,
            "modal_metadata": modal_metadata,
        }

        chunks.append({
            "content_text": chunk_text_str,
            "metadata": metadata,
        })

        chunk_index += 1
        if end == len(tokens):
            break
        pos = end - OVERLAP_TOKENS  # slide with overlap

    return chunks


def count_tokens(text: str) -> int:
    """Count tokens in a string."""
    return len(_ENCODING.encode(text))
