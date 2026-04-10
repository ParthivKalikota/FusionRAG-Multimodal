"""
Shared dataclasses for the ingestion pipeline.
ParsedChunk flows from parsers → CLIP embedder → DB insertion.
"""
from dataclasses import dataclass, field
from typing import Optional
from PIL import Image


@dataclass
class ParsedChunk:
    """Intermediate representation of a parsed chunk before embedding."""

    chunk_type: str                    # text | table | image | audio
    content_text: Optional[str]        # textual content for all types
    image: Optional[Image.Image]       # PIL image (image chunks only)
    image_path: Optional[str]          # saved image path (image chunks only)
    metadata: dict = field(default_factory=dict)
    # Unified metadata shape:
    # {
    #   chunk_index, chunk_type, source_file, language,
    #   word_count, char_start, char_end,
    #   modal_metadata: { ...modality-specific }
    # }


@dataclass
class SegmentResult:
    """A timed transcript segment from a STT provider."""
    text: str
    start_time: float   # seconds
    end_time: float     # seconds
    segment_index: int
