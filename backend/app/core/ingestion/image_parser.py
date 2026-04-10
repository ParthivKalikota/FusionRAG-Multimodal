"""
Image Parser — handles standalone image file uploads (JPEG, PNG, WEBP, etc.)
The image is saved to storage and returned as a single ParsedChunk
for CLIP image embedding.
"""
import shutil
from pathlib import Path

from PIL import Image

from app.config import get_settings
from app.core.ingestion.schemas import ParsedChunk

settings = get_settings()


def parse_image(file_path: str, filename: str) -> list[ParsedChunk]:
    """Load an image file and return a single image ParsedChunk.

    Args:
        file_path: Absolute path to the image file.
        filename:  Original filename for metadata.

    Returns:
        A list containing a single image ParsedChunk.
    """
    image_dir = Path(settings.storage_dir)
    image_dir.mkdir(parents=True, exist_ok=True)

    # Load and convert to RGB
    pil_image = Image.open(file_path).convert("RGB")
    width, height = pil_image.size

    # Save a copy to storage directory as PNG
    stem = Path(filename).stem
    save_name = f"{stem}_upload.png"
    save_path = str(image_dir / save_name)
    pil_image.save(save_path, "PNG")

    metadata = {
        "chunk_index": 0,
        "chunk_type": "image",
        "source_file": filename,
        "language": "en",
        "word_count": 0,
        "char_start": None,
        "char_end": None,
        "modal_metadata": {
            "page_number": None,
            "image_index": 1,
            "width": width,
            "height": height,
            "format": "PNG",
            "caption": None,
            "surrounding_text": None,
            "image_path": save_path,
        },
    }

    return [
        ParsedChunk(
            chunk_type="image",
            content_text=None,  # no text — will use image embedding
            image=pil_image,
            image_path=save_path,
            metadata=metadata,
        )
    ]
