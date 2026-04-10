"""
CLIP Embedder — produces 512-dim vectors for both text and images.
Text and image embeddings share the same vector space (key multimodal property).
Model is loaded once and cached for the lifetime of the process.
"""
import asyncio
from functools import lru_cache
from typing import Optional

import torch
from PIL import Image

from app.config import get_settings

settings = get_settings()


@lru_cache(maxsize=1)
def _load_clip():
    """Load CLIP model + processor once and cache."""
    from transformers import CLIPModel, CLIPProcessor

    print(f"[CLIP] Loading model: {settings.clip_model}")
    model = CLIPModel.from_pretrained(settings.clip_model)
    processor = CLIPProcessor.from_pretrained(settings.clip_model)
    model.eval()  # inference mode
    return model, processor


class CLIPEmbedder:
    """Singleton-style embedder using CLIP ViT-B/32.

    Usage:
        embedder = CLIPEmbedder()
        text_vec = await embedder.embed_text("Revenue grew in Q4")
        img_vec  = await embedder.embed_image(pil_image)
    """

    async def embed_text(self, text: str) -> list[float]:
        """Embed a text string → 512-dim normalized vector."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._embed_text_sync, text)

    async def embed_image(self, image: Image.Image) -> list[float]:
        """Embed a PIL image → 512-dim normalized vector."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._embed_image_sync, image)

    async def embed_texts_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts — more efficient than individual calls."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._embed_texts_batch_sync, texts)

    @staticmethod
    def _embed_text_sync(text: str) -> list[float]:
        model, processor = _load_clip()
        # CLIP text encoder max is 77 tokens — truncate if needed
        inputs = processor(
            text=[text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        )
        with torch.no_grad():
            features = model.get_text_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
        return features[0].cpu().numpy().tolist()

    @staticmethod
    def _embed_texts_batch_sync(texts: list[str]) -> list[list[float]]:
        model, processor = _load_clip()
        inputs = processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        )
        with torch.no_grad():
            features = model.get_text_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().tolist()

    @staticmethod
    def _embed_image_sync(image: Image.Image) -> list[float]:
        model, processor = _load_clip()
        # Convert to RGB in case of RGBA/grayscale
        image = image.convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            features = model.get_image_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
        return features[0].cpu().numpy().tolist()


# Singleton instance — import and use directly
embedder = CLIPEmbedder()
