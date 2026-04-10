"""
Audio Parser — transcribes audio files and produces timed text chunks.
Delegates transcription to the active STT provider (cloud or local).

Chunking strategy:
  - Local Whisper: uses native segment timestamps (~30s blocks)
  - OpenAI Whisper API: uses verbose_json for segment timestamps
"""
import asyncio
import shutil
import tempfile
from pathlib import Path

from app.core.ingestion.schemas import ParsedChunk, SegmentResult


# Target chunk duration in seconds — group segments into ~30s blocks
TARGET_CHUNK_SECONDS = 30.0


def _group_segments(segments: list[SegmentResult]) -> list[list[SegmentResult]]:
    """Group Whisper segments into ~30-second blocks."""
    groups: list[list[SegmentResult]] = []
    current_group: list[SegmentResult] = []
    group_start = 0.0

    for seg in segments:
        current_group.append(seg)
        duration = seg.end_time - group_start
        if duration >= TARGET_CHUNK_SECONDS:
            groups.append(current_group)
            current_group = []
            group_start = seg.end_time

    if current_group:
        groups.append(current_group)

    return groups


async def _get_segments_local(audio_path: str) -> list[SegmentResult]:
    """Get timed segments using local Whisper model.

    On Windows, paths with spaces can cause Whisper/ffmpeg to raise
    [Errno 22] Invalid argument. We work around this by copying the
    audio file to the system temp directory (which has no spaces) first.
    """
    from app.config import get_settings
    settings = get_settings()

    def _run() -> list[SegmentResult]:
        import os
        import sys
        import whisper

        # ── Ensure ffmpeg is on PATH (fixes Windows: uvicorn started before PATH update) ──
        if sys.platform == "win32":
            try:
                import winreg
                key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment")
                user_path, _ = winreg.QueryValueEx(key, "PATH")
                winreg.CloseKey(key)
                # Inject any new entries the current process doesn't have
                current = os.environ.get("PATH", "")
                for entry in user_path.split(";"):
                    if entry and entry not in current:
                        os.environ["PATH"] = entry + ";" + os.environ["PATH"]
            except Exception:
                pass  # Not critical — Whisper will raise a clearer error if ffmpeg is missing

        # Reuse cached model from LocalWhisper provider
        from app.providers.local.local_whisper import _load_whisper_model
        model = _load_whisper_model()

        # Copy to a space-free temp path to avoid Windows [Errno 22] bug
        suffix = Path(audio_path).suffix or ".audio"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name
        try:
            shutil.copy2(audio_path, tmp_path)
            result = model.transcribe(tmp_path, fp16=False, verbose=False)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        return [
            SegmentResult(
                text=seg["text"].strip(),
                start_time=seg["start"],
                end_time=seg["end"],
                segment_index=i,
            )
            for i, seg in enumerate(result.get("segments", []))
        ]

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _run)


async def _get_segments_cloud(audio_path: str) -> list[SegmentResult]:
    """Get timed segments using OpenAI Whisper API (verbose_json)."""
    from openai import AsyncOpenAI
    from app.config import get_settings
    settings = get_settings()

    client = AsyncOpenAI(api_key=settings.openai_api_key)
    with open(audio_path, "rb") as f:
        transcript = await client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="verbose_json",
        )

    segments = getattr(transcript, "segments", []) or []
    return [
        SegmentResult(
            text=seg.text.strip() if hasattr(seg, "text") else str(seg.get("text", "")).strip(),
            start_time=seg.start if hasattr(seg, "start") else seg.get("start", 0.0),
            end_time=seg.end if hasattr(seg, "end") else seg.get("end", 0.0),
            segment_index=i,
        )
        for i, seg in enumerate(segments)
    ]


async def parse_audio(file_path: str, filename: str) -> list[ParsedChunk]:
    """Transcribe audio and return timed text ParsedChunks.

    Args:
        file_path: Absolute path to audio file.
        filename:  Original filename for metadata.

    Returns:
        List of ParsedChunk with audio chunk_type and timestamp metadata.
    """
    from app.config import get_settings
    settings = get_settings()

    # Get segments with timestamps from the appropriate STT provider
    if settings.stt_provider == "openai":
        segments = await _get_segments_cloud(file_path)
    else:
        segments = await _get_segments_local(file_path)

    if not segments:
        return []

    # Group into ~30-second chunks
    groups = _group_segments(segments)
    chunks: list[ParsedChunk] = []

    for chunk_index, group in enumerate(groups):
        combined_text = " ".join(seg.text for seg in group).strip()
        if not combined_text:
            continue

        start_time = group[0].start_time
        end_time = group[-1].end_time

        # Format timestamp for display (MM:SS)
        def fmt(t: float) -> str:
            m, s = divmod(int(t), 60)
            return f"{m:02d}:{s:02d}"

        metadata = {
            "chunk_index": chunk_index,
            "chunk_type": "audio",
            "source_file": filename,
            "language": "en",
            "word_count": len(combined_text.split()),
            "char_start": None,
            "char_end": None,
            "modal_metadata": {
                "segment_index": chunk_index,
                "timestamp_start": round(start_time, 2),
                "timestamp_end": round(end_time, 2),
                "timestamp_display": f"{fmt(start_time)} → {fmt(end_time)}",
                "confidence": None,  # not available from API in simple mode
            },
        }

        chunks.append(ParsedChunk(
            chunk_type="audio",
            content_text=combined_text,
            image=None,
            image_path=None,
            metadata=metadata,
        ))

    return chunks
