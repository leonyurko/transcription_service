"""
Handles all audio ingestion: file upload saving and URL download.
For video files, the audio track is extracted via ffmpeg before transcription.
Returns a local path ready for the transcriber.
"""
import logging
import os
import subprocess
import uuid
from pathlib import Path

import aiofiles
import ffmpeg
import httpx
from fastapi import HTTPException, UploadFile

from config import settings

logger = logging.getLogger(__name__)

_SUPPORTED_EXTENSIONS = {
    ".mp3", ".mp4", ".m4a", ".wav", ".flac", ".ogg",
    ".opus", ".webm", ".weba", ".mkv", ".avi", ".mov", ".aac",
    ".wma", ".aiff", ".aif",
}

# Extensions that carry a video stream — audio must be extracted first
_VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".weba"}


def _tmp_path(suffix: str = "") -> Path:
    tmp = Path(settings.tmp_dir)
    tmp.mkdir(parents=True, exist_ok=True)
    return tmp / f"{uuid.uuid4().hex}{suffix}"


def is_video(path: Path) -> bool:
    """Return True if the file extension indicates a video container."""
    return path.suffix.lower() in _VIDEO_EXTENSIONS


def extract_audio(video_path: Path) -> Path:
    """
    Extract the audio track from a video file and write it as a 16 kHz
    mono WAV to the tmp directory.  Returns the path to the WAV file.
    Raises HTTPException(422) with the ffmpeg error message on failure.
    """
    out_path = _tmp_path(".wav")
    try:
        (
            ffmpeg
            .input(str(video_path))
            .output(
                str(out_path),
                vn=None,          # drop video stream
                acodec="pcm_s16le",
                ar=16000,         # 16 kHz — optimal for Whisper
                ac=1,             # mono
            )
            .overwrite_output()
            .run(quiet=True)
        )
    except ffmpeg.Error as exc:
        stderr = exc.stderr.decode(errors="replace") if exc.stderr else str(exc)
        logger.error("ffmpeg audio extraction failed: %s", stderr)
        out_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=422,
            detail=f"Failed to extract audio from video: {stderr.strip()}",
        ) from exc
    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail="ffmpeg is not installed or not found on PATH. "
                   "Install it from https://ffmpeg.org/download.html",
        )
    logger.debug("Extracted audio to %s", out_path)
    return out_path


# ── YouTube support ───────────────────────────────────────────────────────────

import re as _re

_YT_PATTERN = _re.compile(
    r"(https?://)?(www\.)?"
    r"(youtube\.com/(watch|shorts|live|embed)|youtu\.be/)",
    _re.IGNORECASE,
)


def is_youtube_url(url: str) -> bool:
    """Return True if the URL points to YouTube."""
    return bool(_YT_PATTERN.search(url))


def download_youtube(url: str) -> Path:
    """
    Download the audio track from a YouTube URL using yt-dlp and return
    the path to a 16 kHz mono WAV file in the tmp directory.
    Raises HTTPException(422) on yt-dlp error.
    """
    import sys as _sys

    # Use a stem WITHOUT extension — yt-dlp appends the final .wav itself.
    stem = _tmp_path("")          # e.g. /tmp/transcribe_service/abc123
    expected = stem.with_suffix(".wav")  # what yt-dlp will produce

    cmd = [
        _sys.executable, "-m", "yt_dlp",   # via python -m, avoids PATH issues
        "--no-playlist",
        "--extract-audio",
        "--audio-format", "wav",
        "--postprocessor-args", "ffmpeg:-ar 16000 -ac 1",
        "--output", str(stem),              # no .wav here — yt-dlp adds it
        "--no-progress",
        "--quiet",
        url,
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace")
        logger.error("yt-dlp failed: %s", stderr)
        expected.unlink(missing_ok=True)
        raise HTTPException(
            status_code=422,
            detail=f"yt-dlp failed to download audio: {stderr.strip()}",
        )

    # Resolve the actual output file (normally <stem>.wav)
    if expected.exists():
        out_path = expected
    else:
        candidates = sorted(stem.parent.glob(stem.name + ".*"))
        if not candidates:
            raise HTTPException(status_code=500, detail="yt-dlp produced no output file.")
        out_path = candidates[0]

    logger.debug("Downloaded YouTube audio to %s", out_path)
    return out_path


async def save_upload(file: UploadFile) -> Path:
    """Persist an uploaded file to the tmp dir and return its path."""
    max_bytes = settings.max_upload_mb * 1024 * 1024
    suffix = Path(file.filename or "audio").suffix.lower() or ".audio"

    if suffix not in _SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{suffix}'. Supported: {sorted(_SUPPORTED_EXTENSIONS)}",
        )

    dest = _tmp_path(suffix)
    size = 0
    chunk_size = 1024 * 1024  # 1 MB

    async with aiofiles.open(dest, "wb") as f:
        while True:
            chunk = await file.read(chunk_size)
            if not chunk:
                break
            size += len(chunk)
            if size > max_bytes:
                dest.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"File exceeds the {settings.max_upload_mb} MB limit.",
                )
            await f.write(chunk)

    logger.debug("Saved upload to %s (%d bytes)", dest, size)
    return dest


async def download_url(url: str) -> Path:
    """Stream a remote audio/video URL to the tmp dir and return its path."""
    max_bytes = settings.max_upload_mb * 1024 * 1024
    suffix = Path(url.split("?")[0]).suffix.lower() or ".audio"

    dest = _tmp_path(suffix if suffix in _SUPPORTED_EXTENSIONS else ".audio")
    size = 0

    async with httpx.AsyncClient(follow_redirects=True, timeout=120) as client:
        async with client.stream("GET", url) as response:
            if response.status_code != 200:
                raise HTTPException(
                    status_code=400,
                    detail=f"Could not fetch URL (HTTP {response.status_code}).",
                )
            async with aiofiles.open(dest, "wb") as f:
                async for chunk in response.aiter_bytes(chunk_size=1024 * 1024):
                    size += len(chunk)
                    if size > max_bytes:
                        dest.unlink(missing_ok=True)
                        raise HTTPException(
                            status_code=413,
                            detail=f"Remote file exceeds the {settings.max_upload_mb} MB limit.",
                        )
                    await f.write(chunk)

    logger.debug("Downloaded URL to %s (%d bytes)", dest, size)
    return dest


def cleanup(path: Path) -> None:
    """Remove a temporary file silently."""
    try:
        os.unlink(path)
    except OSError:
        pass
