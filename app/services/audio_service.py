"""
Handles all audio ingestion: file upload saving and URL download.
Returns a local path ready for the transcriber.
"""
import logging
import os
import uuid
from pathlib import Path

import aiofiles
import httpx
from fastapi import HTTPException, UploadFile

from config import settings

logger = logging.getLogger(__name__)

_SUPPORTED_EXTENSIONS = {
    ".mp3", ".mp4", ".m4a", ".wav", ".flac", ".ogg",
    ".opus", ".webm", ".weba", ".mkv", ".avi", ".mov", ".aac",
    ".wma", ".aiff", ".aif",
}


def _tmp_path(suffix: str = "") -> Path:
    tmp = Path(settings.tmp_dir)
    tmp.mkdir(parents=True, exist_ok=True)
    return tmp / f"{uuid.uuid4().hex}{suffix}"


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
