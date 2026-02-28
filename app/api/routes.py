"""
FastAPI route definitions.
"""
import logging
import uuid
import webbrowser
from pathlib import Path

import aiofiles
from fastapi import (
    APIRouter, BackgroundTasks, Body, Depends, File, Form,
    HTTPException, Security, UploadFile,
)
from fastapi.responses import FileResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.transcriber import Transcriber
from app.db import database as db
from app.models.schemas import (
    HealthResponse,
    TranscribeOptions,
    TranscribeResponse,
    TranscribeUrlRequest,
)
from app.services import audio_service, translation_service
from config import settings

logger = logging.getLogger(__name__)
router = APIRouter()
_bearer = HTTPBearer(auto_error=False)

_SUPPORTED_EXT = {
    ".mp3", ".mp4", ".m4a", ".wav", ".flac", ".ogg",
    ".opus", ".webm", ".weba", ".mkv", ".avi", ".mov", ".aac",
    ".wma", ".aiff", ".aif",
}

_KNOWN_MODELS = [
    "tiny", "tiny.en",
    "base", "base.en",
    "small", "small.en",
    "medium", "medium.en",
    "large-v1", "large-v2", "large-v3",
    "large-v3-turbo",
    "distil-large-v2", "distil-large-v3",
    "distil-medium.en", "distil-small.en",
]


# ── Auth ──────────────────────────────────────────────────────────────────────

def _check_auth(
    creds: HTTPAuthorizationCredentials | None = Security(_bearer),
) -> None:
    if settings.api_key is None:
        return
    if creds is None or creds.credentials != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


# ── Internal helpers ──────────────────────────────────────────────────────────

async def _save_to_uploads(file: UploadFile) -> tuple[Path, int]:
    """Save an uploaded file to the persistent uploads directory."""
    max_bytes = settings.max_upload_mb * 1024 * 1024
    suffix = Path(file.filename or "audio").suffix.lower() or ".audio"
    if suffix not in _SUPPORTED_EXT:
        raise HTTPException(415, f"Unsupported file type '{suffix}'.")
    dest = db.UPLOADS_DIR / f"{uuid.uuid4().hex}{suffix}"
    size = 0
    async with aiofiles.open(dest, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > max_bytes:
                dest.unlink(missing_ok=True)
                raise HTTPException(413, f"File exceeds {settings.max_upload_mb} MB limit.")
            await f.write(chunk)
    return dest, size


async def _run_transcription(
    job_id: int,
    audio_path: Path,
    options: TranscribeOptions,
    delete_after: bool,
    translation_lang: str | None = None,
) -> None:
    """Background task: extract audio if video, transcribe, persist result, then optionally translate."""
    wav_path: Path | None = None
    try:
        # Extract audio track first if the upload is a video file
        if audio_service.is_video(audio_path):
            wav_path = audio_service.extract_audio(audio_path)
            transcribe_path = wav_path
        else:
            transcribe_path = audio_path

        result = await Transcriber.get().transcribe(transcribe_path, options, job_id=job_id)
        await db.update_job_done(job_id, result)

        if translation_lang and result.segments:
            try:
                import json as _json
                seg_dicts = [{"start": s.start, "end": s.end, "text": s.text} for s in result.segments]
                t_segs = await translation_service.translate_segments(seg_dicts, translation_lang)
                t_text = " ".join(s["text"] for s in t_segs)
                await db.update_job_translation(job_id, t_text, translation_lang, _json.dumps(t_segs))
            except Exception as t_exc:
                logger.warning(
                    "Translation failed for job %d (lang=%s): %s", job_id, translation_lang, t_exc
                )
    except Exception as exc:
        logger.exception("Transcription failed for job %d", job_id)
        await db.update_job_error(job_id, exc)
    finally:
        if wav_path:
            audio_service.cleanup(wav_path)
        if delete_after:
            audio_service.cleanup(audio_path)


def _scan_local_models() -> list[dict]:
    cache = Path.home() / ".cache" / "huggingface" / "hub"
    local: set[str] = set()
    if cache.exists():
        for d in cache.iterdir():
            if d.name.startswith("models--Systran--faster-whisper-"):
                local.add(d.name[len("models--Systran--faster-whisper-"):])
    return [{"name": m, "local": m in local} for m in _KNOWN_MODELS]


# ── Utility routes ─────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, tags=["Utility"])
async def health() -> HealthResponse:
    t = Transcriber.get()
    return HealthResponse(
        status="ok",
        model=settings.model_size,
        device=t.device,
        compute_type=t.compute_type,
    )


@router.get("/models", tags=["Utility"])
async def list_models(_: None = Depends(_check_auth)) -> list[dict]:
    """List all known Whisper models, marking which are cached locally."""
    return _scan_local_models()


@router.get("/config", tags=["Utility"])
async def get_config(_: None = Depends(_check_auth)) -> dict:
    t = Transcriber.get()
    return {
        "model_size": settings.model_size,
        "device": t.device,
        "compute_type": t.compute_type,
        "api_key_set": settings.api_key is not None,
    }


@router.patch("/config", tags=["Utility"])
async def update_config(
    body: dict = Body(...),
    _: None = Depends(_check_auth),
) -> dict:
    """Hot-swap the active Whisper model and/or device (blocks until loaded)."""
    Transcriber.reload(
        model_size=body.get("model_size"),
        device=body.get("device"),
    )
    t = Transcriber.get()
    return {"status": "ok", "model_size": settings.model_size, "device": t.device}


# ── Async job routes ───────────────────────────────────────────────────────────
# NOTE: /jobs/file and /jobs/url must be registered BEFORE /jobs/{job_id}
# so FastAPI doesn't interpret "file"/"url" as an integer job_id.

@router.post("/jobs/file", tags=["Jobs"], status_code=202)
async def job_from_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    language: str | None = Form(default=None),
    task: str = Form(default="transcribe"),
    beam_size: int = Form(default=5),
    word_timestamps: bool = Form(default=False),
    vad_filter: bool = Form(default=True),
    temperature: float = Form(default=0.0),
    initial_prompt: str | None = Form(default=None),
    condition_on_previous_text: bool = Form(default=True),
    hotwords: str | None = Form(default=None),
    translation_lang: str | None = Form(default=None),
    _: None = Depends(_check_auth),
) -> dict:
    """Upload a file and queue it for transcription. Returns job_id immediately."""
    audio_path, file_size = await _save_to_uploads(file)
    options = TranscribeOptions(
        language=language or None,
        task=task,  # type: ignore[arg-type]
        beam_size=beam_size,
        word_timestamps=word_timestamps,
        vad_filter=vad_filter,
        temperature=temperature,
        initial_prompt=initial_prompt,
        condition_on_previous_text=condition_on_previous_text,
        hotwords=hotwords,
    )
    job_id = await db.create_job(
        filename=file.filename or audio_path.name,
        file_path=audio_path,
        file_size=file_size,
        model=settings.model_size,
        lang_req=language or "",
        task=task,
        translation_lang=translation_lang or None,
    )
    background_tasks.add_task(
        _run_transcription, job_id, audio_path, options, False, translation_lang or None
    )
    return {"job_id": job_id, "status": "processing"}


@router.post("/jobs/url", tags=["Jobs"], status_code=202)
async def job_from_url(
    background_tasks: BackgroundTasks,
    body: TranscribeUrlRequest,
    _: None = Depends(_check_auth),
) -> dict:
    """Queue a URL for download + transcription. Returns job_id immediately."""
    url_str = str(body.url)
    filename = Path(url_str.split("?")[0]).name or "audio"
    job_id = await db.create_job(
        filename=filename,
        file_path=None,
        file_size=None,
        model=settings.model_size,
        lang_req=body.language or "",
        task=body.task,
    )
    options = TranscribeOptions(**body.model_dump(exclude={"url"}))

    async def _bg() -> None:
        import asyncio, functools
        audio_path: Path | None = None
        loop = asyncio.get_event_loop()
        try:
            if audio_service.is_zoom_url(url_str):
                audio_path = await loop.run_in_executor(
                    None,
                    functools.partial(
                        audio_service.download_zoom,
                        url_str,
                        passcode=body.passcode,
                    ),
                )
            elif audio_service.is_youtube_url(url_str):
                audio_path = await loop.run_in_executor(
                    None,
                    functools.partial(audio_service.download_youtube, url_str),
                )
            else:
                audio_path = await audio_service.download_url(url_str)
            await _run_transcription(job_id, audio_path, options, True)
        except Exception as exc:
            await db.update_job_error(job_id, exc)

    background_tasks.add_task(_bg)
    return {"job_id": job_id, "status": "processing"}


@router.get("/zoom/auth", tags=["Zoom"])
async def zoom_auth(
    url: str | None = None,
    _: None = Depends(_check_auth),
) -> dict:
    """Open the Zoom recording (or sign-in page) in the user's default browser."""
    target = url or "https://zoom.us/signin"
    webbrowser.open(target)
    return {"opened": True, "url": target}


@router.get("/jobs", tags=["Jobs"])
async def list_jobs(_: None = Depends(_check_auth)) -> list[dict]:
    return await db.list_jobs()


@router.get("/jobs/{job_id}", tags=["Jobs"])
async def get_job(job_id: int, _: None = Depends(_check_auth)) -> dict:
    job = await db.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found.")
    return job


@router.delete("/jobs/{job_id}", tags=["Jobs"], status_code=204)
async def delete_job(job_id: int, _: None = Depends(_check_auth)) -> None:
    if not await db.delete_job(job_id):
        raise HTTPException(404, "Job not found.")


@router.patch("/jobs/{job_id}", tags=["Jobs"])
async def rename_job(
    job_id: int,
    body: dict = Body(...),
    _: None = Depends(_check_auth),
) -> dict:
    """Rename the display filename of a job (used as audio/transcript download name)."""
    new_name = (body.get("filename") or "").strip()
    if not new_name:
        raise HTTPException(422, "filename must not be empty.")
    if not await db.rename_job(job_id, new_name):
        raise HTTPException(404, "Job not found.")
    return {"job_id": job_id, "filename": new_name}


@router.get("/audio/{job_id}", tags=["Jobs"])
async def get_audio(job_id: int, _: None = Depends(_check_auth)) -> FileResponse:
    job = await db.get_job(job_id)
    if not job or not job.get("file_path"):
        raise HTTPException(404, "Audio file not found.")
    p = Path(job["file_path"])
    if not p.exists():
        raise HTTPException(404, "Audio file missing from disk.")
    return FileResponse(str(p), filename=job["filename"])


# ── Legacy sync routes (kept for backward compatibility) ───────────────────────

@router.post(
    "/transcribe/file",
    response_model=TranscribeResponse,
    tags=["Transcription (sync)"],
    summary="Synchronous file transcription — blocks until done",
)
async def transcribe_file(
    file: UploadFile = File(...),
    language: str | None = Form(default=None),
    task: str = Form(default="transcribe"),
    beam_size: int = Form(default=5),
    word_timestamps: bool = Form(default=False),
    vad_filter: bool = Form(default=True),
    temperature: float = Form(default=0.0),
    initial_prompt: str | None = Form(default=None),
    condition_on_previous_text: bool = Form(default=True),
    hotwords: str | None = Form(default=None),
    _: None = Depends(_check_auth),
) -> TranscribeResponse:
    options = TranscribeOptions(
        language=language,
        task=task,  # type: ignore[arg-type]
        beam_size=beam_size,
        word_timestamps=word_timestamps,
        vad_filter=vad_filter,
        temperature=temperature,
        initial_prompt=initial_prompt,
        condition_on_previous_text=condition_on_previous_text,
        hotwords=hotwords,
    )
    audio_path: Path | None = None
    wav_path: Path | None = None
    try:
        audio_path = await audio_service.save_upload(file)
        if audio_service.is_video(audio_path):
            wav_path = audio_service.extract_audio(audio_path)
            return await Transcriber.get().transcribe(wav_path, options)
        return await Transcriber.get().transcribe(audio_path, options)
    finally:
        if wav_path:
            audio_service.cleanup(wav_path)
        if audio_path:
            audio_service.cleanup(audio_path)


@router.post(
    "/transcribe/url",
    response_model=TranscribeResponse,
    tags=["Transcription (sync)"],
    summary="Synchronous URL transcription — blocks until done",
)
async def transcribe_url(
    body: TranscribeUrlRequest,
    _: None = Depends(_check_auth),
) -> TranscribeResponse:
    audio_path: Path | None = None
    wav_path: Path | None = None
    try:
        if audio_service.is_zoom_url(str(body.url)):
            audio_path = audio_service.download_zoom(str(body.url), passcode=body.passcode)
        elif audio_service.is_youtube_url(str(body.url)):
            audio_path = audio_service.download_youtube(str(body.url))
        else:
            audio_path = await audio_service.download_url(str(body.url))
        options = TranscribeOptions(**body.model_dump(exclude={"url"}))
        if audio_service.is_video(audio_path):
            wav_path = audio_service.extract_audio(audio_path)
            return await Transcriber.get().transcribe(wav_path, options)
        return await Transcriber.get().transcribe(audio_path, options)
    finally:
        if wav_path:
            audio_service.cleanup(wav_path)
        if audio_path:
            audio_service.cleanup(audio_path)
