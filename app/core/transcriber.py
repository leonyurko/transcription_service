"""
Core Whisper transcription engine.
Wraps faster-whisper and exposes a clean async-friendly interface.
"""
import asyncio
import logging
from functools import partial
from pathlib import Path

from faster_whisper import WhisperModel

from app.db.database import set_progress
from config import settings
from app.models.schemas import Segment, TranscribeOptions, TranscribeResponse, WordTimestamp

logger = logging.getLogger(__name__)

_CUDA_ERROR_HINTS = ("cublas", "cudnn", "cublaslt", "cuda", "nvcuda")


def _is_cuda_error(exc: Exception) -> bool:
    return any(k in str(exc).lower() for k in _CUDA_ERROR_HINTS)


class Transcriber:
    """Singleton wrapper around a faster-whisper WhisperModel instance."""

    _instance: "Transcriber | None" = None

    def __init__(self) -> None:
        device = settings.device
        compute_type = settings.compute_type

        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

        if compute_type == "auto":
            compute_type = "int8" if device == "cpu" else "float16"

        self._device = device
        self._compute_type = compute_type
        self._build_model(device, compute_type)

    # ── Model loading ────────────────────────────────────────────────────────────

    def _build_model(self, device: str, compute_type: str) -> None:
        logger.info(
            "Loading faster-whisper model '%s' on %s (%s) ...",
            settings.model_size, device, compute_type,
        )
        self._model = WhisperModel(
            settings.model_size,
            device=device,
            compute_type=compute_type,
            cpu_threads=settings.cpu_threads,
            num_workers=settings.num_workers,
        )
        self._device = device
        self._compute_type = compute_type
        logger.info("Model ready on %s.", device)

    def _fallback_to_cpu(self, original_error: Exception) -> None:
        logger.warning(
            "CUDA error during inference (%s). Reloading model on CPU.", original_error
        )
        self._build_model("cpu", "int8")

    # ── Public API ──────────────────────────────────────────────────────────────

    async def transcribe(
        self,
        audio_path: str | Path,
        options: TranscribeOptions,
        job_id: int | None = None,
    ) -> TranscribeResponse:
        """Run transcription in a thread pool so the event loop stays free."""
        loop = asyncio.get_event_loop()
        fn = partial(self._transcribe_sync, str(audio_path), options, job_id)
        return await loop.run_in_executor(None, fn)

    # ── Internal ────────────────────────────────────────────────────────────────

    def _transcribe_sync(
        self,
        audio_path: str,
        options: TranscribeOptions,
        job_id: int | None,
    ) -> TranscribeResponse:
        try:
            return self._run(audio_path, options, job_id)
        except RuntimeError as exc:
            if _is_cuda_error(exc) and self._device != "cpu":
                self._fallback_to_cpu(exc)
                logger.info("Retrying transcription on CPU...")
                return self._run(audio_path, options, job_id)
            raise

    def _run(
        self,
        audio_path: str,
        options: TranscribeOptions,
        job_id: int | None,
    ) -> TranscribeResponse:
        segments_iter, info = self._model.transcribe(
            audio_path,
            language=options.language,
            task=options.task,
            beam_size=options.beam_size,
            word_timestamps=options.word_timestamps,
            vad_filter=options.vad_filter,
            vad_parameters={"min_silence_duration_ms": settings.vad_min_silence_duration_ms},
            temperature=options.temperature if isinstance(options.temperature, list) else [options.temperature],
            initial_prompt=options.initial_prompt,
            condition_on_previous_text=options.condition_on_previous_text,
            hotwords=options.hotwords,
        )

        segments: list[Segment] = []
        full_text_parts: list[str] = []
        duration = info.duration or 1  # guard against zero-length audio

        for seg in segments_iter:
            words = None
            if options.word_timestamps and seg.words:
                words = [
                    WordTimestamp(
                        word=w.word,
                        start=round(w.start, 3),
                        end=round(w.end, 3),
                        probability=round(w.probability, 4),
                    )
                    for w in seg.words
                ]
            segments.append(
                Segment(
                    id=seg.id,
                    start=round(seg.start, 3),
                    end=round(seg.end, 3),
                    text=seg.text.strip(),
                    words=words,
                    avg_logprob=round(seg.avg_logprob, 4),
                    compression_ratio=round(seg.compression_ratio, 4),
                    no_speech_prob=round(seg.no_speech_prob, 4),
                )
            )
            full_text_parts.append(seg.text.strip())

            # Write progress (capped at 99 — 100 is set when job is marked done)
            if job_id is not None:
                pct = min(int(seg.end / duration * 100), 99)
                set_progress(job_id, pct)

        return TranscribeResponse(
            text=" ".join(full_text_parts),
            language=info.language,
            language_probability=round(info.language_probability, 4),
            duration=round(info.duration, 3),
            segments=segments,
            model=settings.model_size,
        )

    # ── Singleton factory ───────────────────────────────────────────────────────

    @classmethod
    def get(cls) -> "Transcriber":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reload(cls, model_size: str | None = None, device: str | None = None) -> None:
        if model_size:
            settings.model_size = model_size
        if device:
            settings.device = device
        logger.info(
            "Reloading → model=%s  device=%s", settings.model_size, settings.device
        )
        cls._instance = None
        cls._instance = cls()

    @property
    def device(self) -> str:
        return self._device

    @property
    def compute_type(self) -> str:
        return self._compute_type
