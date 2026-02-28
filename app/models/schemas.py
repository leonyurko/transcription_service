from pydantic import BaseModel, Field, HttpUrl
from typing import Literal


# ── Request models ─────────────────────────────────────────────────────────────

class TranscribeOptions(BaseModel):
    language: str | None = Field(
        default=None,
        description="BCP-47 language code (e.g. 'en', 'de'). None = auto-detect.",
    )
    task: Literal["transcribe", "translate"] = Field(
        default="transcribe",
        description="'transcribe' keeps original language. 'translate' outputs English.",
    )
    beam_size: int = Field(default=5, ge=1, le=10)
    word_timestamps: bool = Field(
        default=False,
        description="Include per-word start/end timestamps in the response.",
    )
    vad_filter: bool = Field(
        default=True,
        description="Skip non-speech segments using Voice Activity Detection.",
    )
    temperature: float | list[float] = Field(
        default=0.0,
        description="Sampling temperature. 0 = greedy decoding.",
    )
    initial_prompt: str | None = Field(
        default=None,
        description="Optional text to prime the transcription context.",
    )
    condition_on_previous_text: bool = Field(
        default=True,
        description="Feed previous output as prompt for the next segment.",
    )
    hotwords: str | None = Field(
        default=None,
        description="Comma-separated hotwords / proper nouns to boost.",
    )


class TranscribeUrlRequest(TranscribeOptions):
    url: HttpUrl = Field(description="Publicly accessible audio/video URL.")
    passcode: str | None = Field(default=None, description="Passcode for password-protected Zoom recordings.")


# ── Response models ─────────────────────────────────────────────────────────────

class WordTimestamp(BaseModel):
    word: str
    start: float
    end: float
    probability: float


class Segment(BaseModel):
    id: int
    start: float
    end: float
    text: str
    words: list[WordTimestamp] | None = None
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float


class TranscribeResponse(BaseModel):
    text: str = Field(description="Full concatenated transcript.")
    language: str = Field(description="Detected or specified language code.")
    language_probability: float
    duration: float = Field(description="Audio duration in seconds.")
    segments: list[Segment]
    model: str


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str
    compute_type: str
