from pydantic_settings import BaseSettings
from typing import Literal


class Settings(BaseSettings):
    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # Model
    model_size: Literal[
        "tiny", "tiny.en",
        "base", "base.en",
        "small", "small.en",
        "medium", "medium.en",
        "large-v1", "large-v2", "large-v3",
        "large-v3-turbo",
        "distil-large-v2", "distil-large-v3",
        "distil-medium.en", "distil-small.en",
    ] = "large-v3-turbo"

    device: Literal["auto", "cpu", "cuda"] = "auto"
    compute_type: Literal["auto", "int8", "int8_float16", "int16", "float16", "float32"] = "auto"
    cpu_threads: int = 4
    num_workers: int = 1

    # Transcription defaults
    default_language: str | None = None   # None = auto-detect
    default_beam_size: int = 5
    vad_filter: bool = True
    vad_min_silence_duration_ms: int = 500

    # File handling
    max_upload_mb: int = 500
    tmp_dir: str = "/tmp/transcribe_service"

    # API
    api_key: str | None = None            # Optional bearer token auth

    class Config:
        env_prefix = "TRANSCRIBE_"
        env_file = ".env"


settings = Settings()
