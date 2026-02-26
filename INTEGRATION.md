# Transcription Service — Integration Reference

## Quick Start

### Option A — Python + uvicorn (local)
```bash
pip install -r requirements.txt
# ffmpeg must be installed on the system: https://ffmpeg.org/download.html
cp .env.example .env        # adjust settings
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Option B — Docker
```bash
cp .env.example .env
docker compose up --build
```

---

## Base URL
```
http://<host>:8000/api/v1
```

---

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/health` | Liveness check, returns model/device info |
| POST | `/api/v1/transcribe/file` | Transcribe an uploaded file (`multipart/form-data`) |
| POST | `/api/v1/transcribe/url` | Transcribe from a remote URL (`application/json`) |

Interactive docs: `http://localhost:8000/docs`

---

## Authentication (optional)

Set `TRANSCRIBE_API_KEY=your-secret` in `.env`.
Pass it as a Bearer token:
```
Authorization: Bearer your-secret
```
Leave the env var empty to disable auth entirely.

---

## POST /transcribe/file

**Content-Type:** `multipart/form-data`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `file` | file | required | Audio/video file |
| `language` | string | auto | BCP-47 code (`en`, `de`, `fr` …) |
| `task` | string | `transcribe` | `transcribe` or `translate` (→ English) |
| `beam_size` | int | 5 | Beam search width (1–10) |
| `word_timestamps` | bool | false | Add per-word start/end times |
| `vad_filter` | bool | true | Skip silence with VAD |
| `temperature` | float | 0.0 | Sampling temp (0 = greedy) |
| `initial_prompt` | string | — | Context primer text |
| `hotwords` | string | — | Comma-separated terms to boost |

### Example — curl
```bash
curl -X POST http://localhost:8000/api/v1/transcribe/file \
  -F "file=@recording.mp3" \
  -F "language=en" \
  -F "word_timestamps=true"
```

### Example — Python (requests)
```python
import requests

with open("recording.mp3", "rb") as f:
    resp = requests.post(
        "http://localhost:8000/api/v1/transcribe/file",
        files={"file": f},
        data={"language": "en", "word_timestamps": "true"},
    )
result = resp.json()
print(result["text"])
```

### Example — JavaScript (fetch)
```js
const form = new FormData();
form.append("file", audioBlob, "audio.mp3");
form.append("language", "en");

const res = await fetch("http://localhost:8000/api/v1/transcribe/file", {
  method: "POST",
  body: form,
});
const { text, segments } = await res.json();
```

---

## POST /transcribe/url

**Content-Type:** `application/json`

```json
{
  "url": "https://example.com/podcast.mp3",
  "language": "en",
  "task": "transcribe",
  "beam_size": 5,
  "word_timestamps": false,
  "vad_filter": true,
  "temperature": 0.0,
  "initial_prompt": null,
  "hotwords": null
}
```

### Example — curl
```bash
curl -X POST http://localhost:8000/api/v1/transcribe/url \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/audio.mp3", "language": "en"}'
```

---

## Response Schema

```json
{
  "text": "Full transcript as a single string.",
  "language": "en",
  "language_probability": 0.9987,
  "duration": 142.35,
  "model": "large-v3-turbo",
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 4.2,
      "text": "Hello and welcome to the show.",
      "avg_logprob": -0.21,
      "compression_ratio": 1.45,
      "no_speech_prob": 0.002,
      "words": null
    }
  ]
}
```

`words` array (when `word_timestamps=true`):
```json
[{ "word": "Hello", "start": 0.0, "end": 0.38, "probability": 0.99 }]
```

---

## Configuration Reference (env vars)

| Variable | Default | Description |
|----------|---------|-------------|
| `TRANSCRIBE_MODEL_SIZE` | `large-v3-turbo` | Whisper model variant |
| `TRANSCRIBE_DEVICE` | `auto` | `auto` / `cpu` / `cuda` |
| `TRANSCRIBE_COMPUTE_TYPE` | `auto` | `auto` / `int8` / `float16` / `float32` |
| `TRANSCRIBE_CPU_THREADS` | `4` | CPU thread count |
| `TRANSCRIBE_NUM_WORKERS` | `1` | Parallel transcription workers |
| `TRANSCRIBE_MAX_UPLOAD_MB` | `500` | Max file size |
| `TRANSCRIBE_API_KEY` | *(empty)* | Bearer token (empty = no auth) |
| `TRANSCRIBE_HOST` | `0.0.0.0` | Bind address |
| `TRANSCRIBE_PORT` | `8000` | Bind port |
| `TRANSCRIBE_VAD_FILTER` | `true` | Enable VAD globally |
| `TRANSCRIBE_DEFAULT_LANGUAGE` | *(empty)* | Pre-set language |

---

## Model Selection Guide

| Model | Speed | Accuracy | VRAM |
|-------|-------|----------|------|
| `tiny` | fastest | lowest | ~1 GB |
| `base` | fast | low | ~1 GB |
| `small` | fast | medium | ~2 GB |
| `medium` | medium | good | ~5 GB |
| `large-v3` | slow | best | ~10 GB |
| `large-v3-turbo` | fast | near-best | ~6 GB |
| `distil-large-v3` | very fast | near-best | ~6 GB |

**Recommendation:** `large-v3-turbo` for GPU, `distil-large-v3` or `small` for CPU-only.

---

## Supported File Formats
`.mp3` `.mp4` `.m4a` `.wav` `.flac` `.ogg` `.opus` `.webm` `.mkv` `.avi` `.mov` `.aac` `.wma` `.aiff`

## System Requirement
**ffmpeg** must be available in `PATH` (used internally by faster-whisper for audio decoding).
- Linux: `apt install ffmpeg`
- macOS: `brew install ffmpeg`
- Windows: https://ffmpeg.org/download.html
