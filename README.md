# Transcribe Service

A self-hosted audio transcription service powered by [faster-whisper](https://github.com/SYSTRAN/faster-whisper), with a built-in web UI and optional post-transcription translation.

## Features

- Transcribe audio/video files or URLs via REST API
- Per-job progress tracking
- Post-transcription translation to 25+ languages (Google Translate, no API key needed)
- Web UI — upload files, browse jobs, read transcripts with timestamps
- Async job queue (jobs run in the background, UI polls for updates)
- Hot-swap Whisper models and CPU/GPU device without restarting
- Docker support

## Supported formats

`mp3`, `mp4`, `m4a`, `wav`, `flac`, `ogg`, `opus`, `webm`, `weba`, `mkv`, `avi`, `mov`, `aac`, `wma`, `aiff`

## Quick start

### Native (Windows / Linux / macOS)

```bash
git clone <repo-url>
cd transcribe_service
python start.py        # installs missing deps, starts server, opens browser
```

`start.py` will:
1. Detect and add NVIDIA CUDA DLL paths (Windows)
2. Install any missing packages from `requirements.txt`
3. Start the server on `http://localhost:8000`
4. Open the browser automatically

### Manual

```bash
pip install -r requirements.txt
python main.py
```

Open `http://localhost:8000` in your browser.

### Docker

```bash
cp .env.example .env          # edit as needed
docker compose up --build
```

## Configuration

Copy `.env.example` to `.env` and edit:

| Variable | Default | Description |
|----------|---------|-------------|
| `TRANSCRIBE_MODEL_SIZE` | `large-v3-turbo` | Whisper model to load |
| `TRANSCRIBE_DEVICE` | `auto` | `auto`, `cpu`, or `cuda` |
| `TRANSCRIBE_COMPUTE_TYPE` | `auto` | `auto`, `int8`, `float16`, `int8_float16` |
| `TRANSCRIBE_API_KEY` | _(none)_ | If set, all API calls require `Authorization: Bearer <key>` |
| `TRANSCRIBE_MAX_UPLOAD_MB` | `500` | Upload size limit |
| `TRANSCRIBE_CPU_THREADS` | `4` | CPU threads for inference |

## API

Base URL: `http://localhost:8000/api/v1`

Interactive docs: `http://localhost:8000/docs`

### Async jobs (recommended)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/jobs/file` | Upload a file, returns `job_id` immediately |
| `POST` | `/jobs/url` | Queue a URL, returns `job_id` immediately |
| `GET` | `/jobs` | List all jobs |
| `GET` | `/jobs/{id}` | Get job detail (status, transcript, segments) |
| `DELETE` | `/jobs/{id}` | Delete job + audio file |
| `GET` | `/audio/{id}` | Stream the original audio |

**Upload example:**
```bash
curl -X POST http://localhost:8000/api/v1/jobs/file \
  -F "file=@recording.mp3" \
  -F "language=en" \
  -F "translation_lang=he"
# → {"job_id": 1, "status": "processing"}

curl http://localhost:8000/api/v1/jobs/1
# → {"status": "done", "full_text": "...", "translated_text": "...", ...}
```

### Sync endpoints (blocking)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/transcribe/file` | Upload + transcribe, returns result directly |
| `POST` | `/transcribe/url` | Download + transcribe, returns result directly |

### Utility

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Service status |
| `GET` | `/models` | List Whisper models (marks locally cached ones) |
| `GET` | `/config` | Current model/device config |
| `PATCH` | `/config` | Hot-swap model or device |

## Models

Default model is `large-v3-turbo` (best speed/accuracy trade-off). Models are downloaded automatically on first use from Hugging Face and cached locally.

Available models: `tiny`, `base`, `small`, `medium`, `large-v1`, `large-v2`, `large-v3`, `large-v3-turbo`, `distil-large-v2`, `distil-large-v3`, `distil-medium.en`, `distil-small.en`

## GPU support (Windows)

If CUDA is not detected automatically, install the NVIDIA packages:

```bash
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
```

Then launch via `start.py` which adds the DLL paths to `PATH` before loading the model.
