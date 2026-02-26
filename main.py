"""
Entry point for the Transcription Service.
Run:  python main.py   or   uvicorn main:app --host 0.0.0.0 --port 8000
"""
import logging

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.core.transcriber import Transcriber
from app.db.database import init_db
from config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)

app = FastAPI(
    title="Transcription Service",
    description="Audio/video transcription powered by faster-whisper.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", include_in_schema=False)
async def serve_ui() -> FileResponse:
    return FileResponse("static/index.html")


@app.on_event("startup")
async def _startup() -> None:
    await init_db()
    Transcriber.get()   # pre-load model


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )
