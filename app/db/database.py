import json
import logging
import sqlite3
from pathlib import Path

import aiosqlite

logger = logging.getLogger(__name__)

DB_PATH = "transcribe_service.db"
UPLOADS_DIR = Path("uploads")


async def init_db() -> None:
    UPLOADS_DIR.mkdir(exist_ok=True)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                filename   TEXT NOT NULL,
                file_path  TEXT,
                file_size  INTEGER,
                model      TEXT,
                lang_req   TEXT,
                task       TEXT DEFAULT 'transcribe',
                status     TEXT DEFAULT 'processing',
                progress   INTEGER DEFAULT 0,
                created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%S', 'now')),
                done_at    TEXT,
                duration   REAL,
                lang_det   TEXT,
                lang_prob  REAL,
                full_text  TEXT,
                segments   TEXT,
                error_msg       TEXT,
                translated_text TEXT,
                translation_lang TEXT
            )
            """
        )
        # Migrations for columns added after initial release
        for col_sql in [
            "ALTER TABLE jobs ADD COLUMN progress INTEGER DEFAULT 0",
            "ALTER TABLE jobs ADD COLUMN translated_text TEXT",
            "ALTER TABLE jobs ADD COLUMN translation_lang TEXT",
            "ALTER TABLE jobs ADD COLUMN translated_segments TEXT",
        ]:
            try:
                await db.execute(col_sql)
            except Exception:
                pass  # column already exists

        # Reset jobs that were in-flight when the server last stopped.
        await db.execute(
            "UPDATE jobs SET status='error', error_msg='Server restarted during processing' "
            "WHERE status='processing'"
        )
        await db.commit()
    logger.info("Database ready at %s", DB_PATH)


# ── Synchronous progress update (called from thread pool) ─────────────────────

def set_progress(job_id: int, pct: int) -> None:
    """Write progress 0-99 synchronously — safe to call from a worker thread."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("UPDATE jobs SET progress=? WHERE id=?", (pct, job_id))
    except Exception:
        pass  # non-critical


# ── Async CRUD ────────────────────────────────────────────────────────────────

async def create_job(
    filename: str,
    file_path,
    file_size,
    model: str,
    lang_req: str,
    task: str,
    translation_lang: str | None = None,
) -> int:
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            "INSERT INTO jobs (filename, file_path, file_size, model, lang_req, task, translation_lang) "
            "VALUES (?,?,?,?,?,?,?)",
            (
                filename,
                str(file_path) if file_path else None,
                file_size,
                model,
                lang_req,
                task,
                translation_lang or None,
            ),
        )
        await db.commit()
        return cur.lastrowid


async def update_job_translation(
    job_id: int,
    translated_text: str,
    translation_lang: str,
    translated_segments: str,
) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE jobs SET translated_text=?, translation_lang=?, translated_segments=? WHERE id=?",
            (translated_text, translation_lang, translated_segments, job_id),
        )
        await db.commit()


async def update_job_done(job_id: int, result) -> None:
    segs = json.dumps([s.model_dump() for s in result.segments])
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            UPDATE jobs SET
                status    = 'done',
                progress  = 100,
                done_at   = strftime('%Y-%m-%dT%H:%M:%S','now'),
                duration  = ?,
                lang_det  = ?,
                lang_prob = ?,
                full_text = ?,
                segments  = ?
            WHERE id = ?
            """,
            (
                result.duration,
                result.language,
                result.language_probability,
                result.text,
                segs,
                job_id,
            ),
        )
        await db.commit()


async def update_job_error(job_id: int, error: Exception) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE jobs SET status='error', done_at=strftime('%Y-%m-%dT%H:%M:%S','now'), error_msg=? WHERE id=?",
            (str(error), job_id),
        )
        await db.commit()


async def list_jobs() -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT id,filename,file_path,file_size,model,lang_req,task,status,progress,"
            "created_at,done_at,duration,lang_det,lang_prob,full_text,error_msg,"
            "translated_text,translation_lang,translated_segments "
            "FROM jobs ORDER BY created_at DESC"
        ) as cur:
            rows = await cur.fetchall()
            return [dict(r) for r in rows]


async def get_job(job_id: int) -> dict | None:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None


async def delete_job(job_id: int) -> bool:
    job = await get_job(job_id)
    if not job:
        return False
    if job.get("file_path"):
        Path(job["file_path"]).unlink(missing_ok=True)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM jobs WHERE id=?", (job_id,))
        await db.commit()
    return True
