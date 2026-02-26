"""
Post-transcription translation using deep-translator (GoogleTranslator).
Translates per-segment so timestamps are preserved in the output.
Runs the synchronous translator in the thread-pool executor so the
async event loop is never blocked.
"""
import asyncio
import logging
from functools import partial

from deep_translator import GoogleTranslator

logger = logging.getLogger(__name__)

_LANG_MAP: dict[str, str] = {
    "en": "english", "de": "german",  "fr": "french",   "es": "spanish",
    "it": "italian", "pt": "portuguese", "nl": "dutch", "pl": "polish",
    "ru": "russian", "zh": "chinese (simplified)", "ja": "japanese",
    "ko": "korean",  "ar": "arabic",   "tr": "turkish",  "he": "hebrew",
    "hi": "hindi",   "sv": "swedish",  "da": "danish",   "fi": "finnish",
    "no": "norwegian", "uk": "ukrainian", "cs": "czech", "ro": "romanian",
    "hu": "hungarian", "id": "indonesian",
}

# Separator that survives Google Translate intact
_SEP = "\n§§§\n"
_CHUNK_SIZE = 4500


def _translate_batch_sync(texts: list[str], lang_code: str) -> list[str]:
    """
    Join all texts with a separator, translate in one call, split back.
    Falls back to individual calls if the separator is mangled.
    """
    target = _LANG_MAP.get(lang_code, lang_code)
    combined = _SEP.join(texts)

    if len(combined) <= _CHUNK_SIZE:
        # Single call for the whole batch
        result = GoogleTranslator(source="auto", target=target).translate(combined) or combined
        parts = result.split(_SEP)
        if len(parts) == len(texts):
            return [p.strip() for p in parts]
        # Separator was lost — fall back to individual calls
        logger.debug("Separator lost during batch translation, falling back to per-segment calls.")

    # Individual calls (used when batch is too large or separator is lost)
    translated = []
    for text in texts:
        try:
            t = GoogleTranslator(source="auto", target=target).translate(text) or text
        except Exception:
            t = text  # keep original on error
        translated.append(t.strip())
    return translated


async def translate_segments(
    segments: list[dict],
    lang_code: str,
) -> list[dict]:
    """
    Translate a list of segment dicts (each has start, end, text).
    Returns new dicts with translated text and original timestamps.
    """
    texts = [s["text"] for s in segments]
    loop = asyncio.get_event_loop()
    translated_texts = await loop.run_in_executor(
        None, partial(_translate_batch_sync, texts, lang_code)
    )
    return [
        {"start": s["start"], "end": s["end"], "text": t}
        for s, t in zip(segments, translated_texts)
    ]
