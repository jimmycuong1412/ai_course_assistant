from io import BytesIO

from gtts import gTTS
from langdetect import LangDetectException, detect

SUPPORTED_LANGS = {"en", "vi"}
DEFAULT_LANG = "en"


def detect_language(text: str) -> str:
    """Detects English/Vietnamese from text, falling back to English otherwise."""
    try:
        detected = detect(text)
        if detected in SUPPORTED_LANGS:
            return detected
    except LangDetectException:
        pass
    return DEFAULT_LANG


def text_to_speech(text: str) -> bytes:
    """Converts text to speech (mp3 bytes) via gTTS, auto-detecting English/Vietnamese."""
    lang = detect_language(text)
    buffer = BytesIO()
    gTTS(text=text, lang=lang).write_to_fp(buffer)
    buffer.seek(0)
    return buffer.read()
