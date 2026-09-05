import re
from io import BytesIO
from typing import Optional
from gtts import gTTS
from langdetect import LangDetectException, detect


class TTSEngine:
    """
    Text-to-Speech Engine using gTTS and langdetect.
    Supports English and Vietnamese with automatic language detection.
    """

    def __init__(self):
        self.supported_langs = {"en", "vi"}
        self.default_lang = "en"

    def _detect_language(self, text: str) -> str:
        """
        Detects English/Vietnamese from text, falling back to default language.
        """
        try:
            detected = detect(text)
            if detected in self.supported_langs:
                return detected
        except LangDetectException:
            pass
        return self.default_lang

    def _clean_markdown(self, text: str) -> str:
        """
        Strips Markdown tags, code blocks, and formatting characters for clean speech synthesis.
        """
        # 1. Remove multi-line code blocks
        clean = re.sub(r"```[\s\S]*?```", "", text)
        # 2. Strip markdown symbols directly
        clean = re.sub(r"[*_`~#>]", "", clean)
        # 3. Replace list bullets with spaces
        clean = re.sub(r"\s*[-+]\s+", " ", clean)
        # 4. Collapse multiple whitespaces
        clean = re.sub(r"\s+", " ", clean).strip()
        # 5. Remove any accidental whitespace before punctuation marks
        clean = re.sub(r"\s+([.,!?;:])", r"\1", clean)
        return clean

    def synthesize(self, text: str) -> Optional[bytes]:
        """
        Converts text to MP3 audio bytes using gTTS with auto language detection.
        """
        cleaned_text = self._clean_markdown(text)
        if not cleaned_text:
            return None

        # Truncate text length to optimize latency
        cleaned_text = cleaned_text[:1000]

        lang = self._detect_language(cleaned_text)
        buffer = BytesIO()
        gTTS(text=cleaned_text, lang=lang).write_to_fp(buffer)
        buffer.seek(0)
        return buffer.read()