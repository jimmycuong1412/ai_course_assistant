import io
import re
from typing import Optional
import soundfile as sf
import torch
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.vits.modeling_vits import VitsModel


class TTSEngine:
    """
    Dual-Language Text-to-Speech Engine using Meta MMS-TTS models.
    Supports fluent Vietnamese (mms-tts-vie) and English (mms-tts-eng).
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Preload models and tokenizers for Vietnamese and English
        self.models = {
            "vie": VitsModel.from_pretrained("facebook/mms-tts-vie").to(self.device),
            "eng": VitsModel.from_pretrained("facebook/mms-tts-eng").to(self.device),
        }
        self.tokenizers = {
            "vie": AutoTokenizer.from_pretrained("facebook/mms-tts-vie"),
            "eng": AutoTokenizer.from_pretrained("facebook/mms-tts-eng"),
        }
        self.sampling_rate = self.models["vie"].config.sampling_rate

    def _detect_language(self, text: str) -> str:
        """
        Detects whether the text contains Vietnamese characters/diacritics.
        """
        vietnamese_pattern = re.compile(
            r"[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]",
            re.IGNORECASE,
        )
        return "vie" if vietnamese_pattern.search(text) else "eng"

    def _clean_markdown(self, text: str) -> str:
        """
        Strips Markdown tags, code blocks, and formatting characters for smooth speech synthesis.
        """
        clean = re.sub(r"```[\s\S]*?```", "", text)
        clean = re.sub(r"`.*?`", "", clean)
        clean = re.sub(r"[*_#>\-]", " ", clean)
        return re.sub(r"\s+", " ", clean).strip()

    def synthesize(self, text: str) -> Optional[bytes]:
        """
        Synthesizes text into spoken WAV audio bytes based on detected language.
        """
        cleaned_text = self._clean_markdown(text)
        if not cleaned_text:
            return None

        # Truncate text length to optimize latency and memory
        cleaned_text = cleaned_text[:500]

        # Route to the appropriate language model
        lang = self._detect_language(cleaned_text)
        tokenizer = self.tokenizers[lang]
        model = self.models[lang]

        inputs = tokenizer(cleaned_text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            waveform = model(**inputs).waveform

        audio_array = waveform.squeeze().cpu().numpy()
        buffer = io.BytesIO()
        sf.write(buffer, audio_array, samplerate=self.sampling_rate, format="WAV")
        buffer.seek(0)
        return buffer.getvalue()