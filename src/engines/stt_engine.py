"""
stt_engine.py - Speech-to-Text with per-language models: VinAI's PhoWhisper for Vietnamese
(Whisper fine-tuned for Vietnamese) and a standard English Whisper checkpoint for English.
"""

import os
from io import BytesIO

import numpy as np
import soundfile as sf
from transformers import pipeline

MODELS_BY_LANGUAGE = {
    "vi": os.getenv("PHOWHISPER_MODEL", "vinai/PhoWhisper-base"),
    "en": os.getenv("WHISPER_EN_MODEL", "openai/whisper-base.en"),
}


class STTEngine:
    """
    Transcribes recorded voice audio to text, lazily loading one ASR pipeline per language.
    """

    def __init__(self):
        self._pipelines = {}

    def _get_pipeline(self, language: str):
        model_name = MODELS_BY_LANGUAGE.get(language, MODELS_BY_LANGUAGE["vi"])
        if model_name not in self._pipelines:
            self._pipelines[model_name] = pipeline("automatic-speech-recognition", model=model_name)
        return self._pipelines[model_name]

    def transcribe(self, audio_bytes: bytes, language: str = "vi") -> str:
        """
        Decodes WAV audio bytes (as produced by st.chat_input's mic recorder) and returns the
        transcribed text using the ASR model for the given language ("vi" or "en").
        """
        audio_array, sample_rate = sf.read(BytesIO(audio_bytes), dtype="float32")
        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=1)

        asr = self._get_pipeline(language)
        result = asr({"array": np.asarray(audio_array), "sampling_rate": sample_rate})
        return result["text"].strip()
