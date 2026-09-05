"""
test_stt_engine.py - Unit tests for STTEngine.

Covers:
- transcribe: happy path (mono audio)
- transcribe: stereo → mono averaging
- transcribe: strips leading/trailing whitespace from result
- _get_pipeline: lazy loading and per-model caching
- _get_pipeline: unknown language falls back to Vietnamese model
"""

from io import BytesIO
from unittest.mock import MagicMock, patch, call
import numpy as np
import pytest
import soundfile as sf


def _make_wav_bytes(n_samples: int = 16000, channels: int = 1, samplerate: int = 16000) -> bytes:
    """Create a minimal valid WAV file in memory."""
    if channels == 1:
        data = np.zeros(n_samples, dtype="float32")
    else:
        data = np.zeros((n_samples, channels), dtype="float32")
    buf = BytesIO()
    sf.write(buf, data, samplerate, format="WAV", subtype="FLOAT")
    buf.seek(0)
    return buf.read()


@pytest.fixture
def stt():
    """Return an STTEngine with the transformers pipeline stubbed out."""
    with patch("src.engines.stt_engine.pipeline") as mock_pipeline_factory:
        mock_asr = MagicMock(return_value={"text": " hello world "})
        mock_pipeline_factory.return_value = mock_asr

        from src.engines.stt_engine import STTEngine
        engine = STTEngine()
        engine._mock_pipeline_factory = mock_pipeline_factory
        engine._mock_asr = mock_asr
        yield engine


class TestSTTEngineTranscribe:

    def test_transcribe_mono_returns_stripped_text(self, stt):
        audio = _make_wav_bytes(channels=1)
        result = stt.transcribe(audio, language="vi")
        assert result == "hello world"

    def test_transcribe_strips_whitespace(self, stt):
        stt._mock_asr.return_value = {"text": "  xin chào  "}
        audio = _make_wav_bytes(channels=1)
        result = stt.transcribe(audio, language="vi")
        assert result == "xin chào"

    def test_transcribe_stereo_averaged_to_mono(self, stt):
        """Stereo audio should be averaged across channels before passing to ASR."""
        audio = _make_wav_bytes(n_samples=8000, channels=2)

        captured = {}

        def _capture_call(payload):
            captured["array"] = payload["array"]
            return {"text": "stereo ok"}

        stt._mock_asr.side_effect = _capture_call
        result = stt.transcribe(audio, language="en")

        assert result == "stereo ok"
        # After mono-conversion the array must be 1-D
        assert captured["array"].ndim == 1
        assert captured["array"].shape[0] == 8000

    def test_transcribe_passes_correct_sampling_rate(self, stt):
        """sampling_rate in the ASR payload must match the WAV file's rate."""
        audio = _make_wav_bytes(samplerate=8000, channels=1)

        captured = {}

        def _capture_call(payload):
            captured["sampling_rate"] = payload["sampling_rate"]
            return {"text": "rate check"}

        stt._mock_asr.side_effect = _capture_call
        stt.transcribe(audio, language="vi")
        assert captured["sampling_rate"] == 8000

    def test_transcribe_english_uses_en_pipeline(self, stt):
        """language='en' must load the English Whisper model, not PhoWhisper."""
        from src.engines.stt_engine import MODELS_BY_LANGUAGE
        audio = _make_wav_bytes(channels=1)
        stt.transcribe(audio, language="en")

        loaded_model = stt._mock_pipeline_factory.call_args[1]["model"] \
            if stt._mock_pipeline_factory.call_args[1] \
            else stt._mock_pipeline_factory.call_args[0][1]

        assert loaded_model == MODELS_BY_LANGUAGE["en"]

    def test_transcribe_vietnamese_uses_vi_pipeline(self, stt):
        from src.engines.stt_engine import MODELS_BY_LANGUAGE
        audio = _make_wav_bytes(channels=1)
        stt.transcribe(audio, language="vi")

        # factory was called once with the Vietnamese model
        factory_call = stt._mock_pipeline_factory.call_args
        model_arg = (
            factory_call[1].get("model") or factory_call[0][1]
            if factory_call[0] else factory_call[1].get("model")
        )
        assert model_arg == MODELS_BY_LANGUAGE["vi"]


class TestSTTEnginePipelineCaching:

    def test_pipeline_loaded_only_once_per_model(self):
        """Calling transcribe twice with the same language must load the pipeline only once."""
        with patch("src.engines.stt_engine.pipeline") as mock_factory:
            mock_factory.return_value = MagicMock(return_value={"text": "ok"})
            from importlib import import_module, reload
            import src.engines.stt_engine as mod
            engine = mod.STTEngine()

            audio = _make_wav_bytes(channels=1)
            engine.transcribe(audio, language="vi")
            engine.transcribe(audio, language="vi")

            # pipeline factory called exactly once for the same language
            assert mock_factory.call_count == 1

    def test_different_languages_load_separate_pipelines(self):
        """'vi' and 'en' must each get their own pipeline instance."""
        with patch("src.engines.stt_engine.pipeline") as mock_factory:
            mock_factory.return_value = MagicMock(return_value={"text": "ok"})
            from importlib import import_module
            import src.engines.stt_engine as mod
            engine = mod.STTEngine()

            audio = _make_wav_bytes(channels=1)
            engine.transcribe(audio, language="vi")
            engine.transcribe(audio, language="en")

            assert mock_factory.call_count == 2

    def test_unknown_language_falls_back_to_vi_model(self):
        """An unsupported language code must use the Vietnamese (default) model."""
        with patch("src.engines.stt_engine.pipeline") as mock_factory:
            mock_factory.return_value = MagicMock(return_value={"text": "ok"})
            import src.engines.stt_engine as mod
            from src.engines.stt_engine import MODELS_BY_LANGUAGE
            engine = mod.STTEngine()

            audio = _make_wav_bytes(channels=1)
            engine.transcribe(audio, language="jp")  # unsupported

            loaded_model = mock_factory.call_args[0][1] \
                if mock_factory.call_args[0] and len(mock_factory.call_args[0]) > 1 \
                else mock_factory.call_args[1].get("model")
            assert loaded_model == MODELS_BY_LANGUAGE["vi"]
