"""
test_tts_engine.py - Unit tests for Text-to-Speech Engine (gTTS & LangDetect).
"""

from unittest.mock import MagicMock, patch
import pytest
from src.engines.tts_engine import TTSEngine


class TestTTSEngine:

    @pytest.fixture
    def tts_engine(self):
        return TTSEngine()

    def test_clean_markdown(self, tts_engine):
        raw_text = (
            "Here is the solution:\n"
            "```python\nprint('Hello World')\n```\n"
            "Use the `Pinecone` library for **vector storage** and *similarity search*."
        )
        cleaned = tts_engine._clean_markdown(raw_text)

        # Code block and markdown symbols must be stripped
        assert "print('Hello World')" not in cleaned
        assert "`" not in cleaned
        assert "*" not in cleaned
        assert "Use the Pinecone library for vector storage and similarity search." in cleaned

    def test_clean_markdown_empty_or_whitespace(self, tts_engine):
        assert tts_engine._clean_markdown("   \n\t  ") == ""
        assert tts_engine._clean_markdown("```code only```") == ""

    def test_detect_language(self, tts_engine):
        # Vietnamese text detection
        vi_text = "Hệ thống tìm kiếm thông tin bài tập số 10 sử dụng vector database."
        assert tts_engine._detect_language(vi_text) == "vi"

        # English text detection
        en_text = "Pinecone Serverless index with two-stage reranker pipeline."
        assert tts_engine._detect_language(en_text) == "en"

        # Fallback for unrecognizable / short text
        assert tts_engine._detect_language("12345 !!!") == "en"

    @patch("src.engines.tts_engine.gTTS")
    def test_synthesize_success(self, mock_gtts_cls, tts_engine):
        # Mock gTTS write_to_fp to write dummy mp3 bytes
        def fake_write_to_fp(fp):
            fp.write(b"MOCK_MP3_AUDIO_BYTES")

        mock_gtts_instance = MagicMock()
        mock_gtts_instance.write_to_fp.side_effect = fake_write_to_fp
        mock_gtts_cls.return_value = mock_gtts_instance

        audio_bytes = tts_engine.synthesize("Xin chào, đây là trợ lý khóa học AI.")

        assert audio_bytes == b"MOCK_MP3_AUDIO_BYTES"
        mock_gtts_cls.assert_called_once()
        assert mock_gtts_cls.call_args.kwargs["lang"] == "vi"

    def test_synthesize_empty_returns_none(self, tts_engine):
        assert tts_engine.synthesize("") is None
        assert tts_engine.synthesize("```only code block```") is None