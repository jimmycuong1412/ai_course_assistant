from unittest.mock import MagicMock, patch

from langdetect import LangDetectException

from tts import detect_language, text_to_speech


@patch("tts.detect", return_value="vi")
def test_detect_language_returns_supported_code(mock_detect):
    assert detect_language("Xin chào") == "vi"


@patch("tts.detect", return_value="fr")
def test_detect_language_falls_back_to_english_for_unsupported(mock_detect):
    assert detect_language("Bonjour") == "en"


@patch("tts.detect", side_effect=LangDetectException(1, "no features in text"))
def test_detect_language_falls_back_to_english_on_failure(mock_detect):
    assert detect_language("123") == "en"


@patch("tts.gTTS")
@patch("tts.detect", return_value="vi")
def test_text_to_speech_uses_detected_supported_language(mock_detect, mock_gtts_cls):
    mock_gtts_instance = MagicMock()
    mock_gtts_instance.write_to_fp.side_effect = lambda buf: buf.write(b"fake-mp3-bytes")
    mock_gtts_cls.return_value = mock_gtts_instance

    result = text_to_speech("Xin chào")

    mock_gtts_cls.assert_called_once_with(text="Xin chào", lang="vi")
    assert result == b"fake-mp3-bytes"


@patch("tts.gTTS")
@patch("tts.detect", return_value="fr")
def test_text_to_speech_falls_back_to_english_for_unsupported_language(mock_detect, mock_gtts_cls):
    mock_gtts_instance = MagicMock()
    mock_gtts_instance.write_to_fp.side_effect = lambda buf: buf.write(b"fake-mp3-bytes")
    mock_gtts_cls.return_value = mock_gtts_instance

    text_to_speech("Bonjour")

    mock_gtts_cls.assert_called_once_with(text="Bonjour", lang="en")


@patch("tts.gTTS")
@patch("tts.detect", side_effect=LangDetectException(1, "no features in text"))
def test_text_to_speech_falls_back_to_english_on_detection_failure(mock_detect, mock_gtts_cls):
    mock_gtts_instance = MagicMock()
    mock_gtts_instance.write_to_fp.side_effect = lambda buf: buf.write(b"fake-mp3-bytes")
    mock_gtts_cls.return_value = mock_gtts_instance

    text_to_speech("123")

    mock_gtts_cls.assert_called_once_with(text="123", lang="en")
