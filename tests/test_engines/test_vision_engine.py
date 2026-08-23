"""
test_vision_engine.py - Unit tests for Multimodal Vision Engine.
"""

from unittest.mock import MagicMock, patch
import pytest
from langchain_core.messages import AIMessage
from src.engines.vision_engine import VisionAnalysisResponse, VisionEngine


class TestVisionEngine:

    @pytest.fixture(autouse=True)
    def setup_mocks(self):
        """Mock ChatOpenAI to avoid actual OpenAI API calls."""
        with patch("src.engines.vision_engine.ChatOpenAI") as self.mock_chat_cls:
            self.mock_llm_instance = MagicMock()
            self.mock_chat_cls.return_value = self.mock_llm_instance
            yield

    def test_analyze_image_bytes_structured_success(self):
        # Mock Structured LLM Response
        mock_structured_llm = MagicMock()
        mock_structured_llm.invoke.return_value = VisionAnalysisResponse(
            summary="Python terminal error traceback screenshot",
            extracted_text_or_error="IndexError: list index out of range at line 42",
            detected_issue="Attempted to access an empty candidate documents list",
        )
        self.mock_llm_instance.with_structured_output.return_value = mock_structured_llm

        engine = VisionEngine()
        dummy_image_bytes = b"FAKE_PNG_BINARY_IMAGE_DATA"

        result_context = engine.analyze_image_bytes(
            image_bytes=dummy_image_bytes,
            user_note="Why is my script throwing this exception?",
        )

        assert "[Uploaded Image Analysis]" in result_context
        assert "Visual Summary: Python terminal error traceback screenshot" in result_context
        assert "Extracted Text/Error: IndexError: list index out of range" in result_context
        assert "Inferred Issue: Attempted to access an empty candidate documents list" in result_context

    def test_analyze_image_bytes_api_failure_fallback(self):
        mock_structured_llm = MagicMock()
        mock_structured_llm.invoke.side_effect = Exception("OpenAI Vision RateLimit Exceeded")
        self.mock_llm_instance.with_structured_output.return_value = mock_structured_llm

        engine = VisionEngine()
        result_context = engine.analyze_image_bytes(image_bytes=b"DUMMY_BYTES")

        assert "[Uploaded Image Analysis Failed]: OpenAI Vision RateLimit Exceeded" in result_context

    def test_describe_document_image_success(self):
        # Mock unstructured LLM response for PDF OCR descriptions
        self.mock_llm_instance.invoke.return_value = AIMessage(
            content="Screenshot shows Pinecone Web Console: Serverless index creation dialog with dimension 1536 and cosine metric."
        )

        engine = VisionEngine()
        description = engine.describe_document_image(image_bytes=b"FAKE_PDF_SCREENSHOT_BYTES")

        assert "Pinecone Web Console" in description
        assert "dimension 1536" in description

    def test_describe_document_image_failure_returns_empty_string(self):
        self.mock_llm_instance.invoke.side_effect = Exception("Vision connection dropped")

        engine = VisionEngine()
        description = engine.describe_document_image(image_bytes=b"FAKE_PDF_SCREENSHOT_BYTES")

        assert description == ""