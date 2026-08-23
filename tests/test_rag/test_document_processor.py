"""
test_document_processor.py - Unit tests for CourseDocumentProcessor.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from langchain_core.documents import Document
from src.rag.document_processor import CourseDocumentProcessor


class TestCourseDocumentProcessor:

    def test_determine_category(self, mock_resources_dir: Path):
        processor = CourseDocumentProcessor(mock_resources_dir, enable_image_analysis=False)

        # Guidelines priority check
        assert processor._determine_category(Path("guidelines/guide_asg_10.pdf")) == "guidelines"
        assert processor._determine_category(Path("assignments/guideline_submission.pdf")) == "guidelines"

        # Workshops check
        assert processor._determine_category(Path("workshops/workshop_04.pdf")) == "workshops"

        # Assignments check
        assert processor._determine_category(Path("assignments/assignment_10.pdf")) == "assignments"

        # General/Other check
        assert processor._determine_category(Path("misc/syllabus.pdf")) == "general"

    def test_extract_doc_code(self, mock_resources_dir: Path):
        processor = CourseDocumentProcessor(mock_resources_dir, enable_image_analysis=False)

        assert processor._extract_doc_code("Assignment 10 - Spec.pdf") == "assignment_10"
        assert processor._extract_doc_code("assignment_03_v2.pdf") == "assignment_03"
        assert processor._extract_doc_code("Workshop 4 Agents.pdf") == "workshop_04"
        assert processor._extract_doc_code("Workshop_11.pdf") == "workshop_11"
        assert processor._extract_doc_code("General_Guideline.pdf") == "other"

    def test_load_and_split_empty_or_missing_directory(self, tmp_path: Path):
        # Non-existent directory
        processor_missing = CourseDocumentProcessor(tmp_path / "non_existent", enable_image_analysis=False)
        assert processor_missing.load_and_split_documents() == []

        # Empty directory
        empty_dir = tmp_path / "empty_res"
        empty_dir.mkdir()
        processor_empty = CourseDocumentProcessor(empty_dir, enable_image_analysis=False)
        assert processor_empty.load_and_split_documents() == []

    @patch("fitz.open")
    def test_load_and_split_documents_with_post_split_headers(
        self, mock_fitz_open, mock_resources_dir: Path
    ):
        # Create a dummy PDF file in assignments directory
        pdf_file = mock_resources_dir / "assignments" / "Assignment 10.pdf"
        pdf_file.write_text("dummy")

        # Mock PyMuPDF Document & Page
        mock_page = MagicMock()
        mock_page.get_text.return_value = (
            "This is a comprehensive overview of Assignment 10. Students must configure "
            "Pinecone Serverless vector database and implement LangGraph ReAct agent pipeline."
        )
        mock_page.get_images.return_value = []

        mock_doc = MagicMock()
        mock_doc.__iter__.return_value = [mock_page]
        mock_doc.__len__.return_value = 1
        mock_fitz_open.return_value = mock_doc

        processor = CourseDocumentProcessor(
            mock_resources_dir,
            chunk_size=300,
            chunk_overlap=50,
            enable_image_analysis=False,
        )

        chunks = processor.load_and_split_documents()

        assert len(chunks) > 0
        first_chunk = chunks[0]

        # Verify metadata
        assert first_chunk.metadata["category"] == "assignments"
        assert first_chunk.metadata["doc_code"] == "assignment_10"
        assert first_chunk.metadata["source_file"] == "Assignment 10.pdf"
        assert first_chunk.metadata["page_number"] == 1
        assert "chunk_id" in first_chunk.metadata

        # Verify post-split context header formatting
        expected_header = "Document: Assignment 10.pdf | Category: ASSIGNMENTS | Code: ASSIGNMENT_10\n"
        assert first_chunk.page_content.startswith(expected_header)