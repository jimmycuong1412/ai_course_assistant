"""
conftest.py - Shared pytest fixtures and mock objects for test suites.
"""

from pathlib import Path
from unittest.mock import MagicMock
import pytest
from langchain_core.documents import Document


@pytest.fixture
def mock_resources_dir(tmp_path: Path) -> Path:
    """Creates a temporary mock directory structure matching course documents."""
    res_dir = tmp_path / "resources"
    res_dir.mkdir()

    # Create subdirectories
    (res_dir / "assignments").mkdir()
    (res_dir / "workshops").mkdir()
    (res_dir / "guidelines").mkdir()

    return res_dir


@pytest.fixture
def mock_documents() -> list[Document]:
    """Sample LangChain Document chunks representing retrieved vector store items."""
    return [
        Document(
            page_content="Vector Database setup guide for assignment 10 with Pinecone.",
            metadata={
                "source_file": "Assignment_10.pdf",
                "page_number": 1,
                "category": "assignments",
                "doc_code": "assignment_10",
            },
        ),
        Document(
            page_content="Detailed rubric and submission requirements for Assignment 10.",
            metadata={
                "source_file": "Assignment_10.pdf",
                "page_number": 2,
                "category": "assignments",
                "doc_code": "assignment_10",
            },
        ),
        Document(
            page_content="Workshop 04 LangGraph ReAct agent architecture and state graphs.",
            metadata={
                "source_file": "Workshop_04.pdf",
                "page_number": 5,
                "category": "workshops",
                "doc_code": "workshop_04",
            },
        ),
    ]


@pytest.fixture
def mock_pinecone_rerank_response():
    """Mocks the response structure returned by pc.inference.rerank."""
    class RerankItem:
        def __init__(self, index: int, score: float):
            self.index = index
            self.score = score

    class RerankResult:
        def __init__(self):
            # Returns items ranked with highest score first (index 1 then index 0)
            self.data = [
                RerankItem(index=1, score=0.9541),
                RerankItem(index=0, score=0.8123),
            ]

    return RerankResult()