"""
test_vector_store.py - Unit tests for CourseVectorStore and Pinecone Inference Reranking.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from langchain_core.documents import Document
from src.rag.vector_store import CourseVectorStore


class TestCourseVectorStore:

    @pytest.fixture(autouse=True)
    def setup_mocks(self):
        """Mock Pinecone and OpenAI environments to prevent live API calls."""
        with patch("src.rag.vector_store.Pinecone") as self.mock_pinecone_cls, \
             patch("src.rag.vector_store.OpenAIEmbeddings") as self.mock_embeddings_cls, \
             patch("src.rag.vector_store.PineconeVectorStore") as self.mock_vector_store_cls:

            self.mock_pc = MagicMock()
            self.mock_pc.list_indexes.return_value = [{"name": "course-knowledge-index"}]
            self.mock_pc.Index.return_value.describe_index_stats.return_value = {"total_vector_count": 50}
            self.mock_pinecone_cls.return_value = self.mock_pc

            self.mock_vs_instance = MagicMock()
            self.mock_vector_store_cls.return_value = self.mock_vs_instance

            yield

    def test_search_empty_query(self, tmp_path: Path):
        store = CourseVectorStore(resources_dir=tmp_path)
        store.last_retrieved_docs = [MagicMock()]

        results = store.search(query="")
        assert results == []
        assert store.last_retrieved_docs == []

    def test_search_metadata_filter_construction(self, tmp_path: Path, mock_documents):
        store = CourseVectorStore(resources_dir=tmp_path)
        self.mock_vs_instance.similarity_search.return_value = mock_documents

        store.search(query="agents", category="workshops", doc_code="workshop_04", top_k=5, fetch_k=15)

        # Ensure filter dictionary is constructed correctly
        self.mock_vs_instance.similarity_search.assert_called_with(
            query="agents",
            k=15,
            filter={"category": "workshops", "doc_code": "workshop_04"},
        )

    def test_search_fallback_when_filter_returns_empty(self, tmp_path: Path, mock_documents):
        store = CourseVectorStore(resources_dir=tmp_path)

        # First call with filter returns empty list, fallback call returns mock_documents
        self.mock_vs_instance.similarity_search.side_effect = [[], mock_documents]

        results = store.search(query="vector setup", category="assignments", doc_code="assignment_99", top_k=5)

        assert self.mock_vs_instance.similarity_search.call_count == 2
        # Fallback call should be executed with query and k=15 without metadata filter
        assert self.mock_vs_instance.similarity_search.call_args_list[1].kwargs == {
            "query": "vector setup",
            "k": 15,
        }
        assert len(results) == len(mock_documents)

    def test_two_stage_rerank_integration(
        self, tmp_path: Path, mock_documents, mock_pinecone_rerank_response
    ):
        store = CourseVectorStore(resources_dir=tmp_path)
        self.mock_vs_instance.similarity_search.return_value = mock_documents
        self.mock_pc.inference.rerank.return_value = mock_pinecone_rerank_response

        # Execute search requesting top_k=2 from candidate pool fetch_k=3
        results = store.search(query="assignment 10 submission", top_k=2, fetch_k=3)

        # 1. Verify pc.inference.rerank invocation
        self.mock_pc.inference.rerank.assert_called_once()
        rerank_call_args = self.mock_pc.inference.rerank.call_args.kwargs
        assert rerank_call_args["model"] == "bge-reranker-v2-m3"
        assert rerank_call_args["query"] == "assignment 10 submission"
        assert rerank_call_args["top_n"] == 2
        assert len(rerank_call_args["documents"]) == 3

        # 2. Verify reordered results based on mock score
        assert len(results) == 2
        # Reranked index 1 (score 0.9541) should become the first result
        assert results[0].metadata["page_number"] == 2
        assert results[0].metadata["rerank_score"] == 0.9541

        # Reranked index 0 (score 0.8123) should become the second result
        assert results[1].metadata["page_number"] == 1
        assert results[1].metadata["rerank_score"] == 0.8123

        # 3. Verify that last_retrieved_docs is updated
        assert store.last_retrieved_docs == results

    def test_two_stage_rerank_exception_fallback(self, tmp_path: Path, mock_documents):
        store = CourseVectorStore(resources_dir=tmp_path)
        self.mock_vs_instance.similarity_search.return_value = mock_documents
        # Simulate network or API error in pc.inference.rerank
        self.mock_pc.inference.rerank.side_effect = Exception("Pinecone API Timeout")

        results = store.search(query="test error", top_k=2, fetch_k=3)

        # Must gracefully fallback to candidate_docs[:top_k]
        assert len(results) == 2
        assert results[0] == mock_documents[0]
        assert results[1] == mock_documents[1]

    def test_format_search_results(self, tmp_path: Path, mock_documents):
        store = CourseVectorStore(resources_dir=tmp_path)
        mock_documents[0].metadata["rerank_score"] = 0.92

        formatted = store.format_search_results(mock_documents)

        assert "--- DOCUMENT 1 | Rerank Score: 0.92 ---" in formatted
        assert "File: Assignment_10.pdf (Page 1)" in formatted
        assert "Category: ASSIGNMENTS | Code: ASSIGNMENT_10" in formatted
        assert "Vector Database setup guide" in formatted