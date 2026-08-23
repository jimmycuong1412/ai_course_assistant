"""
test_tools.py - Unit tests for LangChain Agent Tools.
"""

import os
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from src.agent.tools import (
    create_course_search_tool,
    create_tavily_search_tool,
    get_agent_tools,
)


class TestAgentTools:

    def test_course_search_tool_execution(self, mock_documents):
        mock_vs = MagicMock()
        mock_vs.search.return_value = mock_documents
        mock_vs.format_search_results.return_value = "--- DOCUMENT 1 ---\nFile: Assignment_10.pdf\nContent: Test content"

        # Create tool bound to mock vector store
        search_tool = create_course_search_tool(mock_vs)

        assert search_tool.name == "search_course_knowledge"
        
        # Invoke tool with specific parameters
        result_text = search_tool.invoke({
            "query": "Pinecone database setup",
            "category": "assignments",
            "doc_code": "assignment_10",
        })

        mock_vs.search.assert_called_once_with(
            query="Pinecone database setup",
            category="assignments",
            doc_code="assignment_10",
            top_k=5,
        )
        mock_vs.format_search_results.assert_called_once_with(mock_documents)
        assert "Assignment_10.pdf" in result_text

    @patch.dict(os.environ, {"TAVILY_API_KEY": ""}, clear=True)
    def test_tavily_tool_disabled_when_api_key_missing(self):
        tool = create_tavily_search_tool()
        assert tool is None

    @patch.dict(os.environ, {"TAVILY_API_KEY": "tvly-mock-key-12345"})
    def test_tavily_tool_enabled_when_api_key_present(self):
        tool = create_tavily_search_tool()
        assert tool is not None
        assert tool.name == "tavily_search"

    @patch.dict(os.environ, {"TAVILY_API_KEY": "tvly-mock-key-12345"})
    def test_get_agent_tools_aggregates_all_tools(self):
        mock_vs = MagicMock()
        tools = get_agent_tools(mock_vs)

        assert len(tools) == 2
        tool_names = [t.name for t in tools]
        assert "search_course_knowledge" in tool_names
        assert "tavily_search" in tool_names