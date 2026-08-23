"""
test_agent_runner.py - Unit tests for CourseAgentRunner and ReAct pipeline.
"""

from unittest.mock import MagicMock, patch
import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from src.agent.agent_runner import CourseAgentRunner


class TestCourseAgentRunner:

    @pytest.fixture(autouse=True)
    def setup_mocks(self):
        """Mock ChatOpenAI and LangGraph create_react_agent to prevent live API calls."""
        with patch("src.agent.agent_runner.ChatOpenAI") as self.mock_llm_cls, \
             patch("src.agent.agent_runner.create_react_agent") as self.mock_react_agent_cls, \
             patch("src.agent.agent_runner.get_agent_tools") as self.mock_tools_func:

            self.mock_executor = MagicMock()
            self.mock_react_agent_cls.return_value = self.mock_executor
            self.mock_tools_func.return_value = []

            yield

    def test_extract_sources_from_metadata_aggregation_and_deduplication(self):
        mock_vs = MagicMock()
        # Chunks from same file and pages must be grouped and deduplicated
        mock_vs.last_retrieved_docs = [
            Document(page_content="p1", metadata={"source_file": "Assignment_10.pdf", "page_number": 1}),
            Document(page_content="p2", metadata={"source_file": "Assignment_10.pdf", "page_number": 2}),
            Document(page_content="p1 again", metadata={"source_file": "Assignment_10.pdf", "page_number": 1}),
            Document(page_content="ws", metadata={"source_file": "Workshop_04.pdf", "page_number": 5}),
        ]

        runner = CourseAgentRunner(vector_store=mock_vs)
        sources = runner._extract_sources_from_metadata()

        assert len(sources) == 2
        # Find Assignment_10.pdf
        asg_source = next(s for s in sources if s["file"] == "Assignment_10.pdf")
        assert asg_source["pages"] == "1, 2"

        # Find Workshop_04.pdf
        ws_source = next(s for s in sources if s["file"] == "Workshop_04.pdf")
        assert ws_source["pages"] == "5"

    def test_extract_sources_empty(self):
        mock_vs = MagicMock()
        mock_vs.last_retrieved_docs = []

        runner = CourseAgentRunner(vector_store=mock_vs)
        sources = runner._extract_sources_from_metadata()
        assert sources == []

    def test_process_query_sliding_window_memory(self):
        mock_vs = MagicMock()
        mock_vs.last_retrieved_docs = []
        runner = CourseAgentRunner(vector_store=mock_vs)

        # Mock agent executor response
        self.mock_executor.invoke.return_value = {
            "messages": [AIMessage(content="Final response from ReAct agent")]
        }

        # Create 14 chat turns (7 user + 7 assistant)
        long_chat_history = []
        for i in range(7):
            long_chat_history.append({"role": "user", "content": f"User question {i+1}"})
            long_chat_history.append({"role": "assistant", "content": f"Assistant answer {i+1}"})

        # Process query with max_history_turns=6
        final_text, sources = runner.process_query(
            user_input="Current prompt question",
            chat_history=long_chat_history,
            max_history_turns=6,
        )

        assert final_text == "Final response from ReAct agent"
        assert sources == []

        # Verify that executor received exactly 6 historical turns + 1 current prompt = 7 messages
        passed_messages = self.mock_executor.invoke.call_args[0][0]["messages"]
        assert len(passed_messages) == 7
        assert passed_messages[-1].content == "Current prompt question"
        # First message in window should be turn index 8 (from the last 6 turns of 14)
        assert passed_messages[0].content == "User question 5"