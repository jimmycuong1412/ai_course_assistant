import json
from unittest.mock import MagicMock
from src.agent.tools import execute_tool_call


def test_execute_tool_call_success():
    # Mock OpenAI Tool Call Object
    tool_call = MagicMock()
    tool_call.id = "call_abc123"
    tool_call.function.name = "search_course_knowledge"
    tool_call.function.arguments = json.dumps({
        "query": "Assignment 4 tenacity",
        "category": "assignments"
    })

    # Mock Search Engine behavior
    mock_search_engine = MagicMock()
    mock_search_engine.search.return_value = [{"content": "Result 1"}]
    mock_search_engine.format_search_results.return_value = "Formatted Result 1"

    response = execute_tool_call(tool_call, mock_search_engine)

    assert len(response) == 1
    assert response[0]["role"] == "tool"
    assert response[0]["tool_call_id"] == "call_abc123"
    assert response[0]["content"] == "Formatted Result 1"
    mock_search_engine.search.assert_called_once_with(
        query="Assignment 4 tenacity", category="assignments", top_k=4
    )


def test_execute_tool_call_unknown_function():
    tool_call = MagicMock()
    tool_call.id = "call_xyz789"
    tool_call.function.name = "unknown_function"
    tool_call.function.arguments = "{}"

    mock_search_engine = MagicMock()
    response = execute_tool_call(tool_call, mock_search_engine)

    assert len(response) == 1
    assert "Error: Tool 'unknown_function' is not recognized." in response[0]["content"]