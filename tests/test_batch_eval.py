import json
from unittest.mock import patch, MagicMock
from batch_eval import process_single_prompt, run_batch_eval


@patch("batch_eval.make_api_call")
def test_process_single_prompt_direct_answer(mock_make_api_call):
    # Test case where no tool call is triggered
    mock_response = MagicMock()
    mock_response.choices[0].message.tool_calls = None
    mock_response.choices[0].message.content = "Direct Response"
    mock_make_api_call.return_value = mock_response

    mock_search_engine = MagicMock()
    item = {"id": "TC_03", "prompt": "Current weather?"}

    res = process_single_prompt(
        item=item,
        search_engine=mock_search_engine,
        azure_endpoint="ep",
        api_key="key",
        model_name="model"
    )

    assert res["id"] == "TC_03"
    assert res["tool_called"] is False
    assert res["final_answer"] == "Direct Response"
    assert res["status"] == "success"


@patch("batch_eval.make_api_call")
@patch("batch_eval.execute_tool_call")
def test_process_single_prompt_with_tool(mock_execute_tool, mock_make_api_call):
    # Turn 1: Trigger tool execution
    tool_call = MagicMock()
    tool_call.function.name = "search_course_knowledge"
    
    mock_response_1 = MagicMock()
    mock_response_1.choices[0].message.tool_calls = [tool_call]
    
    # Turn 2: Synthesize final answer
    mock_response_2 = MagicMock()
    mock_response_2.choices[0].message.content = "Synthesized Answer"

    mock_make_api_call.side_effect = [mock_response_1, mock_response_2]
    mock_execute_tool.return_value = [{"role": "tool", "content": "Search content"}]

    mock_search_engine = MagicMock()
    item = {"id": "TC_01", "prompt": "Assignment 4 tenacity?"}

    res = process_single_prompt(
        item=item,
        search_engine=mock_search_engine,
        azure_endpoint="ep",
        api_key="key",
        model_name="model"
    )

    assert res["id"] == "TC_01"
    assert res["tool_called"] is True
    assert res["final_answer"] == "Synthesized Answer"
    assert mock_make_api_call.call_count == 2