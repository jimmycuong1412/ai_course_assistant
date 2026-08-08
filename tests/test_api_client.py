from unittest.mock import patch, MagicMock
import pytest
from openai import RateLimitError
from api_client import make_api_call


@patch("api_client.OpenAI")
def test_make_api_call_structure(mock_openai_cls):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    
    mock_response = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    messages = [{"role": "user", "content": "Hello"}]
    res = make_api_call(
        azure_endpoint="https://fake-endpoint.com",
        api_key="fake-key",
        model_name="gpt-4o-mini",
        messages=messages,
        temperature=0.0
    )

    assert res == mock_response
    mock_openai_cls.assert_called_once()
    
    # Validate headers contain connection controls and anti-caching IDs
    headers = mock_openai_cls.call_args.kwargs["default_headers"]
    assert headers["Connection"] == "close"
    assert "X-Request-ID" in headers


@patch("api_client.OpenAI")
def test_make_api_call_retry_on_rate_limit(mock_openai_cls):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client

    # Simulate hitting RateLimitError twice, succeeding on the third try
    fake_response = MagicMock()
    rate_limit_err = RateLimitError(
        message="Rate limit exceeded",
        response=MagicMock(status_code=429),
        body={}
    )
    mock_client.chat.completions.create.side_effect = [
        rate_limit_err,
        rate_limit_err,
        fake_response
    ]

    res = make_api_call(
        azure_endpoint="https://fake-endpoint.com",
        api_key="fake-key",
        model_name="gpt-4o-mini",
        messages=[{"role": "user", "content": "Test"}]
    )

    assert res == fake_response
    assert mock_client.chat.completions.create.call_count == 3