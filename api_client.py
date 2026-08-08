import uuid
from openai import APIConnectionError, APIError, OpenAI, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential


@retry(
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, APIError)),
    wait=wait_random_exponential(min=1, max=10),
    stop=stop_after_attempt(5),
    reraise=True,
)
def _execute_completion_with_retry(client: OpenAI, **kwargs):
    """Executes OpenAI chat completion wrapped with tenacity exponential backoff retries."""
    return client.chat.completions.create(**kwargs)


def make_api_call(
    azure_endpoint: str,
    api_key: str,
    model_name: str,
    messages: list,
    tools=None,
    stream=False,
    temperature=0.0,
):
    """
    Unified OpenAI API call helper.
    Creates a fresh client instance per request with anti-caching headers and tenacity retries.
    """
    request_id = str(uuid.uuid4())
    client = OpenAI(
        base_url=azure_endpoint,
        api_key=api_key,
        default_headers={
            "Connection": "close",
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "X-Request-ID": request_id,
        },
    )

    kwargs = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "user": request_id,
    }
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    if stream:
        kwargs["stream"] = True

    return _execute_completion_with_retry(client, **kwargs)