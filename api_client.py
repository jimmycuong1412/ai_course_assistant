import os
from typing import Any, Dict, List, Optional
from openai import APIConnectionError, APIError, OpenAI, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential


class APIClient:
    """
    Unified OpenAI API client wrapper.
    Initializes the OpenAI client instance once using environment variables
    and provides retry-wrapped methods for completions and embeddings.
    """

    def __init__(self):
        """
        Reads credentials directly from environment variables.
        """
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
        self.base_url = os.getenv("AZURE_OPENAI_ENDPOINT", None)
        self.default_chat_model = os.getenv("AZURE_OPENAI_MODEL", "gpt-4o-mini")
        self.default_embed_model = "text-embedding-3-small"

        client_kwargs = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url

        self.client = OpenAI(**client_kwargs)

    @retry(
        retry=retry_if_exception_type((RateLimitError, APIConnectionError, APIError)),
        wait=wait_random_exponential(min=1, max=10),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def make_api_call(
        self,
        messages: List[Dict[str, Any]],
        model_name: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
        temperature: float = 0.0,
    ):
        """
        Executes chat completion with retry logic and tool calling support.
        """
        kwargs = {
            "model": model_name or self.default_chat_model,
            "messages": messages,
            "temperature": temperature,
        }

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if stream:
            kwargs["stream"] = True

        return self.client.chat.completions.create(**kwargs)

    @retry(
        retry=retry_if_exception_type((RateLimitError, APIConnectionError, APIError)),
        wait=wait_random_exponential(min=1, max=10),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def create_embedding(
        self, text: str, model_name: Optional[str] = None
    ) -> List[float]:
        """
        Generates an embedding vector using a dedicated embedding model (default: text-embedding-3-small).
        """
        clean_text = text.replace("\n", " ").strip()[:8000]
        response = self.client.embeddings.create(
            input=[clean_text],
            model=model_name or self.default_embed_model,
        )
        return response.data[0].embedding

    @retry(
        retry=retry_if_exception_type((RateLimitError, APIConnectionError, APIError)),
        wait=wait_random_exponential(min=1, max=10),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def create_embeddings_batch(
        self, texts: List[str], model_name: Optional[str] = None
    ) -> List[List[float]]:
        """
        Generates embedding vectors for a batch of texts using a dedicated embedding model.
        """
        clean_texts = [t.replace("\n", " ").strip()[:8000] for t in texts]
        response = self.client.embeddings.create(
            input=clean_texts,
            model=model_name or self.default_embed_model,
        )
        return [item.embedding for item in response.data]