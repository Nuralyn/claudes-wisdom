"""OpenAI LLM provider."""

from __future__ import annotations

import os

from wisdom.llm.provider import LLMProvider
from wisdom.logging_config import get_logger

logger = get_logger("llm.openai")


class OpenAIProvider(LLMProvider):
    """OpenAI provider via the OpenAI SDK."""

    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self._client = None

    @property
    def name(self) -> str:
        return "openai"

    def _get_client(self):
        if self._client is None:
            import openai
            self._client = openai.OpenAI()
        return self._client

    def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> str:
        client = self._get_client()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    def is_available(self) -> bool:
        if not os.environ.get("OPENAI_API_KEY"):
            return False
        try:
            import openai  # noqa: F401
            return True
        except ImportError:
            return False
