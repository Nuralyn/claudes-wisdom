"""Anthropic (Claude) LLM provider."""

from __future__ import annotations

import os

from wisdom.llm.provider import LLMProvider
from wisdom.logging_config import get_logger

logger = get_logger("llm.anthropic")


class AnthropicProvider(LLMProvider):
    """Claude provider via the Anthropic SDK."""

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self.model = model
        self._client = None

    @property
    def name(self) -> str:
        return "anthropic"

    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic()
        return self._client

    def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> str:
        client = self._get_client()
        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        response = client.messages.create(**kwargs)
        return response.content[0].text

    def is_available(self) -> bool:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return False
        try:
            import anthropic  # noqa: F401
            return True
        except ImportError:
            return False
