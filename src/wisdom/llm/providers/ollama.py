"""Ollama (local) LLM provider."""

from __future__ import annotations

from wisdom.llm.provider import LLMProvider
from wisdom.logging_config import get_logger

logger = get_logger("llm.ollama")


class OllamaProvider(LLMProvider):
    """Local LLM provider via Ollama."""

    def __init__(self, model: str = "llama3.2", host: str | None = None):
        self.model = model
        self.host = host
        self._client = None

    @property
    def name(self) -> str:
        return "ollama"

    def _get_client(self):
        if self._client is None:
            import ollama
            if self.host:
                self._client = ollama.Client(host=self.host)
            else:
                self._client = ollama.Client()
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
        response = client.chat(
            model=self.model,
            messages=messages,
            options={"temperature": temperature, "num_predict": max_tokens},
        )
        return response["message"]["content"]

    def is_available(self) -> bool:
        try:
            import ollama
            client = ollama.Client(host=self.host) if self.host else ollama.Client()
            client.list()
            return True
        except Exception:
            return False
