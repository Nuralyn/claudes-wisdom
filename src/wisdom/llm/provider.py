"""LLM provider abstraction and registry."""

from __future__ import annotations

from abc import ABC, abstractmethod

from wisdom.config import LLMConfig
from wisdom.exceptions import ProviderError
from wisdom.logging_config import get_logger

logger = get_logger("llm.provider")


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name identifier."""
        ...

    @abstractmethod
    def generate(self, prompt: str, system: str = "", temperature: float = 0.3, max_tokens: int = 4096) -> str:
        """Generate a text completion."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this provider is configured and available."""
        ...


class ProviderRegistry:
    """Registry of LLM providers with fallback chain."""

    def __init__(self):
        self._providers: dict[str, LLMProvider] = {}
        self._default: str | None = None

    def register(self, provider: LLMProvider) -> None:
        self._providers[provider.name] = provider
        logger.info("Registered LLM provider: %s", provider.name)

    def set_default(self, name: str) -> None:
        if name not in self._providers:
            raise ProviderError(f"Provider '{name}' not registered")
        self._default = name

    def get(self, name: str | None = None) -> LLMProvider:
        """Get a provider by name, or the default, or first available."""
        if name and name in self._providers:
            return self._providers[name]
        if self._default and self._default in self._providers:
            return self._providers[self._default]
        # Try any available
        for p in self._providers.values():
            if p.is_available():
                return p
        raise ProviderError("No LLM provider available")

    def list_available(self) -> list[str]:
        return [n for n, p in self._providers.items() if p.is_available()]

    @property
    def has_provider(self) -> bool:
        return any(p.is_available() for p in self._providers.values())

    def auto_register(self, config: LLMConfig) -> None:
        """Attempt to register all known providers."""
        # Try Anthropic
        try:
            from wisdom.llm.providers.anthropic import AnthropicProvider
            provider = AnthropicProvider(model=config.anthropic_model)
            if provider.is_available():
                self.register(provider)
        except Exception:
            pass

        # Try OpenAI
        try:
            from wisdom.llm.providers.openai_provider import OpenAIProvider
            provider = OpenAIProvider(model=config.openai_model)
            if provider.is_available():
                self.register(provider)
        except Exception:
            pass

        # Try Ollama
        try:
            from wisdom.llm.providers.ollama import OllamaProvider
            provider = OllamaProvider(model=config.ollama_model)
            if provider.is_available():
                self.register(provider)
        except Exception:
            pass

        if config.default_provider in self._providers:
            self.set_default(config.default_provider)
        elif self._providers:
            self.set_default(next(iter(self._providers)))
