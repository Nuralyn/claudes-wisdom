"""Custom exceptions for the Wisdom System."""


class WisdomError(Exception):
    """Base exception for the wisdom system."""


class StorageError(WisdomError):
    """Error in storage layer."""


class NotFoundError(WisdomError):
    """Entity not found."""

    def __init__(self, entity_type: str, entity_id: str):
        self.entity_type = entity_type
        self.entity_id = entity_id
        super().__init__(f"{entity_type} not found: {entity_id}")


class ExtractionError(WisdomError):
    """Error during knowledge extraction."""


class SynthesisError(WisdomError):
    """Error during wisdom synthesis."""


class ProviderError(WisdomError):
    """Error with LLM provider."""


class ValidationError(WisdomError):
    """Validation error."""


class ConfigError(WisdomError):
    """Configuration error."""
