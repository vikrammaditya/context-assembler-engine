"""Domain exceptions for Context Assembler Engine."""

from typing import Any


class ContextAssemblerError(Exception):
    """Base exception for all Context Assembler errors."""


class ContextOverflowError(ContextAssemblerError):
    """Raised when context exceeds hard token budget and cannot be budgeted."""

    def __init__(
        self,
        message: str,
        budget: int,
        requested_tokens: int,
        tier0_tokens: int,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.budget = budget
        self.requested_tokens = requested_tokens
        self.tier0_tokens = tier0_tokens
        self.details = details or {}


class InvalidBlockError(ContextAssemblerError):
    """Raised when a context block has invalid schema or empty content."""
