"""Port definitions (protocols) for pluggable components."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class TokenizerPort(Protocol):
    """Protocol for token counting and directional truncation."""

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in the given text string."""
        ...

    def truncate_tokens(self, text: str, max_tokens: int, from_head: bool = False) -> str:
        """Truncate text so its token count does not exceed max_tokens."""
        ...
