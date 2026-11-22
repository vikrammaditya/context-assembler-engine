"""Tiktoken adapter implementing TokenizerPort."""

from pathlib import Path

import tiktoken

from core.ports import TokenizerPort

ELLIPSIS_MARKER = "... [truncated]"


class TiktokenAdapter(TokenizerPort):
    """TokenizerPort implementation using OpenAI's tiktoken."""

    def __init__(
        self,
        encoding_name: str = "cl100k_base",
        cache_dir: Path | str | None = None,
    ) -> None:
        self.encoding_name = encoding_name
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._encoding = tiktoken.get_encoding(encoding_name)
        self._marker_tokens = len(self._encoding.encode(ELLIPSIS_MARKER))

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(self._encoding.encode(text))

    def truncate_tokens(self, text: str, max_tokens: int, from_head: bool = False) -> str:
        if max_tokens <= 0:
            return ""

        tokens = self._encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text

        marker = ELLIPSIS_MARKER + "\n" if from_head else "\n" + ELLIPSIS_MARKER
        marker_tokens = len(self._encoding.encode(marker))

        if max_tokens <= marker_tokens:
            kept_tokens = tokens[-max_tokens:] if from_head else tokens[:max_tokens]
            return self._encoding.decode(kept_tokens)

        usable_tokens = max_tokens - marker_tokens
        while usable_tokens > 0:
            if from_head:
                candidate = marker + self._encoding.decode(tokens[-usable_tokens:])
            else:
                candidate = self._encoding.decode(tokens[:usable_tokens]) + marker
            if len(self._encoding.encode(candidate)) <= max_tokens:
                return candidate
            usable_tokens -= 1

        kept_tokens = tokens[-max_tokens:] if from_head else tokens[:max_tokens]
        return self._encoding.decode(kept_tokens)
