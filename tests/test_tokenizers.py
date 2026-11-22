"""Tests for tokenizers and protocols."""

from connectors.tokenizers.tiktoken_adapter import TiktokenAdapter
from core.ports import TokenizerPort


class DummyTokenizer:
    """Mock tokenizer conforming to TokenizerPort (E12)."""

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def truncate_tokens(self, text: str, max_tokens: int, from_head: bool = False) -> str:
        words = text.split()
        if len(words) <= max_tokens:
            return text
        if from_head:
            return "... " + " ".join(words[-max_tokens:])
        return " ".join(words[:max_tokens]) + " ..."


def test_tiktoken_adapter() -> None:
    adapter = TiktokenAdapter()
    assert isinstance(adapter, TokenizerPort)
    tokens = adapter.count_tokens("Hello world, this is a test prompt.")
    assert tokens > 0

    truncated = adapter.truncate_tokens(
        "One two three four five six seven eight nine ten eleven twelve", 8, from_head=False
    )
    assert "[truncated]" in truncated


def test_guardrail_tokenizer_decoupling() -> None:
    # E12: Pluggable tokenizer decoupling
    dummy = DummyTokenizer()
    assert isinstance(dummy, TokenizerPort)
    assert dummy.count_tokens("alpha beta gamma") == 3
    assert dummy.truncate_tokens("alpha beta gamma", 2, from_head=True) == "... beta gamma"
