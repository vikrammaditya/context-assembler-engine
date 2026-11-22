"""Tests for domain models and exceptions."""

import pytest
from pydantic import ValidationError

from core.exceptions import ContextOverflowError, InvalidBlockError
from core.models import ContextBlock, PriorityTier, TruncationPolicy


def test_valid_context_block() -> None:
    block = ContextBlock(
        id="block-1",
        role="user",
        content="Hello world",
        priority=PriorityTier.CONVERSATION_HISTORY,
        policy=TruncationPolicy.DROP,
    )
    assert block.id == "block-1"
    assert block.role == "user"
    assert block.content == "Hello world"
    assert block.priority == PriorityTier.CONVERSATION_HISTORY
    assert block.policy == TruncationPolicy.DROP


def test_invalid_block_schema() -> None:
    # E9: Invalid block schema rejection
    with pytest.raises((InvalidBlockError, ValidationError)):
        ContextBlock(id="bad-1", role="user", content="   ")

    with pytest.raises((InvalidBlockError, ValidationError)):
        ContextBlock(id="", role="user", content="content")

    with pytest.raises((InvalidBlockError, ValidationError)):
        ContextBlock(id="bad-2", role="", content="content")


def test_context_overflow_error_attributes() -> None:
    err = ContextOverflowError(
        message="Overflow",
        budget=100,
        requested_tokens=150,
        tier0_tokens=150,
        details={"keys": [1, 2]},
    )
    assert err.budget == 100
    assert err.requested_tokens == 150
    assert err.tier0_tokens == 150
    assert err.details == {"keys": [1, 2]}
