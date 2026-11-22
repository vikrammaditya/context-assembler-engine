"""Tests for budget manager and priority pruning."""

import pytest

from connectors.tokenizers.tiktoken_adapter import TiktokenAdapter
from core.budget import BudgetManager
from core.exceptions import ContextOverflowError
from core.models import ContextBlock, PriorityTier, TruncationPolicy


def test_budget_manager_tier0_overflow() -> None:
    tokenizer = TiktokenAdapter()
    bm = BudgetManager(budget=10, tokenizer=tokenizer)
    block0 = ContextBlock(
        id="sys-1",
        role="system",
        content="This is a very long system prompt that definitely contains more than ten tokens.",
        priority=PriorityTier.CRITICAL_SYSTEM,
    )
    with pytest.raises(ContextOverflowError):
        bm.prune_and_budget([block0])


def test_guardrail_system_instruction_preservation() -> None:
    # E11: System instruction preservation
    tokenizer = TiktokenAdapter()
    bm = BudgetManager(budget=15, tokenizer=tokenizer)
    b0 = ContextBlock(
        id="sys",
        role="system",
        content="System instruction guardrail",
        priority=PriorityTier.CRITICAL_SYSTEM,
    )
    b3_1 = ContextBlock(
        id="rag-1",
        role="user",
        content="Retrieval chunk 1 with lots of words repeating words repeating words",
        priority=PriorityTier.RETRIEVED_KNOWLEDGE,
        policy=TruncationPolicy.DROP,
    )
    b3_2 = ContextBlock(
        id="rag-2",
        role="user",
        content="Retrieval chunk 2 with more words repeating words repeating words",
        priority=PriorityTier.RETRIEVED_KNOWLEDGE,
        policy=TruncationPolicy.DROP,
    )
    retained, dropped, _ = bm.prune_and_budget([b0, b3_1, b3_2])
    retained_ids = [b.id for b in retained]
    assert "sys" in retained_ids
    assert len(dropped) >= 1


def test_guardrail_budget_invariant() -> None:
    # E10: Hard budget invariant
    tokenizer = TiktokenAdapter()
    budget = 100
    bm = BudgetManager(budget=budget, tokenizer=tokenizer)

    blocks = [
        ContextBlock(
            id=f"b-{i}",
            role="user",
            content=f"Context chunk number {i} with some additional filler text for tokens.",
            priority=PriorityTier.RETRIEVED_KNOWLEDGE
            if i % 2 == 0
            else PriorityTier.CONVERSATION_HISTORY,
            policy=TruncationPolicy.DROP if i % 2 == 0 else TruncationPolicy.TRUNCATE_TAIL,
        )
        for i in range(25)
    ]
    retained, _, _ = bm.prune_and_budget(blocks)
    total_tokens = sum(tokenizer.count_tokens(b.content) for b in retained)
    assert total_tokens <= budget
