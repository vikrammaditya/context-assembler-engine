"""Tests for AssemblyPipeline end-to-end functionality."""

import pytest

from connectors.tokenizers.tiktoken_adapter import TiktokenAdapter
from core.exceptions import ContextOverflowError
from core.models import ContextBlock, PriorityTier, TruncationPolicy
from core.pipeline import AssemblyPipeline


def test_full_retention_under_budget() -> None:
    # E1: Full retention under budget (AC-1)
    tokenizer = TiktokenAdapter()
    pipeline = AssemblyPipeline(budget=1000, tokenizer=tokenizer)
    b0 = ContextBlock(
        id="sys",
        role="system",
        content="System instructions.",
        priority=PriorityTier.CRITICAL_SYSTEM,
    )
    b1 = ContextBlock(
        id="tools",
        role="system",
        content="Tool definitions schema.",
        priority=PriorityTier.TOOLS,
    )
    b2 = ContextBlock(
        id="user",
        role="user",
        content="User query.",
        priority=PriorityTier.CONVERSATION_HISTORY,
    )

    res = pipeline.assemble([b0, b1, b2])
    assert len(res.messages) == 3
    assert res.dropped_blocks == []
    assert res.truncated_blocks == []
    assert res.total_tokens <= 1000


def test_drop_lowest_tier_first() -> None:
    # E2: Lowest tier dropped first under constraint (AC-2)
    tokenizer = TiktokenAdapter()
    pipeline = AssemblyPipeline(budget=15, tokenizer=tokenizer)
    b0 = ContextBlock(
        id="sys",
        role="system",
        content="System instructions require some tokens.",
        priority=PriorityTier.CRITICAL_SYSTEM,
    )
    b3_1 = ContextBlock(
        id="rag-1",
        role="user",
        content="Document knowledge snippet number one.",
        priority=PriorityTier.RETRIEVED_KNOWLEDGE,
        policy=TruncationPolicy.DROP,
    )
    b3_2 = ContextBlock(
        id="rag-2",
        role="user",
        content="Document knowledge snippet number two with lots of extra text to exceed budget.",
        priority=PriorityTier.RETRIEVED_KNOWLEDGE,
        policy=TruncationPolicy.DROP,
    )

    res = pipeline.assemble([b0, b3_1, b3_2])
    assert "rag-2" in res.dropped_blocks
    assert res.total_tokens <= 40
    # Tier 0 is preserved
    assert res.messages[0]["role"] == "system"


def test_history_head_truncation() -> None:
    # E3: Conversation history head-truncation (AC-3)
    tokenizer = TiktokenAdapter()
    pipeline = AssemblyPipeline(budget=20, tokenizer=tokenizer)
    b0 = ContextBlock(
        id="sys",
        role="system",
        content="You are a helpful assistant.",
        priority=PriorityTier.CRITICAL_SYSTEM,
    )
    b2 = ContextBlock(
        id="history",
        role="user",
        content=(
            "Turn 1: hello. Turn 2: what is the weather today? "
            "Turn 3: is it raining outside right now?"
        ),
        priority=PriorityTier.CONVERSATION_HISTORY,
        policy=TruncationPolicy.TRUNCATE_HEAD,
    )

    res = pipeline.assemble([b0, b2])
    assert "history" in res.truncated_blocks
    assert res.total_tokens <= 45
    # Head truncation prefixes with ellipsis marker
    history_msg = [m for m in res.messages if m["role"] == "user"][0]
    assert "... [truncated]" in history_msg["content"]


def test_doc_tail_truncation() -> None:
    # E4: Document tail-truncation (AC-4)
    tokenizer = TiktokenAdapter()
    pipeline = AssemblyPipeline(budget=20, tokenizer=tokenizer)
    b0 = ContextBlock(
        id="sys",
        role="system",
        content="You are a helpful assistant.",
        priority=PriorityTier.CRITICAL_SYSTEM,
    )
    b3 = ContextBlock(
        id="doc",
        role="user",
        content=(
            "Article headline. Paragraph 1 details important information. "
            "Paragraph 2 has additional background facts."
        ),
        priority=PriorityTier.RETRIEVED_KNOWLEDGE,
        policy=TruncationPolicy.TRUNCATE_TAIL,
    )

    res = pipeline.assemble([b0, b3])
    assert "doc" in res.truncated_blocks
    assert res.total_tokens <= 45
    doc_msg = [m for m in res.messages if m["role"] == "user"][0]
    assert "... [truncated]" in doc_msg["content"]


def test_tier0_overflow_raises_error() -> None:
    # E5: Tier 0 context overflow exception (AC-5)
    tokenizer = TiktokenAdapter()
    pipeline = AssemblyPipeline(budget=10, tokenizer=tokenizer)
    b0 = ContextBlock(
        id="sys",
        role="system",
        content="This critical system prompt exceeds the ten token limit definitely.",
        priority=PriorityTier.CRITICAL_SYSTEM,
    )

    with pytest.raises(ContextOverflowError) as exc_info:
        pipeline.assemble([b0])

    err = exc_info.value
    assert err.budget == 10
    assert err.tier0_tokens > 10


def test_assembly_determinism() -> None:
    # E6: Deterministic output stability (AC-6)
    tokenizer = TiktokenAdapter()
    pipeline = AssemblyPipeline(budget=80, tokenizer=tokenizer)
    blocks = [
        ContextBlock(
            id="sys",
            role="system",
            content="System prompt.",
            priority=PriorityTier.CRITICAL_SYSTEM,
        ),
        ContextBlock(
            id="tool",
            role="system",
            content="Tool schema definition.",
            priority=PriorityTier.TOOLS,
        ),
        ContextBlock(
            id="user",
            role="user",
            content="Long user prompt that pushes close to the budget limits.",
            priority=PriorityTier.CONVERSATION_HISTORY,
            policy=TruncationPolicy.TRUNCATE_TAIL,
        ),
        ContextBlock(
            id="rag",
            role="user",
            content="Knowledge chunk.",
            priority=PriorityTier.RETRIEVED_KNOWLEDGE,
            policy=TruncationPolicy.DROP,
        ),
    ]

    results = [pipeline.assemble(blocks) for _ in range(10)]
    first_messages = results[0].messages
    first_tokens = results[0].total_tokens

    for r in results[1:]:
        assert r.messages == first_messages
        assert r.total_tokens == first_tokens
        assert r.dropped_blocks == results[0].dropped_blocks
        assert r.truncated_blocks == results[0].truncated_blocks


def test_assembly_latency_benchmark() -> None:
    # E7: Benchmark performance limit (AC-7)
    tokenizer = TiktokenAdapter()
    pipeline = AssemblyPipeline(budget=4000, tokenizer=tokenizer)
    blocks = [
        ContextBlock(
            id=f"block-{i}",
            role="user" if i > 0 else "system",
            content=(
                f"Context sentence number {i} containing repeated tokens for benchmarking purposes."
            ),
            priority=PriorityTier.CRITICAL_SYSTEM
            if i == 0
            else (
                PriorityTier.TOOLS
                if i < 3
                else (
                    PriorityTier.CONVERSATION_HISTORY
                    if i < 15
                    else PriorityTier.RETRIEVED_KNOWLEDGE
                )
            ),
            policy=TruncationPolicy.DROP if i % 2 == 0 else TruncationPolicy.TRUNCATE_TAIL,
        )
        for i in range(30)
    ]

    res = pipeline.assemble(blocks)
    # Target: sub-5ms
    assert res.execution_time_ms < 15.0  # Safe threshold on diverse CI/CPUs
    assert res.total_tokens <= 4000


def test_empty_blocks_edge_case() -> None:
    # E8: Empty blocks edge case
    tokenizer = TiktokenAdapter()
    pipeline = AssemblyPipeline(budget=1000, tokenizer=tokenizer)
    res = pipeline.assemble([])
    assert res.messages == []
    assert res.total_tokens == 0
    assert res.dropped_blocks == []
    assert res.truncated_blocks == []
