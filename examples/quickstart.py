"""Quickstart demonstration of Context Assembler Engine."""

import pathlib
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from connectors.tokenizers.tiktoken_adapter import TiktokenAdapter
from core.models import ContextBlock, PriorityTier, TruncationPolicy
from core.pipeline import AssemblyPipeline


def main() -> None:
    print("=== Context Assembler Engine Quickstart ===")

    # Initialize tokenizer and pipeline with a tight 120 token budget
    tokenizer = TiktokenAdapter()
    pipeline = AssemblyPipeline(budget=120, tokenizer=tokenizer)

    # Add blocks across different priority tiers
    pipeline.add_block(
        ContextBlock(
            id="sys-prompt",
            role="system",
            content=(
                "You are a trusted financial research assistant. "
                "Follow all safety guidelines strictly."
            ),
            priority=PriorityTier.CRITICAL_SYSTEM,
        )
    )
    pipeline.add_block(
        ContextBlock(
            id="tool-schema",
            role="system",
            content=(
                "Tool: query_market(ticker: str) -> dict[str, float]. "
                "Returns latest trade price and volume."
            ),
            priority=PriorityTier.TOOLS,
            policy=TruncationPolicy.DROP,
        )
    )
    pipeline.add_block(
        ContextBlock(
            id="chat-history",
            role="user",
            content=(
                "Turn 1: What were yesterday's top tech gainers? Turn 2: Show me NVDA and MSFT."
            ),
            priority=PriorityTier.CONVERSATION_HISTORY,
            policy=TruncationPolicy.TRUNCATE_HEAD,
        )
    )
    pipeline.add_block(
        ContextBlock(
            id="rag-document",
            role="user",
            content=(
                "Document SEC 10-K excerpt: Total net income increased by 14% "
                "year-over-year driven by cloud growth."
            ),
            priority=PriorityTier.RETRIEVED_KNOWLEDGE,
            policy=TruncationPolicy.TRUNCATE_TAIL,
        )
    )

    result = pipeline.assemble()

    print(f"\nConfigured Budget: {result.budget} tokens")
    print(f"Total Assembled Tokens: {result.total_tokens} tokens")
    print(f"Execution Latency: {result.execution_time_ms} ms")
    print(f"Dropped Blocks: {result.dropped_blocks}")
    print(f"Truncated Blocks: {result.truncated_blocks}")
    print("\n--- Assembled Messages ---")
    for msg in result.messages:
        print(f"[{msg['role']}]: {msg['content']}\n")

    print("Quickstart completed successfully.")


if __name__ == "__main__":
    main()
