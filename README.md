# Context Assembler Engine

A high-performance, deterministic prompt context assembly engine with dynamic token budgeting, priority-ranked block injection, and overflow protection for LLM application and agent pipelines.

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type Checked: Mypy](https://img.shields.io/badge/type%20checked-mypy-blue.svg)](https://mypy-lang.org/)

---

## Highlights

- **Dynamic Token Budgeting:** Set an absolute token budget for your prompt; the engine mathematically guarantees total tokens remain within limits.
- **Priority-Ranked Tiers:** Retain critical safety rules and system instructions (Tier 0) and tool schemas (Tier 1) while automatically pruning chat history (Tier 2) and RAG documents (Tier 3) under budget pressure.
- **Directional Truncation:** Smart trimming policies preserve recency for chat memory (head-truncation) and preserve introductory context for documents (tail-truncation) with clean ellipsis indicators.
- **Zero Silent Failures:** Raises explicit `ContextOverflowError` if mandatory Tier 0 instructions cannot fit, preventing broken or unsafe prompts from ever reaching the LLM.
- **Sub-Millisecond Speed:** Synchronous in-memory pipeline executing in under 0.5ms for standard workloads with zero network dependencies.
- **Pluggable Tokenizer Interface:** Out-of-the-box `tiktoken` connector (`cl100k_base` default) with a clean `TokenizerPort` protocol to plug in custom or model-specific token counters.

---

## Priority Tiers

| Tier | Name | Target Content | Default Policy | Pruning Behavior |
| --- | --- | --- | --- | --- |
| **0** | `CRITICAL_SYSTEM` | System instructions, safety rules | Non-prunable | Never dropped. Raises `ContextOverflowError` if budget breached. |
| **1** | `TOOLS` | Function calling schemas, tool definitions | `DROP` | Pruned only if lower tiers are already exhausted. |
| **2** | `CONVERSATION_HISTORY` | Multi-turn chat memory | `TRUNCATE_HEAD` | Truncates oldest turns first, preserving recent context. |
| **3** | `RETRIEVED_KNOWLEDGE` | RAG documents, citations | `TRUNCATE_TAIL` / `DROP` | First to be pruned or trimmed when budget is tight. |

---

## Installation

Using `uv` (recommended):
```bash
uv add context-assembler-engine
```

Or using `pip`:
```bash
pip install context-assembler-engine
```

---

## Quickstart

```python
from connectors.tokenizers.tiktoken_adapter import TiktokenAdapter
from core.models import ContextBlock, PriorityTier, TruncationPolicy
from core.pipeline import AssemblyPipeline

# 1. Initialize pipeline with target token budget
tokenizer = TiktokenAdapter()
pipeline = AssemblyPipeline(budget=150, tokenizer=tokenizer)

# 2. Register context blocks across priority tiers
pipeline.add_block(
    ContextBlock(
        id="sys-prompt",
        role="system",
        content="You are a trusted financial assistant. Follow safety guidelines strictly.",
        priority=PriorityTier.CRITICAL_SYSTEM,
    )
)

pipeline.add_block(
    ContextBlock(
        id="tool-schema",
        role="system",
        content="Tool: query_stock(ticker: str) -> dict. Returns trade volume and price.",
        priority=PriorityTier.TOOLS,
        policy=TruncationPolicy.DROP,
    )
)

pipeline.add_block(
    ContextBlock(
        id="chat-history",
        role="user",
        content="Turn 1: What was yesterday's market trend? Turn 2: Summarize tech stocks.",
        priority=PriorityTier.CONVERSATION_HISTORY,
        policy=TruncationPolicy.TRUNCATE_HEAD,
    )
)

pipeline.add_block(
    ContextBlock(
        id="rag-knowledge",
        role="user",
        content="SEC 10-K report details: Cloud revenue grew 14% year-over-year.",
        priority=PriorityTier.RETRIEVED_KNOWLEDGE,
        policy=TruncationPolicy.TRUNCATE_TAIL,
    )
)

# 3. Assemble context
result = pipeline.assemble()

print(f"Total Tokens: {result.total_tokens} / {result.budget}")
print(f"Dropped Blocks: {result.dropped_blocks}")
print(f"Truncated Blocks: {result.truncated_blocks}")
print(f"Execution Latency: {result.execution_time_ms} ms")

# 4. Use formatted messages with OpenAI / Anthropic / any LLM SDK
messages = result.to_openai()
# response = client.chat.completions.create(model="gpt-4o", messages=messages)
```

---

## Truncation Policies

- **`DROP`**: Entire block is dropped if it cannot fit within available budget.
- **`TRUNCATE_HEAD`**: Trims oldest tokens from the beginning and prepends `... [truncated]\n` (ideal for chat history).
- **`TRUNCATE_TAIL`**: Trims newest tokens from the end and appends `\n... [truncated]` (ideal for documents).

---

## Custom Tokenizer

Implement the lightweight `TokenizerPort` protocol to use any tokenizer:

```python
from core.ports import TokenizerPort

class MyCustomTokenizer(TokenizerPort):
    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def truncate_tokens(self, text: str, max_tokens: int, from_head: bool = False) -> str:
        words = text.split()
        if len(words) <= max_tokens:
            return text
        return " ".join(words[-max_tokens:]) if from_head else " ".join(words[:max_tokens])

pipeline = AssemblyPipeline(budget=1000, tokenizer=MyCustomTokenizer())
```

---

## Development & Testing

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest

# Check code formatting and linting
uv run ruff check . && uv run ruff format --check .

# Type checking
uv run mypy .

# Run demo
uv run python examples/quickstart.py
```

---

## License

MIT License.
