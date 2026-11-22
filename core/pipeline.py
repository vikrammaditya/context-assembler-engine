"""Main Context Assembly Pipeline."""

import time
from typing import Any

from core.budget import BudgetManager
from core.models import AssemblyResult, ContextBlock
from core.ports import TokenizerPort


class AssemblyPipeline:
    """Orchestrates deterministic prompt context assembly and dynamic token budgeting."""

    def __init__(self, budget: int, tokenizer: TokenizerPort) -> None:
        if budget <= 0:
            raise ValueError("Token budget must be greater than zero.")
        self.budget = budget
        self.tokenizer = tokenizer
        self.budget_manager = BudgetManager(budget=self.budget, tokenizer=self.tokenizer)
        self._blocks: list[ContextBlock] = []

    def add_block(self, block: ContextBlock) -> "AssemblyPipeline":
        """Add a context block to the assembly pipeline."""
        self._blocks.append(block)
        return self

    def assemble(self, blocks: list[ContextBlock] | None = None) -> AssemblyResult:
        """Assemble context blocks deterministically within budget constraints."""
        start_time = time.perf_counter()
        target_blocks = blocks if blocks is not None else self._blocks

        retained, dropped_ids, truncated_ids = self.budget_manager.prune_and_budget(target_blocks)

        messages: list[dict[str, str]] = [{"role": b.role, "content": b.content} for b in retained]

        total_tokens = sum(self.tokenizer.count_tokens(m["content"]) for m in messages)
        duration_ms = (time.perf_counter() - start_time) * 1000.0

        metadata: dict[str, Any] = {
            "initial_blocks_count": len(target_blocks),
            "retained_blocks_count": len(retained),
            "dropped_count": len(dropped_ids),
            "truncated_count": len(truncated_ids),
            "budget_utilization_ratio": round(total_tokens / self.budget, 4)
            if self.budget > 0
            else 0,
        }

        return AssemblyResult(
            messages=messages,
            total_tokens=total_tokens,
            budget=self.budget,
            dropped_blocks=dropped_ids,
            truncated_blocks=truncated_ids,
            execution_time_ms=round(duration_ms, 3),
            metadata=metadata,
        )
