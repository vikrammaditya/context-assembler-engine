"""Budget allocation and priority-based pruning logic."""

from core.exceptions import ContextOverflowError
from core.models import ContextBlock, PriorityTier, TruncationPolicy
from core.ports import TokenizerPort


class BudgetManager:
    """Manages token budgeting and priority-based pruning of context blocks."""

    def __init__(self, budget: int, tokenizer: TokenizerPort) -> None:
        self.budget = budget
        self.tokenizer = tokenizer

    def prune_and_budget(
        self, blocks: list[ContextBlock]
    ) -> tuple[list[ContextBlock], list[str], list[str]]:
        """Prune blocks to fit within budget based on priority tiers.

        Returns:
            Tuple of (retained_blocks, dropped_block_ids, truncated_block_ids)
        """
        if not blocks:
            return [], [], []

        block_tokens: dict[str, int] = {
            b.id: self.tokenizer.count_tokens(b.content) for b in blocks
        }

        tier0_tokens = sum(
            block_tokens[b.id] for b in blocks if b.priority == PriorityTier.CRITICAL_SYSTEM
        )
        if tier0_tokens > self.budget:
            msg = (
                f"Critical Tier 0 blocks require {tier0_tokens} tokens, "
                f"exceeding budget of {self.budget}."
            )
            raise ContextOverflowError(
                msg,
                budget=self.budget,
                requested_tokens=tier0_tokens,
                tier0_tokens=tier0_tokens,
                details={
                    "tier0_block_ids": [
                        b.id for b in blocks if b.priority == PriorityTier.CRITICAL_SYSTEM
                    ]
                },
            )

        total_tokens = sum(block_tokens.values())
        if total_tokens <= self.budget:
            return list(blocks), [], []

        retained: dict[str, ContextBlock] = {b.id: b for b in blocks}
        dropped_ids: list[str] = []
        truncated_ids: list[str] = []

        candidates = [b for b in blocks if b.priority != PriorityTier.CRITICAL_SYSTEM]
        candidates.sort(key=lambda b: (int(b.priority), blocks.index(b)), reverse=True)

        current_tokens = total_tokens

        for block in candidates:
            if current_tokens <= self.budget:
                break

            tokens_this_block = block_tokens[block.id]

            if block.policy == TruncationPolicy.DROP:
                del retained[block.id]
                dropped_ids.append(block.id)
                current_tokens -= tokens_this_block
            elif block.policy in (TruncationPolicy.TRUNCATE_HEAD, TruncationPolicy.TRUNCATE_TAIL):
                tokens_to_keep = self.budget - (current_tokens - tokens_this_block)
                if tokens_to_keep < 10:
                    del retained[block.id]
                    dropped_ids.append(block.id)
                    current_tokens -= tokens_this_block
                else:
                    from_head = block.policy == TruncationPolicy.TRUNCATE_HEAD
                    truncated_content = self.tokenizer.truncate_tokens(
                        block.content, tokens_to_keep, from_head=from_head
                    )
                    new_token_count = self.tokenizer.count_tokens(truncated_content)

                    retained[block.id] = block.model_copy(update={"content": truncated_content})
                    truncated_ids.append(block.id)
                    current_tokens = (current_tokens - tokens_this_block) + new_token_count
                    block_tokens[block.id] = new_token_count

        if current_tokens > self.budget:
            for tid in list(truncated_ids):
                if current_tokens <= self.budget:
                    break
                current_tokens -= block_tokens[tid]
                del retained[tid]
                truncated_ids.remove(tid)
                dropped_ids.append(tid)

        ordered_retained = [b for b in blocks if b.id in retained]
        final_retained = [retained[b.id] for b in ordered_retained]

        return final_retained, dropped_ids, truncated_ids
