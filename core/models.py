"""Domain models for Context Assembler Engine."""

from enum import IntEnum, StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator

from core.exceptions import InvalidBlockError


class PriorityTier(IntEnum):
    """Priority tiers dictating retention order under budget constraints."""

    CRITICAL_SYSTEM = 0  # Safety, system instructions (never pruned)
    TOOLS = 1  # Tool definitions, function schemas
    CONVERSATION_HISTORY = 2  # Multi-turn chat messages
    RETRIEVED_KNOWLEDGE = 3  # RAG documents, citations


class TruncationPolicy(StrEnum):
    """Truncation policy applied to a context block when budget is tight."""

    DROP = "DROP"  # Drop the entire block if it doesn't fit
    TRUNCATE_HEAD = "TRUNCATE_HEAD"  # Truncate oldest tokens from head (for history)
    TRUNCATE_TAIL = "TRUNCATE_TAIL"  # Truncate newest tokens from tail (for docs)


class ContextBlock(BaseModel):
    """A unit of prompt context submitted for assembly."""

    id: str = Field(..., description="Unique identifier for the block")
    role: str = Field(..., description="Message role (e.g. system, user, assistant, tool)")
    content: str = Field(..., description="Text content of the context block")
    priority: PriorityTier = Field(
        default=PriorityTier.RETRIEVED_KNOWLEDGE,
        description="Priority tier dictating pruning order",
    )
    policy: TruncationPolicy = Field(
        default=TruncationPolicy.DROP,
        description="Policy applied when block must be pruned",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary caller metadata",
    )

    @field_validator("content")
    @classmethod
    def validate_content_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise InvalidBlockError("Context block content cannot be empty or whitespace only.")
        return v

    @field_validator("id", "role")
    @classmethod
    def validate_non_empty_strings(cls, v: str) -> str:
        if not v or not v.strip():
            raise InvalidBlockError("Context block id and role must be non-empty strings.")
        return v.strip()


class AssemblyResult(BaseModel):
    """The final assembled prompt context conforming to budget."""

    messages: list[dict[str, str]] = Field(
        default_factory=list,
        description="Standardized list of messages [{'role': ..., 'content': ...}]",
    )
    total_tokens: int = Field(..., description="Total token count of assembled prompt")
    budget: int = Field(..., description="Configured token budget limit")
    dropped_blocks: list[str] = Field(
        default_factory=list,
        description="IDs of blocks dropped completely during assembly",
    )
    truncated_blocks: list[str] = Field(
        default_factory=list,
        description="IDs of blocks partially truncated during assembly",
    )
    execution_time_ms: float = Field(
        ...,
        description="Assembly duration in milliseconds",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Detailed audit information regarding token allocations and pruning",
    )

    def to_openai(self) -> list[dict[str, str]]:
        """Export messages for OpenAI SDK."""
        return self.messages

    def to_dict(self) -> dict[str, Any]:
        """Export as dictionary."""
        return self.model_dump()
