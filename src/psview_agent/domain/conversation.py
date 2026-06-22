"""Conversation session models."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Self
from uuid import UUID

from pydantic import Field, field_validator, model_validator

from psview_agent.domain.agent import AgentConfiguration, OutreachPlan
from psview_agent.domain.base import StrictModel
from psview_agent.domain.candidate import CandidateProfile
from psview_agent.domain.enums import (
    AgentAction,
    CandidateIntent,
    ConversationStage,
    EngagementLevel,
    MessageRole,
)
from psview_agent.utils.collections import dedupe_casefold
from psview_agent.utils.text import remove_empty_strings
from psview_agent.utils.time import ensure_utc


class ConversationMessage(StrictModel):
    id: UUID
    role: MessageRole
    content: str = Field(min_length=1, max_length=1000)
    created_at: datetime

    @field_validator("created_at", mode="after")
    @classmethod
    def validate_timestamp(cls, value: datetime) -> datetime:
        return ensure_utc(value)


class ConversationState(StrictModel):
    stage: ConversationStage = ConversationStage.INITIAL_OUTREACH
    turn_count: int = Field(default=0, ge=0)
    engagement_level: EngagementLevel = EngagementLevel.MEDIUM
    known_motivations: list[str] = Field(default_factory=list)
    known_concerns: list[str] = Field(default_factory=list)
    answered_topics: list[str] = Field(default_factory=list)
    unanswered_topics: list[str] = Field(default_factory=list)
    company_fact_ids_already_used: list[str] = Field(default_factory=list)
    last_candidate_intent: CandidateIntent | None = None
    last_action: AgentAction | None = None
    is_closed: bool = False
    close_reason: str | None = None

    @field_validator(
        "known_motivations",
        "known_concerns",
        "answered_topics",
        "unanswered_topics",
        "company_fact_ids_already_used",
        mode="before",
    )
    @classmethod
    def normalize_lists(cls, value: object) -> object:
        if isinstance(value, list):
            return dedupe_casefold(remove_empty_strings([str(item) for item in value]))
        return value

    @model_validator(mode="after")
    def validate_closed_state(self) -> Self:
        if self.is_closed and self.stage is not ConversationStage.CLOSED:
            raise ValueError("closed state requires stage=closed")
        if self.stage is ConversationStage.CLOSED and not self.is_closed:
            raise ValueError("stage=closed requires is_closed=True")
        if self.is_closed and not self.close_reason:
            raise ValueError("closed state requires close_reason")
        return self


class ConversationSession(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    conversation_id: UUID
    configuration: AgentConfiguration
    candidate: CandidateProfile
    target_role: str = Field(min_length=2, max_length=300)
    target_role_description: str | None = Field(default=None, min_length=10, max_length=50000)
    outreach_plan: OutreachPlan
    state: ConversationState
    messages: list[ConversationMessage] = Field(min_length=1)
    created_at: datetime
    updated_at: datetime

    @field_validator("created_at", "updated_at", mode="after")
    @classmethod
    def validate_timestamps(cls, value: datetime) -> datetime:
        return ensure_utc(value)

    @model_validator(mode="after")
    def validate_message_order(self) -> Self:
        seen_ids: set[UUID] = set()
        last_time: datetime | None = None
        for message in self.messages:
            if message.id in seen_ids:
                raise ValueError("duplicate message id")
            seen_ids.add(message.id)
            if last_time and message.created_at < last_time:
                raise ValueError("message timestamps must be ordered")
            last_time = message.created_at
        return self
