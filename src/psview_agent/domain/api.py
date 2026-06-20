"""API request and response models."""

from __future__ import annotations

from datetime import datetime
from typing import Self

from pydantic import Field, model_validator

from psview_agent.domain.agent import AgentConfiguration
from psview_agent.domain.base import StrictModel
from psview_agent.domain.candidate import CandidateProfile
from psview_agent.domain.company import CompanyContextInput
from psview_agent.domain.conversation import (
    ConversationMessage,
    ConversationSession,
    ConversationState,
)
from psview_agent.domain.decisions import DecisionTrace
from psview_agent.domain.evaluation import EvaluationSummary


class HealthResponse(StrictModel):
    status: str
    service: str
    version: str
    environment: str


class ReadyResponse(StrictModel):
    status: str


class ConfigureAgentRequest(StrictModel):
    company_context: CompanyContextInput


class ConfigureAgentResponse(StrictModel):
    configuration: AgentConfiguration


class StartConversationRequest(StrictModel):
    configuration: AgentConfiguration | None = None
    company_context: CompanyContextInput | None = None
    candidate: CandidateProfile
    target_role: str = Field(min_length=2, max_length=300)
    target_role_description: str | None = Field(default=None, min_length=10, max_length=4000)

    @model_validator(mode="after")
    def validate_configuration_source(self) -> Self:
        has_configuration = self.configuration is not None
        has_company_context = self.company_context is not None
        if has_configuration == has_company_context:
            raise ValueError("provide exactly one of configuration or company_context")
        return self


class StartConversationResponse(StrictModel):
    session: ConversationSession
    initial_decision_trace: DecisionTrace


class ConversationTurnRequest(StrictModel):
    session: ConversationSession
    candidate_reply: str = Field(min_length=1, max_length=4000)


class ConversationTurnResponse(StrictModel):
    candidate_message: ConversationMessage
    agent_message: ConversationMessage
    updated_state: ConversationState
    decision_trace: DecisionTrace
    evaluation: EvaluationSummary
    updated_at: datetime


class ErrorDetail(StrictModel):
    field: str | None = None
    message: str


class ErrorResponse(StrictModel):
    code: str
    message: str
    request_id: str
    details: list[ErrorDetail] = Field(default_factory=list)
