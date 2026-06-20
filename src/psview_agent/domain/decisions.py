"""Candidate analysis, decision, and trace models."""

from __future__ import annotations

from typing import Self

from pydantic import Field, field_validator, model_validator

from psview_agent.domain.base import StrictModel
from psview_agent.domain.company import EvidenceFact
from psview_agent.domain.enums import (
    AgentAction,
    CandidateIntent,
    ConversationStage,
    EngagementLevel,
    Sentiment,
)
from psview_agent.domain.retrieval import RetrievedEvidence
from psview_agent.utils.collections import dedupe_casefold
from psview_agent.utils.text import remove_empty_strings


def _normalize_list(value: object) -> object:
    if isinstance(value, list):
        return dedupe_casefold(remove_empty_strings([str(item) for item in value]))
    return value


class CandidateAnalysis(StrictModel):
    intent: CandidateIntent
    sentiment: Sentiment
    engagement_level: EngagementLevel
    observed_signals: list[str] = Field(default_factory=list, max_length=8)
    expressed_motivations: list[str] = Field(default_factory=list, max_length=8)
    candidate_concerns: list[str] = Field(default_factory=list, max_length=8)
    questions_or_topics: list[str] = Field(default_factory=list, max_length=8)
    explicit_opt_out: bool
    reply_summary: str = Field(min_length=5, max_length=300)
    retrieval_topics: list[str] = Field(default_factory=list, max_length=8)
    confidence: float = Field(ge=0, le=1)

    @field_validator(
        "observed_signals",
        "expressed_motivations",
        "candidate_concerns",
        "questions_or_topics",
        "retrieval_topics",
        mode="before",
    )
    @classmethod
    def normalize_lists(cls, value: object) -> object:
        return _normalize_list(value)

    @model_validator(mode="after")
    def validate_opt_out_rules(self) -> Self:
        if self.intent is CandidateIntent.DO_NOT_CONTACT and not self.explicit_opt_out:
            raise ValueError("do_not_contact requires explicit_opt_out=True")
        if self.explicit_opt_out and self.intent not in {
            CandidateIntent.DO_NOT_CONTACT,
            CandidateIntent.CLEAR_REJECTION,
        }:
            raise ValueError("explicit opt-out only allowed for do_not_contact or clear_rejection")
        return self


class AgentDecisionDraft(StrictModel):
    current_stage: ConversationStage
    proposed_next_stage: ConversationStage
    objective: str = Field(min_length=5, max_length=300)
    proposed_action: AgentAction
    company_fact_ids_to_use: list[str] = Field(default_factory=list, max_length=8)
    missing_information: list[str] = Field(default_factory=list, max_length=8)
    should_continue: bool
    should_ask_question: bool
    rationale_summary: str = Field(min_length=5, max_length=300)
    confidence: float = Field(ge=0, le=1)

    @field_validator("company_fact_ids_to_use", "missing_information", mode="before")
    @classmethod
    def normalize_lists(cls, value: object) -> object:
        return _normalize_list(value)


class AgentDecision(StrictModel):
    candidate_intent: CandidateIntent
    sentiment: Sentiment
    engagement_level: EngagementLevel
    current_stage: ConversationStage
    next_stage: ConversationStage
    objective: str = Field(min_length=5, max_length=300)
    selected_action: AgentAction
    observed_signals: list[str] = Field(default_factory=list, max_length=8)
    candidate_concerns: list[str] = Field(default_factory=list, max_length=8)
    retrieved_evidence: list[RetrievedEvidence] = Field(default_factory=list, max_length=12)
    company_fact_ids_to_use: list[str] = Field(default_factory=list, max_length=8)
    missing_information: list[str] = Field(default_factory=list, max_length=8)
    should_continue: bool
    should_ask_question: bool
    confidence: float = Field(ge=0, le=1)
    rationale_summary: str = Field(min_length=5, max_length=300)
    policy_overrides: list[str] = Field(default_factory=list, max_length=12)

    @field_validator(
        "observed_signals",
        "candidate_concerns",
        "company_fact_ids_to_use",
        "missing_information",
        "policy_overrides",
        mode="before",
    )
    @classmethod
    def normalize_lists(cls, value: object) -> object:
        return _normalize_list(value)


class DecisionTrace(StrictModel):
    candidate_intent: CandidateIntent
    sentiment: Sentiment
    engagement_level: EngagementLevel
    current_stage: ConversationStage
    next_stage: ConversationStage
    objective: str
    selected_action: AgentAction
    observed_signals: list[str]
    candidate_concerns: list[str]
    retrieved_company_facts: list[RetrievedEvidence]
    company_facts_used: list[EvidenceFact]
    missing_information: list[str]
    should_continue: bool
    confidence: float = Field(ge=0, le=1)
    rationale_summary: str
    policy_overrides: list[str]
