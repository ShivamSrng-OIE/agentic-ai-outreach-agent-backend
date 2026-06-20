"""Agent persona, configuration, and outreach models."""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import Field, field_validator, model_validator

from psview_agent.domain.base import StrictModel
from psview_agent.domain.company import (
    CompanyContextInput,
    CompanyProfile,
    CompanyProfileDraft,
    EvidenceCorpus,
    EvidenceFactDraft,
)
from psview_agent.domain.enums import OutreachStage
from psview_agent.domain.evaluation import SupportedClaim, _sorted_unique_ids
from psview_agent.utils.collections import dedupe_casefold
from psview_agent.utils.text import remove_empty_strings


def _normalize_list(values: object) -> object:
    if isinstance(values, list):
        return dedupe_casefold(remove_empty_strings([str(item) for item in values]))
    return values


class AgentPersonaDraft(StrictModel):
    name: str = Field(min_length=2, max_length=40)
    role_identity: str = Field(min_length=5, max_length=120)
    personality_summary: str = Field(min_length=20, max_length=500)
    traits: list[str] = Field(min_length=3, max_length=6)
    communication_principles: list[str] = Field(min_length=3, max_length=8)
    questioning_style: str = Field(min_length=5, max_length=200)
    objection_handling_style: str = Field(min_length=5, max_length=200)
    boundaries: list[str] = Field(min_length=2, max_length=20)
    language_to_avoid: list[str] = Field(default_factory=list, max_length=10)

    @field_validator(
        "traits", "communication_principles", "boundaries", "language_to_avoid", mode="before"
    )
    @classmethod
    def normalize_lists(cls, value: object) -> object:
        return _normalize_list(value)


class AgentPersona(AgentPersonaDraft):
    pass


class CompanyAgentConfigurationDraft(StrictModel):
    company_profile: CompanyProfileDraft
    evidence_facts: list[EvidenceFactDraft]
    persona: AgentPersonaDraft


class AgentConfiguration(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    configuration_id: UUID
    company_context: CompanyContextInput
    company_profile: CompanyProfile
    persona: AgentPersona
    evidence_corpus: EvidenceCorpus
    created_at: datetime


class OutreachMessageDraft(StrictModel):
    stage: OutreachStage
    objective: str = Field(min_length=3, max_length=200)
    trigger: str = Field(min_length=3, max_length=200)
    message: str = Field(min_length=10, max_length=1000)
    supported_claims: list[SupportedClaim] = Field(default_factory=list, max_length=8)
    company_fact_ids_used: list[str] = Field(default_factory=list, max_length=8)

    @field_validator("company_fact_ids_used", mode="before")
    @classmethod
    def normalize_fact_ids(cls, value: object) -> object:
        return _normalize_list(value)

    @model_validator(mode="after")
    def derive_company_fact_ids(self) -> OutreachMessageDraft:
        derived_ids = _sorted_unique_ids(
            [fact_id for claim in self.supported_claims for fact_id in claim.evidence_fact_ids]
        )
        object.__setattr__(self, "company_fact_ids_used", derived_ids)
        return self


class OutreachMessage(OutreachMessageDraft):
    id: str = Field(pattern=r"^outreach_\d{3}$")


class OutreachPlanDraft(StrictModel):
    overall_intent: str = Field(min_length=5, max_length=300)
    messages: list[OutreachMessageDraft] = Field(min_length=3, max_length=3)


class OutreachPlan(StrictModel):
    overall_intent: str = Field(min_length=5, max_length=300)
    messages: list[OutreachMessage] = Field(min_length=3, max_length=3)
