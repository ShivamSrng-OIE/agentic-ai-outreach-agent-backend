"""Generated response and evaluation models."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import Field, field_validator, model_validator

from psview_agent.domain.base import StrictModel
from psview_agent.utils.collections import dedupe_casefold
from psview_agent.utils.text import remove_empty_strings


def _sorted_unique_ids(values: list[str]) -> list[str]:
    return sorted(dedupe_casefold(values))


class SupportedClaim(StrictModel):
    claim: str = Field(min_length=5, max_length=300)
    evidence_fact_ids: list[str] = Field(min_length=1, max_length=8)

    @field_validator("evidence_fact_ids", mode="before")
    @classmethod
    def normalize_ids(cls, value: object) -> object:
        if isinstance(value, list):
            return _sorted_unique_ids(remove_empty_strings([str(item) for item in value]))
        return value


class GeneratedResponseDraft(StrictModel):
    message: str = Field(min_length=1, max_length=1000)
    supported_claims: list[SupportedClaim] = Field(default_factory=list, max_length=8)
    company_fact_ids_used: list[str] = Field(default_factory=list, max_length=8)

    @field_validator("company_fact_ids_used", mode="before")
    @classmethod
    def normalize_ids(cls, value: object) -> object:
        if isinstance(value, list):
            return _sorted_unique_ids(remove_empty_strings([str(item) for item in value]))
        return value

    @model_validator(mode="after")
    def derive_company_fact_ids(self) -> GeneratedResponseDraft:
        derived_ids = _sorted_unique_ids(
            [
                fact_id
                for claim in self.supported_claims
                for fact_id in claim.evidence_fact_ids
            ]
        )
        if derived_ids:
            object.__setattr__(self, "company_fact_ids_used", derived_ids)
        return self


class DeterministicResponseCheck(StrictModel):
    passed: bool
    violations: list[str] = Field(default_factory=list)
    question_count: int = Field(ge=0)
    character_count: int = Field(ge=0)


class ResponseEvaluation(StrictModel):
    personality_consistency: float = Field(ge=0, le=1)
    company_grounding: float = Field(ge=0, le=1)
    candidate_relevance: float = Field(ge=0, le=1)
    action_alignment: float = Field(ge=0, le=1)
    conversational_naturalness: float = Field(ge=0, le=1)
    repetition_risk: float = Field(ge=0, le=1)
    unsupported_claims: list[str] = Field(default_factory=list)
    personality_violations: list[str] = Field(default_factory=list)
    policy_violations: list[str] = Field(default_factory=list)
    passed: bool
    revision_instructions: list[str] = Field(default_factory=list)

    @field_validator(
        "unsupported_claims",
        "personality_violations",
        "policy_violations",
        "revision_instructions",
        mode="before",
    )
    @classmethod
    def normalize_lists(cls, value: object) -> object:
        if isinstance(value, list):
            return dedupe_casefold(remove_empty_strings([str(item) for item in value]))
        return value


class EvaluationSummary(StrictModel):
    personality_consistency: float = Field(ge=0, le=1)
    company_grounding: float = Field(ge=0, le=1)
    candidate_relevance: float = Field(ge=0, le=1)
    action_alignment: float = Field(ge=0, le=1)
    conversational_naturalness: float = Field(ge=0, le=1)
    repetition_risk: float = Field(ge=0, le=1)
    passed: bool
    revised: bool
    fallback_used: bool


@dataclass(frozen=True, slots=True)
class EvaluationThresholds:
    personality_consistency: float = 0.75
    company_grounding: float = 0.75
    candidate_relevance: float = 0.75
    action_alignment: float = 0.80
    conversational_naturalness: float = 0.70
    repetition_risk: float = 0.40


def evaluation_passes(
    evaluation: ResponseEvaluation, thresholds: EvaluationThresholds | None = None
) -> tuple[bool, list[str]]:
    """Compute pass/fail in application code."""
    active = thresholds or EvaluationThresholds()
    reasons: list[str] = []
    if evaluation.personality_consistency < active.personality_consistency:
        reasons.append("personality_consistency")
    if evaluation.company_grounding < active.company_grounding:
        reasons.append("company_grounding")
    if evaluation.candidate_relevance < active.candidate_relevance:
        reasons.append("candidate_relevance")
    if evaluation.action_alignment < active.action_alignment:
        reasons.append("action_alignment")
    if evaluation.conversational_naturalness < active.conversational_naturalness:
        reasons.append("conversational_naturalness")
    if evaluation.repetition_risk > active.repetition_risk:
        reasons.append("repetition_risk")
    if evaluation.unsupported_claims:
        reasons.append("unsupported_claims")
    if evaluation.policy_violations:
        reasons.append("policy_violations")
    return (not reasons, reasons)
