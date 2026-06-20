"""Company context, grounding, and evidence contracts."""

from __future__ import annotations

from typing import Self

from pydantic import Field, field_validator, model_validator

from psview_agent.domain.base import StrictModel
from psview_agent.domain.enums import EvidenceKind, SourceField
from psview_agent.utils.collections import dedupe_casefold
from psview_agent.utils.text import normalize_whitespace, remove_empty_strings


def _normalize_list(values: list[str]) -> list[str]:
    return dedupe_casefold(remove_empty_strings(values))


class CompanyContextInput(StrictModel):
    company_name: str = Field(min_length=2, max_length=120)
    company_description: str = Field(min_length=20, max_length=3000)
    culture_and_values: str = Field(min_length=10, max_length=2500)
    hiring_profiles: str = Field(min_length=10, max_length=2500)
    communication_tone: str = Field(min_length=3, max_length=1000)
    recruiting_intent: str = Field(min_length=10, max_length=1500)
    additional_context: str = Field(default="", max_length=2000)

    @field_validator("*", mode="after")
    @classmethod
    def reject_blank_values(cls, value: str) -> str:
        if value == "" and cls.__name__ == "CompanyContextInput":
            return value
        if not normalize_whitespace(value) and value != "":
            raise ValueError("value must not be blank")
        return value


class SourceSegment(StrictModel):
    id: str = Field(pattern=r"^segment_\d{3}$")
    source_field: SourceField
    text: str = Field(min_length=1, max_length=700)
    ordinal: int = Field(ge=1)


class CompanyIdentity(StrictModel):
    name: str = Field(min_length=2, max_length=120)
    summary: str = Field(min_length=10, max_length=500)
    industry_or_category: str = Field(min_length=2, max_length=120)
    mission: str | None = Field(default=None, min_length=5, max_length=300)


class CompanyCulture(StrictModel):
    values: list[str] = Field(min_length=1, max_length=8)
    working_style: list[str] = Field(default_factory=list, max_length=8)
    differentiators: list[str] = Field(default_factory=list, max_length=8)

    @field_validator("values", "working_style", "differentiators", mode="before")
    @classmethod
    def normalize_lists(cls, value: object) -> object:
        if isinstance(value, list):
            return _normalize_list([str(item) for item in value])
        return value


class HiringProfile(StrictModel):
    target_profiles: list[str] = Field(min_length=1, max_length=10)
    desired_signals: list[str] = Field(default_factory=list, max_length=10)
    likely_candidate_motivations: list[str] = Field(default_factory=list, max_length=10)

    @field_validator(
        "target_profiles", "desired_signals", "likely_candidate_motivations", mode="before"
    )
    @classmethod
    def normalize_lists(cls, value: object) -> object:
        if isinstance(value, list):
            return _normalize_list([str(item) for item in value])
        return value


class CommunicationProfile(StrictModel):
    tone_attributes: list[str] = Field(min_length=2, max_length=8)
    preferred_language_patterns: list[str] = Field(default_factory=list, max_length=8)
    language_to_avoid: list[str] = Field(default_factory=list, max_length=10)

    @field_validator(
        "tone_attributes", "preferred_language_patterns", "language_to_avoid", mode="before"
    )
    @classmethod
    def normalize_lists(cls, value: object) -> object:
        if isinstance(value, list):
            return _normalize_list([str(item) for item in value])
        return value


class CompanyProfileDraft(StrictModel):
    identity: CompanyIdentity
    culture: CompanyCulture
    hiring_profile: HiringProfile
    communication_profile: CommunicationProfile


class CompanyProfile(CompanyProfileDraft):
    pass


class EvidenceFactDraft(StrictModel):
    fact: str = Field(min_length=5, max_length=300)
    kind: EvidenceKind = EvidenceKind.COMPANY_IDENTITY
    source_segment_ids: list[str] = Field(min_length=1, max_length=8)
    retrieval_tags: list[str] = Field(default_factory=list, max_length=10)

    @field_validator("source_segment_ids", "retrieval_tags", mode="before")
    @classmethod
    def normalize_lists(cls, value: object) -> object:
        if isinstance(value, list):
            return _normalize_list([str(item) for item in value])
        return value


class EvidenceFact(StrictModel):
    id: str = Field(pattern=r"^fact_\d{3}$")
    fact: str = Field(min_length=5, max_length=300)
    kind: EvidenceKind = EvidenceKind.COMPANY_IDENTITY
    source_segment_ids: list[str] = Field(min_length=1, max_length=8)
    retrieval_tags: list[str] = Field(default_factory=list, max_length=10)

    @field_validator("source_segment_ids", "retrieval_tags", mode="before")
    @classmethod
    def normalize_lists(cls, value: object) -> object:
        if isinstance(value, list):
            return _normalize_list([str(item) for item in value])
        return value


class EvidenceCorpus(StrictModel):
    source_segments: list[SourceSegment] = Field(min_length=1)
    evidence_facts: list[EvidenceFact] = Field(min_length=3, max_length=20)

    @model_validator(mode="after")
    def validate_references(self) -> Self:
        segment_ids = {segment.id for segment in self.source_segments}
        seen_facts: set[str] = set()
        for fact in self.evidence_facts:
            if fact.fact.casefold() in seen_facts:
                raise ValueError("duplicate evidence fact")
            seen_facts.add(fact.fact.casefold())
            unknown = [
                segment_id
                for segment_id in fact.source_segment_ids
                if segment_id not in segment_ids
            ]
            if unknown:
                raise ValueError(f"unknown source segment ids: {', '.join(unknown)}")
        return self
