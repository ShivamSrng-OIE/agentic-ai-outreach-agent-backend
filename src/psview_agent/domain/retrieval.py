"""Retrieval query and result models."""

from pydantic import Field, field_validator

from psview_agent.domain.base import StrictModel
from psview_agent.domain.company import EvidenceFact
from psview_agent.domain.enums import AgentAction
from psview_agent.utils.collections import dedupe_casefold
from psview_agent.utils.text import remove_empty_strings


class RetrievalQuery(StrictModel):
    text: str = Field(min_length=2, max_length=50000)
    target_role: str = Field(min_length=2, max_length=300)
    target_role_description: str | None = Field(default=None, min_length=10, max_length=50000)
    topics: list[str] = Field(default_factory=list, max_length=12)
    action: AgentAction | None = None

    @field_validator("topics", mode="before")
    @classmethod
    def normalize_topics(cls, value: object) -> object:
        if isinstance(value, list):
            normalized = dedupe_casefold(remove_empty_strings([str(item) for item in value]))
            return normalized[:12]
        return value


class RetrievedEvidence(StrictModel):
    evidence: EvidenceFact
    rank: int = Field(ge=1)
    raw_relevance_score: float = Field(ge=0)
    normalized_relevance: float = Field(ge=0, le=1)
    matched_terms: list[str] = Field(default_factory=list)

    @field_validator("matched_terms", mode="before")
    @classmethod
    def normalize_terms(cls, value: object) -> object:
        if isinstance(value, list):
            return dedupe_casefold(remove_empty_strings([str(item) for item in value]))
        return value
