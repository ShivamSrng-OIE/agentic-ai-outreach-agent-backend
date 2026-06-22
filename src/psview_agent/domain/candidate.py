"""Candidate domain models."""

from pydantic import Field

from psview_agent.domain.base import StrictModel
from psview_agent.utils.text import normalize_whitespace


class ExtractedCandidateProfile(StrictModel):
    name: str = Field(min_length=1, max_length=100)
    current_role: str = Field(min_length=2, max_length=200)
    background_summary: str = Field(min_length=10, max_length=2000)


class CandidateProfile(StrictModel):
    name: str = Field(min_length=1, max_length=100)
    current_role: str = Field(min_length=2, max_length=200)
    background_summary: str = Field(min_length=10, max_length=2000)
    resume_text: str | None = Field(default=None, max_length=50000)

    @classmethod
    def validate_target_role(cls, value: str) -> str:
        normalized = normalize_whitespace(value)
        if not 2 <= len(normalized) <= 300:
            raise ValueError("target role must be between 2 and 300 characters")
        return normalized
