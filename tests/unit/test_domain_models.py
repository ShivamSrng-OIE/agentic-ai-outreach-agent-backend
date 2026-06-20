"""Tests for strict domain validation."""

import pytest
from pydantic import ValidationError

from psview_agent.domain.company import CompanyContextInput
from psview_agent.domain.conversation import ConversationState
from psview_agent.domain.enums import ConversationStage


def test_company_context_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        CompanyContextInput.model_validate(
            {
                "company_name": "Acme",
                "company_description": "x" * 25,
                "culture_and_values": "x" * 20,
                "hiring_profiles": "x" * 20,
                "communication_tone": "clear",
                "recruiting_intent": "x" * 20,
                "extra_field": "nope",
            }
        )


def test_company_context_rejects_whitespace_only_values() -> None:
    with pytest.raises(ValidationError):
        CompanyContextInput(
            company_name="   ",
            company_description="x" * 25,
            culture_and_values="x" * 20,
            hiring_profiles="x" * 20,
            communication_tone="clear",
            recruiting_intent="x" * 20,
        )


def test_closed_state_requires_closed_stage_and_reason() -> None:
    with pytest.raises(ValidationError):
        ConversationState(is_closed=True, stage=ConversationStage.DISCOVERY)
    with pytest.raises(ValidationError):
        ConversationState(stage=ConversationStage.CLOSED, is_closed=False)
    state = ConversationState(
        stage=ConversationStage.CLOSED,
        is_closed=True,
        close_reason="opt_out",
    )
    assert state.close_reason == "opt_out"
