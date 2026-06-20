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


def test_list_validation_truncation_retrieval_query() -> None:
    from psview_agent.domain.retrieval import RetrievalQuery
    query = RetrievalQuery(
        text="Acme backend developer",
        target_role="Developer",
        topics=[f"topic_{i}" for i in range(20)]
    )
    assert len(query.topics) == 12
    assert query.topics == [f"topic_{i}" for i in range(12)]


def test_list_validation_truncation_candidate_analysis() -> None:
    from psview_agent.domain.decisions import CandidateAnalysis
    from psview_agent.domain.enums import CandidateIntent, Sentiment, EngagementLevel
    
    analysis = CandidateAnalysis(
        intent=CandidateIntent.INTERESTED,
        sentiment=Sentiment.POSITIVE,
        engagement_level=EngagementLevel.HIGH,
        observed_signals=[f"signal_{i}" for i in range(15)],
        expressed_motivations=[f"motivation_{i}" for i in range(15)],
        candidate_concerns=[f"concern_{i}" for i in range(15)],
        questions_or_topics=[f"question_{i}" for i in range(15)],
        explicit_opt_out=False,
        reply_summary="Hello, yes I am interested.",
        retrieval_topics=[f"rtopic_{i}" for i in range(15)],
        confidence=0.9
    )
    assert len(analysis.observed_signals) == 8
    assert len(analysis.expressed_motivations) == 8
    assert len(analysis.candidate_concerns) == 8
    assert len(analysis.questions_or_topics) == 8
    assert len(analysis.retrieval_topics) == 8


def test_list_validation_truncation_evaluation() -> None:
    from psview_agent.domain.evaluation import SupportedClaim, GeneratedResponseDraft
    claim = SupportedClaim(
        claim="We support remote work.",
        evidence_fact_ids=[f"fact_{i}" for i in range(15)]
    )
    assert len(claim.evidence_fact_ids) == 8

    draft = GeneratedResponseDraft(
        message="Hello candidate.",
        supported_claims=[
            SupportedClaim(claim="Claim A", evidence_fact_ids=["fact_1"]),
            SupportedClaim(claim="Claim B", evidence_fact_ids=["fact_2"])
        ] * 10,
        company_fact_ids_used=[f"used_{i}" for i in range(15)]
    )
    assert len(draft.supported_claims) == 8
    # Derived from the 8 sliced claims (which only have 'fact_1' and 'fact_2')
    assert len(draft.company_fact_ids_used) == 2

    # Test derived fact truncation
    draft_derived_trunc = GeneratedResponseDraft(
        message="Hello candidate.",
        supported_claims=[
            SupportedClaim(claim=f"Claim {i}", evidence_fact_ids=[f"fact_{i}_1", f"fact_{i}_2"])
            for i in range(5)
        ]
    )
    assert len(draft_derived_trunc.supported_claims) == 5
    assert len(draft_derived_trunc.company_fact_ids_used) == 8


