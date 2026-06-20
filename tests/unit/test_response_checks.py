"""Tests for deterministic response checks."""

from pathlib import Path

from tests.fixtures.domain import sample_configuration

from psview_agent.agent.response_checks import run_response_checks
from psview_agent.core.config import get_settings
from psview_agent.domain.decisions import AgentDecision
from psview_agent.domain.enums import (
    AgentAction,
    CandidateIntent,
    ConversationStage,
    EngagementLevel,
    Sentiment,
)
from psview_agent.domain.evaluation import GeneratedResponseDraft, SupportedClaim
from psview_agent.domain.retrieval import RetrievedEvidence


def _decision() -> AgentDecision:
    configuration = sample_configuration()
    retrieved = [
        RetrievedEvidence(
            evidence=configuration.evidence_corpus.evidence_facts[0],
            rank=1,
            raw_relevance_score=1.0,
            normalized_relevance=1.0,
            matched_terms=["engineer"],
        )
    ]
    return AgentDecision(
        candidate_intent=CandidateIntent.ASKS_ABOUT_ROLE,
        sentiment=Sentiment.NEUTRAL,
        engagement_level=EngagementLevel.MEDIUM,
        current_stage=ConversationStage.DISCOVERY,
        next_stage=ConversationStage.INFORMATION_EXCHANGE,
        objective="Answer the role question.",
        selected_action=AgentAction.ANSWER_CANDIDATE_QUESTION,
        observed_signals=[],
        candidate_concerns=[],
        retrieved_evidence=retrieved,
        company_fact_ids_to_use=[retrieved[0].evidence.id],
        missing_information=[],
        should_continue=True,
        should_ask_question=False,
        confidence=0.8,
        rationale_summary="Answer directly.",
        policy_overrides=[],
    )


def test_response_checks_flag_multiple_questions(config_path: Path) -> None:
    check = run_response_checks(
        settings=get_settings(),
        decision=_decision(),
        response=GeneratedResponseDraft(
            message="Would this be interesting? Can I share more?",
            supported_claims=[
                SupportedClaim(
                    claim="This role is aimed at product-minded engineers",
                    evidence_fact_ids=["fact_001"],
                )
            ],
        ),
        history=[],
    )
    assert not check.passed
    assert "multiple_questions" in check.violations


def test_response_checks_flag_unretrieved_fact_id(config_path: Path) -> None:
    check = run_response_checks(
        settings=get_settings(),
        decision=_decision(),
        response=GeneratedResponseDraft(
            message="Here is a grounded answer.",
            supported_claims=[
                SupportedClaim(
                    claim="The company offers a grounded answer",
                    evidence_fact_ids=["fact_999"],
                )
            ],
        ),
        history=[],
    )
    assert "unretrieved_evidence_id" in check.violations


def test_response_checks_require_claim_level_attribution_for_grounded_answers(
    config_path: Path,
) -> None:
    check = run_response_checks(
        settings=get_settings(),
        decision=_decision(),
        response=GeneratedResponseDraft(
            message="The role is aimed at strong builders with relevant experience.",
            company_fact_ids_used=["fact_001"],
        ),
        history=[],
    )
    assert "missing_supported_claims" in check.violations


def test_response_checks_reject_unsupported_fit_language(config_path: Path) -> None:
    check = run_response_checks(
        settings=get_settings(),
        decision=_decision(),
        response=GeneratedResponseDraft(
            message="You seem like a strong fit for the role.",
            supported_claims=[
                SupportedClaim(
                    claim="The role is aimed at product-minded engineers",
                    evidence_fact_ids=["fact_001"],
                )
            ],
        ),
        history=[],
    )
    assert "unsupported_fit_language" in check.violations
