"""Tests for deterministic policies."""

from pathlib import Path

from tests.fixtures.domain import sample_configuration

from psview_agent.agent.policies import enforce_policies
from psview_agent.core.config import get_settings
from psview_agent.domain.conversation import ConversationState
from psview_agent.domain.decisions import AgentDecisionDraft, CandidateAnalysis
from psview_agent.domain.enums import (
    AgentAction,
    CandidateIntent,
    ConversationStage,
    EngagementLevel,
    Sentiment,
)


def _draft() -> AgentDecisionDraft:
    return AgentDecisionDraft(
        current_stage=ConversationStage.DISCOVERY,
        proposed_next_stage=ConversationStage.INFORMATION_EXCHANGE,
        objective="Continue the conversation.",
        proposed_action=AgentAction.ASK_DISCOVERY_QUESTION,
        company_fact_ids_to_use=[],
        missing_information=[],
        should_continue=True,
        should_ask_question=True,
        rationale_summary="A default draft decision.",
        confidence=0.8,
    )


def test_policy_for_explicit_opt_out(config_path: Path) -> None:
    configuration = sample_configuration()
    analysis = CandidateAnalysis(
        intent=CandidateIntent.DO_NOT_CONTACT,
        sentiment=Sentiment.NEGATIVE,
        engagement_level=EngagementLevel.LOW,
        observed_signals=[],
        expressed_motivations=[],
        candidate_concerns=[],
        questions_or_topics=[],
        explicit_opt_out=True,
        reply_summary="Do not contact me again.",
        retrieval_topics=[],
        confidence=0.9,
    )
    decision = enforce_policies(
        draft=_draft(),
        analysis=analysis,
        state=ConversationState(),
        retrieved_evidence=[],
        settings=get_settings(),
        evidence_corpus=configuration.evidence_corpus,
    )
    assert decision.selected_action.value == "gracefully_exit"
    assert decision.next_stage.value == "closed"


def test_policy_for_missing_compensation(config_path: Path) -> None:
    configuration = sample_configuration()
    analysis = CandidateAnalysis(
        intent=CandidateIntent.ASKS_ABOUT_COMPENSATION,
        sentiment=Sentiment.NEUTRAL,
        engagement_level=EngagementLevel.MEDIUM,
        observed_signals=[],
        expressed_motivations=[],
        candidate_concerns=[],
        questions_or_topics=["compensation"],
        explicit_opt_out=False,
        reply_summary="What is the salary?",
        retrieval_topics=["compensation"],
        confidence=0.9,
    )
    decision = enforce_policies(
        draft=_draft(),
        analysis=analysis,
        state=ConversationState(),
        retrieved_evidence=[],
        settings=get_settings(),
        evidence_corpus=configuration.evidence_corpus,
    )
    assert decision.selected_action.value == "clarify_missing_information"
    assert "compensation" in decision.missing_information
