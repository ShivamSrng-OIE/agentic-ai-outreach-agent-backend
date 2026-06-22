"""Tests for workload-aware model routing in the gateway."""

from __future__ import annotations

from typing import cast

import pytest
from openai import AsyncOpenAI
from tests.fixtures.domain import sample_candidate, sample_configuration

from psview_agent.core.config import Settings, default_settings_dict
from psview_agent.domain.agent import OutreachPlanDraft
from psview_agent.domain.conversation import ConversationState
from psview_agent.domain.decisions import (
    AgentDecision,
    AgentDecisionDraft,
    CandidateAnalysis,
)
from psview_agent.domain.enums import (
    AgentAction,
    CandidateIntent,
    ConversationStage,
    EngagementLevel,
    Sentiment,
)
from psview_agent.domain.evaluation import GeneratedResponseDraft, SupportedClaim
from psview_agent.domain.retrieval import RetrievedEvidence
from psview_agent.integrations.models.gateway import (
    ModelWorkload,
    OpenAICompatibleModelGateway,
)


def _settings() -> Settings:
    payload = default_settings_dict()
    model_payload = cast(dict[str, object], payload["model"])
    model_payload["api_key"] = "test-key"
    model_payload["model_name"] = "fallback-model"
    model_payload["general_chat_model_name"] = "google/gemma-4-31b-it:free"
    model_payload["structured_json_model_name"] = "nex-agi/nex-n2-pro:free"
    model_payload["coding_backend_model_name"] = "cohere/north-mini-code:free"
    model_payload["resume_parsing_model_name"] = "meta-llama/llama-3-8b-instruct:free"
    return Settings.model_validate(payload)


def _decision() -> AgentDecision:
    configuration = sample_configuration()
    return AgentDecision(
        candidate_intent=CandidateIntent.ASKS_ABOUT_ROLE,
        sentiment=Sentiment.NEUTRAL,
        engagement_level=EngagementLevel.MEDIUM,
        current_stage=ConversationStage.INFORMATION_EXCHANGE,
        next_stage=ConversationStage.INFORMATION_EXCHANGE,
        objective="Answer the candidate clearly.",
        selected_action=AgentAction.ANSWER_CANDIDATE_QUESTION,
        observed_signals=[],
        candidate_concerns=[],
        retrieved_evidence=[
            RetrievedEvidence(
                evidence=configuration.evidence_corpus.evidence_facts[0],
                rank=1,
                raw_relevance_score=0.9,
                normalized_relevance=1.0,
                matched_terms=["builders"],
            ),
            RetrievedEvidence(
                evidence=configuration.evidence_corpus.evidence_facts[1],
                rank=2,
                raw_relevance_score=0.8,
                normalized_relevance=0.89,
                matched_terms=["communication"],
            ),
        ],
        company_fact_ids_to_use=[configuration.evidence_corpus.evidence_facts[0].id],
        missing_information=[],
        should_continue=True,
        should_ask_question=False,
        confidence=0.9,
        rationale_summary="Keep the response grounded.",
        policy_overrides=[],
    )


def _analysis() -> CandidateAnalysis:
    return CandidateAnalysis(
        intent=CandidateIntent.ASKS_ABOUT_ROLE,
        sentiment=Sentiment.NEUTRAL,
        engagement_level=EngagementLevel.MEDIUM,
        observed_signals=["asked_about_role"],
        expressed_motivations=[],
        candidate_concerns=[],
        questions_or_topics=["role"],
        explicit_opt_out=False,
        reply_summary="The candidate asked for more detail about the role.",
        retrieval_topics=["role"],
        confidence=0.84,
    )


def test_gateway_resolves_named_models_by_workload() -> None:
    gateway = OpenAICompatibleModelGateway(
        client=cast(AsyncOpenAI, object()),
        settings=_settings(),
    )

    assert gateway._resolve_model_name(ModelWorkload.GENERAL_CHAT) == ("google/gemma-4-31b-it:free")
    assert gateway._resolve_model_name(ModelWorkload.STRUCTURED_JSON) == ("nex-agi/nex-n2-pro:free")
    assert gateway._resolve_model_name(ModelWorkload.CODING_BACKEND) == (
        "cohere/north-mini-code:free"
    )
    assert gateway._resolve_model_name(ModelWorkload.RESUME_PARSING) == (
        "meta-llama/llama-3-8b-instruct:free"
    )


def test_generate_response_preserves_empty_fact_ids_when_claims_are_missing() -> None:
    gateway = OpenAICompatibleModelGateway(
        client=cast(AsyncOpenAI, object()),
        settings=_settings(),
    )
    decision = _decision()
    normalized = gateway._normalize_response_fact_ids(
        draft=GeneratedResponseDraft(
            message="Thanks for the question - I'm happy to share more.",
            company_fact_ids_used=[],
        ),
        decision=decision,
    )

    assert normalized.company_fact_ids_used == []


def test_outreach_plan_polishes_metadata_without_backfilling_fact_ids() -> None:
    gateway = OpenAICompatibleModelGateway(
        client=cast(AsyncOpenAI, object()),
        settings=_settings(),
    )
    configuration = sample_configuration()
    retrieved = [
        RetrievedEvidence(
            evidence=configuration.evidence_corpus.evidence_facts[0],
            rank=1,
            raw_relevance_score=0.9,
            normalized_relevance=1.0,
            matched_terms=["builders"],
        ),
        RetrievedEvidence(
            evidence=configuration.evidence_corpus.evidence_facts[1],
            rank=2,
            raw_relevance_score=0.8,
            normalized_relevance=0.89,
            matched_terms=["communication"],
        ),
    ]
    normalized = gateway._normalize_outreach_plan(
        draft=OutreachPlanDraft.model_validate(
            {
                "overall_intent": "engage strong engineers",
                "messages": [
                    {
                        "stage": "initial_outreach",
                        "objective": "introduce the opportunity",
                        "trigger": "candidate_profile",
                        "message": "Hi Casey - your background looks relevant.",
                        "company_fact_ids_used": [],
                    },
                    {
                        "stage": "follow_up",
                        "objective": "follow up respectfully",
                        "trigger": "no_response",
                        "message": "Following up in case this is relevant.",
                        "company_fact_ids_used": [],
                    },
                    {
                        "stage": "final_closeout",
                        "objective": "close the loop",
                        "trigger": "no_response",
                        "message": "I will close the loop for now.",
                        "company_fact_ids_used": [],
                    },
                ],
            }
        ),
        retrieved_evidence=retrieved,
    )

    assert [message.company_fact_ids_used for message in normalized.messages] == [
        [],
        [],
        [],
    ]
    assert normalized.messages[0].objective == (
        "Open a relevant conversation with a specific reason for reaching out."
    )
    assert normalized.messages[0].trigger == "candidate background appears relevant"
    assert normalized.messages[1].objective == "Follow up briefly and politely after no response."
    assert normalized.messages[1].trigger == "no reply to the initial note"
    assert normalized.messages[2].objective == (
        "Close the loop gracefully while leaving the door open."
    )
    assert normalized.messages[2].trigger == "no reply after the follow-up"


@pytest.mark.asyncio
async def test_generate_candidate_response_uses_general_chat_workload() -> None:
    gateway = OpenAICompatibleModelGateway(
        client=cast(AsyncOpenAI, object()),
        settings=_settings(),
    )
    seen_workloads: list[ModelWorkload] = []

    async def fake_request_structured(**kwargs: object) -> GeneratedResponseDraft:
        seen_workloads.append(cast(ModelWorkload, kwargs["workload"]))
        return GeneratedResponseDraft(
            message="Thanks for the question. Here is a grounded answer.",
            supported_claims=[
                SupportedClaim(
                    claim="Here is a grounded answer",
                    evidence_fact_ids=[_decision().company_fact_ids_to_use[0]],
                )
            ],
        )

    gateway._request_structured = fake_request_structured  # type: ignore[assignment]

    await gateway.generate_candidate_response(
        configuration=sample_configuration(),
        candidate=sample_candidate(),
        target_role="Senior Engineer",
        target_role_description="Own backend systems and product integrations in a small team.",
        state=ConversationState(stage=ConversationStage.INFORMATION_EXCHANGE),
        decision=_decision(),
        history=[],
        candidate_reply="Can you share more about the role?",
    )

    assert seen_workloads == [ModelWorkload.GENERAL_CHAT]


@pytest.mark.asyncio
async def test_generate_candidate_response_sanitizes_text_and_derives_ids() -> None:
    gateway = OpenAICompatibleModelGateway(
        client=cast(AsyncOpenAI, object()),
        settings=_settings(),
    )
    decision = _decision()

    async def fake_request_structured(**kwargs: object) -> GeneratedResponseDraft:
        _ = kwargs
        return GeneratedResponseDraft(
            message="Hi Casey\u00a0\u2014 I\u00e2\u20ac\u2122m glad you asked...",
            supported_claims=[
                SupportedClaim(
                    claim="your background looks relevant to the role",
                    evidence_fact_ids=[decision.company_fact_ids_to_use[0]],
                )
            ],
        )

    gateway._request_structured = fake_request_structured  # type: ignore[assignment]

    response = await gateway.generate_candidate_response(
        configuration=sample_configuration(),
        candidate=sample_candidate(),
        target_role="Senior Engineer",
        target_role_description="Own backend systems and product integrations in a small team.",
        state=ConversationState(stage=ConversationStage.INFORMATION_EXCHANGE),
        decision=decision,
        history=[],
        candidate_reply="Can you share more about the role?",
    )

    assert response.message == "Hi Casey - I'm glad you asked..."
    assert response.company_fact_ids_used == [decision.company_fact_ids_to_use[0]]


@pytest.mark.asyncio
async def test_plan_next_action_uses_coding_backend_workload() -> None:
    gateway = OpenAICompatibleModelGateway(
        client=cast(AsyncOpenAI, object()),
        settings=_settings(),
    )
    seen_workloads: list[ModelWorkload] = []

    async def fake_request_structured(**kwargs: object) -> AgentDecisionDraft:
        seen_workloads.append(cast(ModelWorkload, kwargs["workload"]))
        return AgentDecisionDraft(
            current_stage=ConversationStage.INFORMATION_EXCHANGE,
            proposed_next_stage=ConversationStage.INFORMATION_EXCHANGE,
            objective="Answer the direct candidate question.",
            proposed_action=AgentAction.ANSWER_CANDIDATE_QUESTION,
            company_fact_ids_to_use=[],
            missing_information=[],
            should_continue=True,
            should_ask_question=False,
            rationale_summary="Direct questions should be answered first.",
            confidence=0.88,
        )

    gateway._request_structured = fake_request_structured  # type: ignore[assignment]

    await gateway.plan_next_action(
        configuration=sample_configuration(),
        candidate=sample_candidate(),
        target_role="Senior Engineer",
        target_role_description="Own backend systems and product integrations in a small team.",
        state=ConversationState(stage=ConversationStage.INFORMATION_EXCHANGE),
        analysis=_analysis(),
        history=[],
        retrieved_evidence=[],
    )

    assert seen_workloads == [ModelWorkload.CODING_BACKEND]


def test_gateway_repair_recursive_and_padding() -> None:
    gateway = OpenAICompatibleModelGateway(
        client=cast(AsyncOpenAI, object()),
        settings=_settings(),
    )
    # 1. Nested dictionary extra keys and casing repair
    content_nested = """
    {
        "message": "Hello Casey",
        "supported_claims": [
            {
                "claim": "We support remote work and flexible hours",
                "evidence_fact_ids": [],
                "extraFieldNested": "remove me"
            }
        ],
        "extraFieldTop": "remove me too"
    }
    """
    repaired = gateway._attempt_repair(
        content=content_nested,
        errors="validation failed",
        output_model=GeneratedResponseDraft,
    )
    assert repaired is not None
    assert repaired.message == "Hello Casey"
    assert len(repaired.supported_claims) == 1
    # Check that the nested claim had its empty list padded to meet min_length=1
    assert repaired.supported_claims[0].evidence_fact_ids == ["dummy"]
    
    # 2. CandidateAnalysis opt-out rules mismatch repair
    content_analysis = """
    {
        "intent": "do_not_contact",
        "sentiment": "neutral",
        "engagement_level": "medium",
        "explicit_opt_out": false,
        "reply_summary": "Please stop emailing me.",
        "confidence": 0.95
    }
    """
    repaired_analysis = gateway._attempt_repair(
        content=content_analysis,
        errors="validation failed",
        output_model=CandidateAnalysis,
    )
    assert repaired_analysis is not None
    assert repaired_analysis.intent == CandidateIntent.DO_NOT_CONTACT
    # Verify that explicit_opt_out was auto-repaired to True to satisfy rule validation
    assert repaired_analysis.explicit_opt_out is True


def test_gateway_mode_attempts_caching_and_recovery() -> None:
    from psview_agent.core.config import StructuredOutputMode
    gateway = OpenAICompatibleModelGateway(
        client=cast(AsyncOpenAI, object()),
        settings=_settings(),
    )
    
    # Initially cache JSON_OBJECT for fallback-model
    gateway._cached_modes["fallback-model"] = StructuredOutputMode.JSON_OBJECT
    
    # Attempts sequence should prioritize JSON_OBJECT, but include the others
    attempts = gateway._mode_attempts("fallback-model")
    assert attempts[0] == StructuredOutputMode.JSON_OBJECT
    assert len(attempts) > 1
    assert StructuredOutputMode.JSON_SCHEMA in attempts

