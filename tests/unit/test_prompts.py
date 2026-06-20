"""Tests for prompt builders."""

from __future__ import annotations

import json

from tests.fixtures.domain import sample_candidate, sample_company_context, sample_configuration

from psview_agent.domain.conversation import ConversationMessage, ConversationState
from psview_agent.domain.decisions import AgentDecision, CandidateAnalysis
from psview_agent.domain.enums import (
    AgentAction,
    CandidateIntent,
    ConversationStage,
    EngagementLevel,
    MessageRole,
    Sentiment,
)
from psview_agent.domain.evaluation import (
    GeneratedResponseDraft,
    ResponseEvaluation,
    SupportedClaim,
)
from psview_agent.domain.retrieval import RetrievedEvidence
from psview_agent.prompts.action_planning import build_action_planning_prompts
from psview_agent.prompts.candidate_analysis import build_candidate_analysis_prompts
from psview_agent.prompts.common import PROMPT_SECURITY_INSTRUCTION, untrusted_json_block
from psview_agent.prompts.company_configuration import build_company_configuration_prompts
from psview_agent.prompts.outreach_planning import build_outreach_planning_prompts
from psview_agent.prompts.response_evaluation import build_response_evaluation_prompts
from psview_agent.prompts.response_generation import build_response_generation_prompts
from psview_agent.prompts.response_revision import build_response_revision_prompts
from psview_agent.retrieval.query_builder import build_initial_retrieval_query


def _message_history() -> list[ConversationMessage]:
    return [
        ConversationMessage.model_validate(
            {
                "id": "f7150a1a-6b57-40d3-982b-17d89441cb67",
                "role": MessageRole.AGENT,
                "content": "Hi Casey, your background looks relevant.",
                "created_at": "2026-06-19T14:30:00Z",
            }
        )
    ]


def _candidate_analysis() -> CandidateAnalysis:
    return CandidateAnalysis(
        intent=CandidateIntent.ASKS_ABOUT_ROLE,
        sentiment=Sentiment.NEUTRAL,
        engagement_level=EngagementLevel.MEDIUM,
        observed_signals=["asked_for_detail"],
        expressed_motivations=["impact"],
        candidate_concerns=["fit"],
        questions_or_topics=["role"],
        explicit_opt_out=False,
        reply_summary="Candidate asked for more detail about the role.",
        retrieval_topics=["role", "engineering"],
        confidence=0.88,
    )


def _agent_decision(configuration_fact_ids: list[str]) -> AgentDecision:
    return AgentDecision(
        candidate_intent=CandidateIntent.ASKS_ABOUT_ROLE,
        sentiment=Sentiment.NEUTRAL,
        engagement_level=EngagementLevel.MEDIUM,
        current_stage=ConversationStage.INFORMATION_EXCHANGE,
        next_stage=ConversationStage.INFORMATION_EXCHANGE,
        objective="Answer the candidate's direct question clearly.",
        selected_action=AgentAction.ANSWER_CANDIDATE_QUESTION,
        observed_signals=["asked_for_detail"],
        candidate_concerns=["fit"],
        retrieved_evidence=[],
        company_fact_ids_to_use=configuration_fact_ids[:2],
        missing_information=[],
        should_continue=True,
        should_ask_question=False,
        confidence=0.9,
        rationale_summary="Direct candidate questions should be answered first.",
        policy_overrides=["direct_question_priority"],
    )


def _evaluation() -> ResponseEvaluation:
    return ResponseEvaluation(
        personality_consistency=0.9,
        company_grounding=0.9,
        candidate_relevance=0.9,
        action_alignment=0.9,
        conversational_naturalness=0.85,
        repetition_risk=0.1,
        unsupported_claims=[],
        personality_violations=[],
        policy_violations=[],
        passed=True,
        revision_instructions=[],
    )


def test_untrusted_json_block_wraps_json_payload() -> None:
    block = untrusted_json_block({"answer": 1, "items": ["a", "b"]})
    assert block.startswith("<untrusted_input>\n")
    assert block.endswith("\n</untrusted_input>")
    payload = block.removeprefix("<untrusted_input>\n").removesuffix("\n</untrusted_input>")
    assert json.loads(payload)


def test_company_configuration_prompt_contains_context_and_segments() -> None:
    context = sample_company_context()
    configuration = sample_configuration()
    system_prompt, user_prompt = build_company_configuration_prompts(
        context=context,
        source_segments=configuration.evidence_corpus.source_segments,
    )
    assert PROMPT_SECURITY_INSTRUCTION in system_prompt
    assert "Produce a distinct company profile" in system_prompt
    assert "Fill supported optional fields" in system_prompt
    assert context.company_name in user_prompt
    assert configuration.evidence_corpus.source_segments[0].id in user_prompt


def test_outreach_planning_prompt_requires_exactly_three_messages() -> None:
    configuration = sample_configuration()
    candidate = sample_candidate()
    query = build_initial_retrieval_query(
        configuration_context=configuration.company_context,
        candidate=candidate,
        target_role="Senior Engineer",
        target_role_description=(
            "Own backend architecture, product integrations, and cross-functional delivery."
        ),
    )
    retrieved = [
        RetrievedEvidence(
            evidence=fact,
            rank=1,
            raw_relevance_score=1.0,
            normalized_relevance=1.0,
            matched_terms=["engineering"],
        )
        for fact in configuration.evidence_corpus.evidence_facts[:2]
    ]
    system_prompt, user_prompt = build_outreach_planning_prompts(
        configuration=configuration,
        candidate=candidate,
        target_role=query.target_role,
        target_role_description=query.target_role_description,
        retrieved_evidence=retrieved,
    )
    assert PROMPT_SECURITY_INSTRUCTION in system_prompt
    assert "Produce exactly three messages" in system_prompt
    assert "thoughtful recruiter" in system_prompt
    assert "meaningful objective and trigger" in system_prompt
    assert "supported_claims" in system_prompt
    assert "strong fit" in system_prompt
    assert candidate.name in user_prompt
    assert "engineering" in user_prompt
    assert "Own backend architecture" in user_prompt


def test_candidate_analysis_prompt_includes_state_history_and_reply() -> None:
    configuration = sample_configuration()
    candidate = sample_candidate()
    history = _message_history()
    state = ConversationState(stage=ConversationStage.INITIAL_OUTREACH)
    system_prompt, user_prompt = build_candidate_analysis_prompts(
        configuration=configuration,
        candidate=candidate,
        target_role="Senior Engineer",
        target_role_description=(
            "Own backend architecture, product integrations, and cross-functional delivery."
        ),
        state=state,
        history=history,
        candidate_reply="Can you share more about the role?",
    )
    assert "Analyze the candidate reply" in system_prompt
    assert PROMPT_SECURITY_INSTRUCTION in system_prompt
    assert "Can you share more about the role?" in user_prompt
    assert history[0].content in user_prompt


def test_action_planning_prompt_requires_selected_evidence_only() -> None:
    configuration = sample_configuration()
    candidate = sample_candidate()
    history = _message_history()
    analysis = _candidate_analysis()
    retrieved = [
        RetrievedEvidence(
            evidence=fact,
            rank=1,
            raw_relevance_score=0.9,
            normalized_relevance=1.0,
            matched_terms=["role"],
        )
        for fact in configuration.evidence_corpus.evidence_facts[:2]
    ]
    system_prompt, user_prompt = build_action_planning_prompts(
        configuration=configuration,
        candidate=candidate,
        target_role="Senior Engineer",
        target_role_description=(
            "Own backend architecture, product integrations, and cross-functional delivery."
        ),
        state=ConversationState(stage=ConversationStage.INFORMATION_EXCHANGE),
        analysis=analysis,
        history=history,
        retrieved_evidence=retrieved,
    )
    assert "Select only evidence IDs from the retrieved evidence" in system_prompt
    assert "Identify missing information" in system_prompt
    assert "rationale_summary must be specific" in system_prompt
    assert "asked_for_detail" in user_prompt
    assert configuration.evidence_corpus.evidence_facts[0].id in user_prompt


def test_response_generation_prompt_includes_decision_constraints() -> None:
    configuration = sample_configuration()
    candidate = sample_candidate()
    decision = _agent_decision([fact.id for fact in configuration.evidence_corpus.evidence_facts])
    system_prompt, user_prompt = build_response_generation_prompts(
        configuration=configuration,
        candidate=candidate,
        target_role="Senior Engineer",
        target_role_description=(
            "Own backend architecture, product integrations, and cross-functional delivery."
        ),
        state=ConversationState(stage=ConversationStage.INFORMATION_EXCHANGE),
        decision=decision,
        history=_message_history(),
        candidate_reply="Can you share more about the role?",
    )
    assert "ask at most one question" in system_prompt
    assert "supported_claims" in system_prompt
    assert "selected_action" in user_prompt
    assert decision.company_fact_ids_to_use[0] in user_prompt


def test_response_evaluation_prompt_serializes_response() -> None:
    configuration = sample_configuration()
    candidate = sample_candidate()
    decision = _agent_decision([fact.id for fact in configuration.evidence_corpus.evidence_facts])
    response = GeneratedResponseDraft(
        message="Thanks for the question. The role is aimed at strong builders.",
        supported_claims=[
            SupportedClaim(
                claim="The role is aimed at strong builders",
                evidence_fact_ids=decision.company_fact_ids_to_use,
            )
        ],
    )
    system_prompt, user_prompt = build_response_evaluation_prompts(
        configuration=configuration,
        candidate=candidate,
        target_role="Senior Engineer",
        target_role_description=(
            "Own backend architecture, product integrations, and cross-functional delivery."
        ),
        state=ConversationState(stage=ConversationStage.INFORMATION_EXCHANGE),
        decision=decision,
        history=_message_history(),
        candidate_reply="Can you share more about the role?",
        response=response,
    )
    assert "Evaluate the candidate-facing response independently" in system_prompt
    assert response.message in user_prompt
    assert configuration.persona.name in user_prompt


def test_response_revision_prompt_includes_evaluation_and_violations() -> None:
    configuration = sample_configuration()
    candidate = sample_candidate()
    decision = _agent_decision([fact.id for fact in configuration.evidence_corpus.evidence_facts])
    response = GeneratedResponseDraft(
        message="Here is a draft response.",
        supported_claims=[
            SupportedClaim(
                claim="Here is a grounded claim",
                evidence_fact_ids=decision.company_fact_ids_to_use,
            )
        ],
    )
    system_prompt, user_prompt = build_response_revision_prompts(
        configuration=configuration,
        candidate=candidate,
        target_role="Senior Engineer",
        target_role_description=(
            "Own backend architecture, product integrations, and cross-functional delivery."
        ),
        state=ConversationState(stage=ConversationStage.INFORMATION_EXCHANGE),
        decision=decision,
        history=_message_history(),
        candidate_reply="Can you share more about the role?",
        response=response,
        evaluation=_evaluation(),
        deterministic_violations=["too_many_questions"],
    )
    assert "Revise the candidate-facing response once" in system_prompt
    assert "supported_claims" in system_prompt
    assert "too_many_questions" in user_prompt
    assert "personality_consistency" in user_prompt
