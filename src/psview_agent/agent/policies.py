"""Deterministic policy enforcement."""

from __future__ import annotations

from collections.abc import Sequence

from psview_agent.core.config import Settings
from psview_agent.domain.company import EvidenceCorpus
from psview_agent.domain.conversation import ConversationState
from psview_agent.domain.decisions import AgentDecision, AgentDecisionDraft, CandidateAnalysis
from psview_agent.domain.enums import AgentAction, CandidateIntent, ConversationStage
from psview_agent.domain.retrieval import RetrievedEvidence


def _has_support(retrieved_evidence: Sequence[RetrievedEvidence], topic: str) -> bool:
    topic_key = topic.casefold()
    for item in retrieved_evidence:
        if topic_key in item.evidence.fact.casefold():
            return True
        if any(topic_key in tag.casefold() for tag in item.evidence.retrieval_tags):
            return True
    return False


def _build_rationale_summary(
    *,
    analysis: CandidateAnalysis,
    selected_action: AgentAction,
    selected_ids: Sequence[str],
    missing_information: Sequence[str],
) -> str:
    topic = (
        analysis.questions_or_topics[0] if analysis.questions_or_topics else "their latest reply"
    )
    if missing_information:
        missing = ", ".join(missing_information[:2])
        return (
            f"The candidate asked about {topic}, but confirmed support is missing for {missing}, "
            "so the reply should acknowledge that gap directly."
        )
    if selected_ids:
        evidence_ids = ", ".join(selected_ids[:3])
        return (
            f"The candidate asked about {topic}, so {selected_action.value} should use "
            f"{evidence_ids} to keep the reply specific and grounded."
        )
    return (
        f"The candidate asked about {topic}, so {selected_action.value} should stay concise "
        "and avoid unsupported claims."
    )


def enforce_policies(
    *,
    draft: AgentDecisionDraft,
    analysis: CandidateAnalysis,
    state: ConversationState,
    retrieved_evidence: Sequence[RetrievedEvidence],
    settings: Settings,
    evidence_corpus: EvidenceCorpus,
) -> AgentDecision:
    """Apply deterministic overrides to the model's draft decision."""
    retrieved_ids = {item.evidence.id for item in retrieved_evidence}
    selected_ids = [
        fact_id for fact_id in draft.company_fact_ids_to_use if fact_id in retrieved_ids
    ]
    overrides: list[str] = []
    selected_action = draft.proposed_action
    next_stage = draft.proposed_next_stage
    should_continue = draft.should_continue
    should_ask_question = draft.should_ask_question
    missing_information = list(draft.missing_information)

    if analysis.explicit_opt_out or analysis.intent is CandidateIntent.DO_NOT_CONTACT:
        selected_action = AgentAction.GRACEFULLY_EXIT
        next_stage = ConversationStage.CLOSED
        should_continue = False
        should_ask_question = False
        overrides.append("explicit_opt_out")
    elif analysis.intent is CandidateIntent.CLEAR_REJECTION:
        selected_action = AgentAction.GRACEFULLY_EXIT
        next_stage = ConversationStage.CLOSED
        should_continue = False
        should_ask_question = False
        overrides.append("clear_rejection")
    elif analysis.intent is CandidateIntent.BUSY_OR_NOT_READY:
        selected_action = AgentAction.PAUSE_RESPECTFULLY
        next_stage = ConversationStage.PAUSED
        should_continue = True
        should_ask_question = False
        overrides.append("busy_or_not_ready")
    elif analysis.intent is CandidateIntent.HOSTILE:
        selected_action = AgentAction.GRACEFULLY_EXIT
        next_stage = ConversationStage.CLOSED
        should_continue = False
        should_ask_question = False
        overrides.append("hostility")
    elif analysis.intent is CandidateIntent.ASKS_IF_AI_OR_AUTOMATED:
        selected_action = AgentAction.DISCLOSE_AI_IDENTITY
        should_ask_question = False
        overrides.append("ai_identity")
    elif analysis.intent in {
        CandidateIntent.ASKS_ABOUT_COMPENSATION,
        CandidateIntent.ASKS_ABOUT_VISA_OR_ELIGIBILITY,
        CandidateIntent.ASKS_ABOUT_LOCATION_OR_WORK_MODE,
    }:
        topic_map = {
            CandidateIntent.ASKS_ABOUT_COMPENSATION: "compensation",
            CandidateIntent.ASKS_ABOUT_VISA_OR_ELIGIBILITY: "visa",
            CandidateIntent.ASKS_ABOUT_LOCATION_OR_WORK_MODE: "location",
        }
        missing_topic = topic_map[analysis.intent]
        if not _has_support(retrieved_evidence, missing_topic):
            selected_action = AgentAction.CLARIFY_MISSING_INFORMATION
            should_ask_question = False
            if missing_topic not in missing_information:
                missing_information.append(missing_topic)
            overrides.append(f"missing_{missing_topic}")

    if analysis.questions_or_topics and selected_action is AgentAction.ASK_DISCOVERY_QUESTION:
        selected_action = AgentAction.ANSWER_CANDIDATE_QUESTION
        overrides.append("direct_question_priority")

    if state.turn_count >= settings.runtime.max_conversation_turns:
        selected_action = AgentAction.PAUSE_RESPECTFULLY
        next_stage = ConversationStage.PAUSED
        should_continue = False
        should_ask_question = False
        overrides.append("turn_limit")

    valid_fact_ids = {fact.id for fact in evidence_corpus.evidence_facts}
    selected_ids = [fact_id for fact_id in selected_ids if fact_id in valid_fact_ids]
    return AgentDecision(
        candidate_intent=analysis.intent,
        sentiment=analysis.sentiment,
        engagement_level=analysis.engagement_level,
        current_stage=draft.current_stage,
        next_stage=next_stage,
        objective=draft.objective,
        selected_action=selected_action,
        observed_signals=analysis.observed_signals,
        candidate_concerns=analysis.candidate_concerns,
        retrieved_evidence=list(retrieved_evidence),
        company_fact_ids_to_use=selected_ids,
        missing_information=missing_information,
        should_continue=should_continue,
        should_ask_question=should_ask_question,
        confidence=draft.confidence,
        rationale_summary=_build_rationale_summary(
            analysis=analysis,
            selected_action=selected_action,
            selected_ids=selected_ids,
            missing_information=missing_information,
        ),
        policy_overrides=overrides,
    )
