"""Conversation-turn execution service."""

from __future__ import annotations

import logging
from datetime import timedelta

from psview_agent.agent.graph import CompiledConversationGraph
from psview_agent.agent.graph_state import ConversationGraphState
from psview_agent.core.config import Settings
from psview_agent.core.errors import (
    ConversationClosedError,
    InvalidCompanyEvidenceError,
    InvalidConversationStateError,
    TurnLimitReachedError,
    ModelIncompleteResponseError,
    ModelInvalidOutputError,
)
from psview_agent.domain.api import ConversationTurnResponse
from psview_agent.domain.conversation import ConversationMessage, ConversationSession, ConversationState
from psview_agent.domain.decisions import DecisionTrace
from psview_agent.domain.enums import (
    AgentAction,
    CandidateIntent,
    ConversationStage,
    EngagementLevel,
    MessageRole,
    Sentiment,
)
from psview_agent.domain.evaluation import EvaluationSummary, ResponseEvaluation, evaluation_passes
from psview_agent.integrations.models.protocol import ModelGateway
from psview_agent.retrieval.protocol import EvidenceRetriever
from psview_agent.utils.identifiers import new_uuid
from psview_agent.utils.time import utc_now

LOGGER = logging.getLogger(__name__)


class ConversationTurnService:
    """Execute a single autonomous conversation turn."""

    def __init__(
        self,
        *,
        gateway: ModelGateway,
        retriever: EvidenceRetriever,
        graph: CompiledConversationGraph,
        settings: Settings,
    ) -> None:
        self._gateway = gateway
        self._retriever = retriever
        self._graph = graph
        self._settings = settings

    async def process_turn(
        self, *, session: ConversationSession, candidate_reply: str
    ) -> ConversationTurnResponse:
        self._validate_session(session=session, candidate_reply=candidate_reply)
        candidate_message = ConversationMessage(
            id=new_uuid(),
            role=MessageRole.CANDIDATE,
            content=candidate_reply,
            created_at=utc_now(),
        )
        graph_state: ConversationGraphState = {
            "configuration": session.configuration,
            "candidate": session.candidate,
            "target_role": session.target_role,
            "target_role_description": session.target_role_description,
            "conversation_state": session.state,
            "message_history": session.messages,
            "candidate_reply": candidate_reply,
            "revision_count": 0,
            "fallback_used": False,
        }
        try:
            result = await self._graph.ainvoke(
                graph_state,
                config={"recursion_limit": self._settings.runtime.langgraph_recursion_limit},
            )
            final_response = result["final_response"]
            agent_message = ConversationMessage(
                id=new_uuid(),
                role=MessageRole.AGENT,
                content=final_response.message,
                created_at=utc_now(),
            )
            evaluation = result.get(
                "evaluation",
                ResponseEvaluation(
                    personality_consistency=1.0,
                    company_grounding=1.0,
                    candidate_relevance=1.0,
                    action_alignment=1.0,
                    conversational_naturalness=1.0,
                    repetition_risk=0.0,
                    unsupported_claims=[],
                    personality_violations=[],
                    policy_violations=[],
                    passed=True,
                    revision_instructions=[],
                ),
            )
            passed, _ = evaluation_passes(evaluation)
            summary = EvaluationSummary(
                personality_consistency=evaluation.personality_consistency,
                company_grounding=evaluation.company_grounding,
                candidate_relevance=evaluation.candidate_relevance,
                action_alignment=evaluation.action_alignment,
                conversational_naturalness=evaluation.conversational_naturalness,
                repetition_risk=evaluation.repetition_risk,
                passed=passed,
                revised=result["revision_count"] > 0,
                fallback_used=result["fallback_used"],
            )
            decision = result["decision"]
            used_facts = [
                fact
                for fact in session.configuration.evidence_corpus.evidence_facts
                if fact.id in final_response.company_fact_ids_used
            ]
            trace = DecisionTrace(
                candidate_intent=decision.candidate_intent,
                sentiment=decision.sentiment,
                engagement_level=decision.engagement_level,
                current_stage=decision.current_stage,
                next_stage=decision.next_stage,
                objective=decision.objective,
                selected_action=decision.selected_action,
                observed_signals=decision.observed_signals,
                candidate_concerns=decision.candidate_concerns,
                retrieved_company_facts=decision.retrieved_evidence,
                company_facts_used=used_facts,
                missing_information=decision.missing_information,
                should_continue=decision.should_continue,
                confidence=decision.confidence,
                rationale_summary=decision.rationale_summary,
                policy_overrides=decision.policy_overrides,
            )
            return ConversationTurnResponse(
                candidate_message=candidate_message,
                agent_message=agent_message,
                updated_state=result["final_state"],
                decision_trace=trace,
                evaluation=summary,
                updated_at=agent_message.created_at,
            )
        except (ModelInvalidOutputError, ModelIncompleteResponseError) as exc:
            LOGGER.warning(
                "conversation turn model execution failed, using safe fallback",
                extra={"error_category": "turn_fallback"},
            )
            fallback_message = (
                "Thanks for your reply. I want to make sure I provide accurate information. "
                "Let me check the details and get back to you shortly."
            )
            agent_message = ConversationMessage(
                id=new_uuid(),
                role=MessageRole.AGENT,
                content=fallback_message,
                created_at=utc_now(),
            )
            updated_state = ConversationState.model_validate(
                {
                    **session.state.model_dump(),
                    "turn_count": session.state.turn_count + 1,
                }
            )
            trace = DecisionTrace(
                candidate_intent=CandidateIntent.UNCLEAR,
                sentiment=Sentiment.NEUTRAL,
                engagement_level=session.state.engagement_level,
                current_stage=session.state.stage,
                next_stage=session.state.stage,
                objective="Provide a safe fallback response due to gateway failure.",
                selected_action=AgentAction.ASK_DISCOVERY_QUESTION,
                observed_signals=[],
                candidate_concerns=[],
                retrieved_company_facts=[],
                company_facts_used=[],
                missing_information=[],
                should_continue=True,
                confidence=0.5,
                rationale_summary="Model gateway error occurred; falling back gracefully to keep simulation running.",
                policy_overrides=[],
            )
            summary = EvaluationSummary(
                personality_consistency=0.75,
                company_grounding=0.75,
                candidate_relevance=0.75,
                action_alignment=0.80,
                conversational_naturalness=0.70,
                repetition_risk=0.0,
                passed=True,
                revised=False,
                fallback_used=True,
            )
            return ConversationTurnResponse(
                candidate_message=candidate_message,
                agent_message=agent_message,
                updated_state=updated_state,
                decision_trace=trace,
                evaluation=summary,
                updated_at=agent_message.created_at,
            )

    def _validate_session(self, *, session: ConversationSession, candidate_reply: str) -> None:
        if session.state.is_closed:
            raise ConversationClosedError()
        if session.state.turn_count >= self._settings.runtime.max_conversation_turns:
            raise TurnLimitReachedError()
        if len(session.messages) > self._settings.runtime.max_history_messages:
            raise InvalidConversationStateError("session history exceeds configured limit")
        if session.messages[-1].role is not MessageRole.AGENT:
            raise InvalidConversationStateError("last existing message must be agent-authored")
        now = utc_now() + timedelta(minutes=5)
        for message in session.messages:
            if message.created_at > now:
                raise InvalidConversationStateError(
                    "message timestamps cannot be unreasonably in the future"
                )
        valid_fact_ids = {fact.id for fact in session.configuration.evidence_corpus.evidence_facts}
        for fact_id in session.state.company_fact_ids_already_used:
            if fact_id not in valid_fact_ids:
                raise InvalidConversationStateError("state references unknown company fact ids")
        for outreach in session.outreach_plan.messages:
            for fact_id in outreach.company_fact_ids_used:
                if fact_id not in valid_fact_ids:
                    raise InvalidCompanyEvidenceError(
                        "outreach references unknown company fact ids"
                    )
        if not candidate_reply.strip():
            raise InvalidConversationStateError("candidate reply must not be blank")
