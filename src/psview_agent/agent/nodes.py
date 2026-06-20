"""LangGraph node implementations."""

from psview_agent.agent.fallbacks import build_safe_fallback
from psview_agent.agent.graph_state import ConversationGraphState
from psview_agent.agent.policies import enforce_policies
from psview_agent.agent.response_checks import run_response_checks
from psview_agent.core.config import Settings
from psview_agent.domain.conversation import ConversationState
from psview_agent.domain.evaluation import GeneratedResponseDraft
from psview_agent.integrations.models.protocol import ModelGateway
from psview_agent.retrieval.protocol import EvidenceRetriever
from psview_agent.retrieval.query_builder import build_turn_retrieval_query
from psview_agent.utils.collections import dedupe_casefold


async def analyze_candidate_reply_node(
    state: ConversationGraphState, *, gateway: ModelGateway
) -> dict[str, object]:
    analysis = await gateway.analyze_candidate_reply(
        configuration=state["configuration"],
        candidate=state["candidate"],
        target_role=state["target_role"],
        target_role_description=state["target_role_description"],
        state=state["conversation_state"],
        history=state["message_history"],
        candidate_reply=state["candidate_reply"],
    )
    return {"analysis": analysis}


def update_conversation_state_node(state: ConversationGraphState) -> dict[str, object]:
    analysis = state["analysis"]
    current = state["conversation_state"]
    updated = ConversationState.model_validate(
        {
            **current.model_dump(),
            "turn_count": current.turn_count + 1,
            "engagement_level": analysis.engagement_level,
            "known_motivations": dedupe_casefold(
                [*current.known_motivations, *analysis.expressed_motivations]
            ),
            "known_concerns": dedupe_casefold(
                [*current.known_concerns, *analysis.candidate_concerns]
            ),
            "answered_topics": current.answered_topics,
            "unanswered_topics": dedupe_casefold(
                [*current.unanswered_topics, *analysis.questions_or_topics]
            ),
            "last_candidate_intent": analysis.intent,
        }
    )
    return {"conversation_state": updated}


def build_retrieval_query_node(state: ConversationGraphState) -> dict[str, object]:
    query = build_turn_retrieval_query(
        candidate_reply=state["candidate_reply"],
        analysis=state["analysis"],
        target_role=state["target_role"],
        target_role_description=state["target_role_description"],
        state=state["conversation_state"],
    )
    return {"retrieval_query": query}


def retrieve_company_evidence_node(
    state: ConversationGraphState,
    *,
    retriever: EvidenceRetriever,
    settings: Settings,
) -> dict[str, object]:
    evidence = retriever.retrieve(
        corpus=state["configuration"].evidence_corpus,
        query=state["retrieval_query"],
        already_used_fact_ids=state["conversation_state"].company_fact_ids_already_used,
        limit=settings.retrieval.top_k,
    )
    return {"retrieved_evidence": evidence}


async def plan_next_action_node(
    state: ConversationGraphState, *, gateway: ModelGateway
) -> dict[str, object]:
    draft = await gateway.plan_next_action(
        configuration=state["configuration"],
        candidate=state["candidate"],
        target_role=state["target_role"],
        target_role_description=state["target_role_description"],
        state=state["conversation_state"],
        analysis=state["analysis"],
        history=state["message_history"],
        retrieved_evidence=state["retrieved_evidence"],
    )
    return {"decision_draft": draft}


def enforce_policies_node(
    state: ConversationGraphState,
    *,
    settings: Settings,
) -> dict[str, object]:
    decision = enforce_policies(
        draft=state["decision_draft"],
        analysis=state["analysis"],
        state=state["conversation_state"],
        retrieved_evidence=state["retrieved_evidence"],
        settings=settings,
        evidence_corpus=state["configuration"].evidence_corpus,
    )
    return {"decision": decision}


async def generate_response_node(
    state: ConversationGraphState, *, gateway: ModelGateway
) -> dict[str, object]:
    response = await gateway.generate_candidate_response(
        configuration=state["configuration"],
        candidate=state["candidate"],
        target_role=state["target_role"],
        target_role_description=state["target_role_description"],
        state=state["conversation_state"],
        decision=state["decision"],
        history=state["message_history"],
        candidate_reply=state["candidate_reply"],
    )
    return {"response_draft": response}


def run_deterministic_checks_node(
    state: ConversationGraphState, *, settings: Settings
) -> dict[str, object]:
    check = run_response_checks(
        settings=settings,
        decision=state["decision"],
        response=state["response_draft"],
        history=state["message_history"],
    )
    return {"deterministic_check": check}


async def evaluate_response_node(
    state: ConversationGraphState, *, gateway: ModelGateway
) -> dict[str, object]:
    evaluation = await gateway.evaluate_candidate_response(
        configuration=state["configuration"],
        candidate=state["candidate"],
        target_role=state["target_role"],
        target_role_description=state["target_role_description"],
        state=state["conversation_state"],
        decision=state["decision"],
        history=state["message_history"],
        candidate_reply=state["candidate_reply"],
        response=state["response_draft"],
    )
    return {"evaluation": evaluation}


async def revise_response_node(
    state: ConversationGraphState, *, gateway: ModelGateway
) -> dict[str, object]:
    deterministic_violations: list[str] = []
    if "deterministic_check" in state:
        deterministic_violations = state["deterministic_check"].violations
    revised = await gateway.revise_candidate_response(
        configuration=state["configuration"],
        candidate=state["candidate"],
        target_role=state["target_role"],
        target_role_description=state["target_role_description"],
        state=state["conversation_state"],
        decision=state["decision"],
        history=state["message_history"],
        candidate_reply=state["candidate_reply"],
        response=state["response_draft"],
        evaluation=state.get("evaluation"),
        deterministic_violations=deterministic_violations,
    )
    return {"response_draft": revised, "revision_count": state["revision_count"] + 1}


def safe_fallback_node(state: ConversationGraphState) -> dict[str, object]:
    fallback = build_safe_fallback(decision=state["decision"])
    return {"response_draft": fallback, "fallback_used": True}


def finalize_node(state: ConversationGraphState) -> dict[str, object]:
    final_response: GeneratedResponseDraft = state["response_draft"]
    current = state["conversation_state"]
    updated_state = ConversationState.model_validate(
        {
            **current.model_dump(),
            "stage": state["decision"].next_stage,
            "company_fact_ids_already_used": dedupe_casefold(
                [
                    *current.company_fact_ids_already_used,
                    *final_response.company_fact_ids_used,
                ]
            ),
            "last_action": state["decision"].selected_action,
            "answered_topics": dedupe_casefold(
                [
                    *current.answered_topics,
                    *state["analysis"].questions_or_topics,
                ]
            ),
            "unanswered_topics": [
                topic
                for topic in current.unanswered_topics
                if topic.casefold()
                not in {item.casefold() for item in state["analysis"].questions_or_topics}
            ],
            "is_closed": state["decision"].next_stage.value == "closed",
            "close_reason": (
                "policy_exit" if state["decision"].next_stage.value == "closed" else None
            ),
        }
    )
    return {"final_response": final_response, "final_state": updated_state}
