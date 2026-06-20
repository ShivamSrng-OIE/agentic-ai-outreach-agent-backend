"""LangGraph workflow assembly."""

from typing import Protocol, cast

from langgraph.graph import END, START, StateGraph

from psview_agent.agent.graph_state import ConversationGraphState
from psview_agent.agent.nodes import (
    analyze_candidate_reply_node,
    build_retrieval_query_node,
    enforce_policies_node,
    evaluate_response_node,
    finalize_node,
    generate_response_node,
    plan_next_action_node,
    retrieve_company_evidence_node,
    revise_response_node,
    run_deterministic_checks_node,
    safe_fallback_node,
    update_conversation_state_node,
)
from psview_agent.agent.routing import route_after_deterministic_checks, route_after_evaluation
from psview_agent.core.config import Settings
from psview_agent.integrations.models.protocol import ModelGateway
from psview_agent.retrieval.protocol import EvidenceRetriever


class CompiledConversationGraph(Protocol):
    """Compiled graph protocol for the turn service."""

    async def ainvoke(
        self,
        input: ConversationGraphState,
        config: dict[str, object] | None = None,
    ) -> ConversationGraphState: ...


def build_conversation_graph(
    *,
    settings: Settings,
    gateway: ModelGateway,
    retriever: EvidenceRetriever,
) -> CompiledConversationGraph:
    """Compile the conversation graph once at startup."""
    graph = StateGraph(ConversationGraphState)

    async def analyze_candidate_reply(state: ConversationGraphState) -> dict[str, object]:
        return await analyze_candidate_reply_node(state, gateway=gateway)

    def update_conversation_state(state: ConversationGraphState) -> dict[str, object]:
        return update_conversation_state_node(state)

    def build_retrieval_query(state: ConversationGraphState) -> dict[str, object]:
        return build_retrieval_query_node(state)

    def retrieve_company_evidence(state: ConversationGraphState) -> dict[str, object]:
        return retrieve_company_evidence_node(state, retriever=retriever, settings=settings)

    async def plan_next_action(state: ConversationGraphState) -> dict[str, object]:
        return await plan_next_action_node(state, gateway=gateway)

    def enforce_policies(state: ConversationGraphState) -> dict[str, object]:
        return enforce_policies_node(state, settings=settings)

    async def generate_response(state: ConversationGraphState) -> dict[str, object]:
        return await generate_response_node(state, gateway=gateway)

    def run_deterministic_checks(state: ConversationGraphState) -> dict[str, object]:
        return run_deterministic_checks_node(state, settings=settings)

    async def evaluate_response(state: ConversationGraphState) -> dict[str, object]:
        return await evaluate_response_node(state, gateway=gateway)

    async def revise_response(state: ConversationGraphState) -> dict[str, object]:
        return await revise_response_node(state, gateway=gateway)

    graph.add_node("analyze_candidate_reply", analyze_candidate_reply)
    graph.add_node("update_conversation_state", update_conversation_state)
    graph.add_node("build_retrieval_query", build_retrieval_query)
    graph.add_node("retrieve_company_evidence", retrieve_company_evidence)
    graph.add_node("plan_next_action", plan_next_action)
    graph.add_node("enforce_policies", enforce_policies)
    graph.add_node("generate_response", generate_response)
    graph.add_node("run_deterministic_checks", run_deterministic_checks)
    graph.add_node("evaluate_response", evaluate_response)
    graph.add_node("revise_response", revise_response)
    graph.add_node("safe_fallback", safe_fallback_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "analyze_candidate_reply")
    graph.add_edge("analyze_candidate_reply", "update_conversation_state")
    graph.add_edge("update_conversation_state", "build_retrieval_query")
    graph.add_edge("build_retrieval_query", "retrieve_company_evidence")
    graph.add_edge("retrieve_company_evidence", "plan_next_action")
    graph.add_edge("plan_next_action", "enforce_policies")
    graph.add_edge("enforce_policies", "generate_response")
    graph.add_edge("generate_response", "run_deterministic_checks")
    graph.add_conditional_edges(
        "run_deterministic_checks",
        lambda state: route_after_deterministic_checks(
            state, max_revisions=settings.runtime.max_revision_attempts
        ),
        {
            "pass": "evaluate_response",
            "revise": "revise_response",
            "fallback": "safe_fallback",
        },
    )
    graph.add_conditional_edges(
        "evaluate_response",
        lambda state: route_after_evaluation(
            state, max_revisions=settings.runtime.max_revision_attempts
        ),
        {
            "pass": "finalize",
            "revise": "revise_response",
            "fallback": "safe_fallback",
        },
    )
    graph.add_edge("revise_response", "run_deterministic_checks")
    graph.add_edge("safe_fallback", "finalize")
    graph.add_edge("finalize", END)
    return cast(CompiledConversationGraph, graph.compile())
