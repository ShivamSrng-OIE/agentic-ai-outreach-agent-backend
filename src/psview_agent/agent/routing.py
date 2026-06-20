"""Routing helpers for the LangGraph workflow."""

from psview_agent.agent.graph_state import ConversationGraphState
from psview_agent.domain.evaluation import evaluation_passes


def route_after_deterministic_checks(state: ConversationGraphState, *, max_revisions: int) -> str:
    check = state["deterministic_check"]
    if check.passed:
        return "pass"
    if state["revision_count"] < max_revisions:
        return "revise"
    return "fallback"


def route_after_evaluation(state: ConversationGraphState, *, max_revisions: int) -> str:
    evaluation = state["evaluation"]
    passed, _ = evaluation_passes(evaluation)
    if passed:
        return "pass"
    if state["revision_count"] < max_revisions:
        return "revise"
    return "fallback"
