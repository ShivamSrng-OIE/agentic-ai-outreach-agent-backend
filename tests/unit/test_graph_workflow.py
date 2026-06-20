"""Tests for the compiled LangGraph workflow."""

from pathlib import Path

import pytest
from tests.fakes.fake_model_gateway import FakeModelGateway
from tests.fixtures.domain import sample_candidate, sample_configuration

from psview_agent.agent.graph import build_conversation_graph
from psview_agent.core.config import get_settings
from psview_agent.domain.conversation import ConversationState
from psview_agent.retrieval.lexical_retriever import LexicalEvidenceRetriever


@pytest.mark.asyncio
async def test_graph_happy_path(config_path: Path) -> None:
    settings = get_settings()
    configuration = sample_configuration()
    candidate = sample_candidate()
    retriever = LexicalEvidenceRetriever(settings.retrieval)
    graph = build_conversation_graph(
        settings=settings,
        gateway=FakeModelGateway(),
        retriever=retriever,
    )
    result = await graph.ainvoke(
        {
            "configuration": configuration,
            "candidate": candidate,
            "target_role": "Senior Backend Engineer",
            "target_role_description": (
                "Own backend services, integrations, and early platform decisions."
            ),
            "conversation_state": ConversationState(),
            "message_history": [],
            "candidate_reply": "Can you tell me more about the role?",
            "revision_count": 0,
            "fallback_used": False,
        },
        config={"recursion_limit": settings.runtime.langgraph_recursion_limit},
    )
    assert result["final_response"].message
    assert result["final_state"].turn_count == 1


@pytest.mark.asyncio
async def test_graph_fallback_path(config_path: Path) -> None:
    settings = get_settings()
    settings = settings.model_copy(
        update={"runtime": settings.runtime.model_copy(update={"max_revision_attempts": 0})}
    )
    configuration = sample_configuration()
    candidate = sample_candidate()
    retriever = LexicalEvidenceRetriever(settings.retrieval)
    graph = build_conversation_graph(
        settings=settings,
        gateway=FakeModelGateway(
            scenarios={
                "generate_candidate_response": lambda **_: {
                    "message": "One? Two?",
                    "company_fact_ids_used": [],
                }
            }
        ),
        retriever=retriever,
    )
    result = await graph.ainvoke(
        {
            "configuration": configuration,
            "candidate": candidate,
            "target_role": "Senior Backend Engineer",
            "target_role_description": (
                "Own backend services, integrations, and early platform decisions."
            ),
            "conversation_state": ConversationState(),
            "message_history": [],
            "candidate_reply": "Can you tell me more about the role?",
            "revision_count": 0,
            "fallback_used": False,
        },
        config={"recursion_limit": settings.runtime.langgraph_recursion_limit},
    )
    assert result["fallback_used"] is True
