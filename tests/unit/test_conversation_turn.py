"""Tests for the conversation turn service."""

from pathlib import Path

import pytest
from tests.fakes.fake_model_gateway import FakeModelGateway
from tests.fixtures.domain import sample_candidate, sample_configuration

from psview_agent.agent.graph import build_conversation_graph
from psview_agent.core.config import get_settings
from psview_agent.core.errors import ConversationClosedError
from psview_agent.domain.conversation import ConversationSession, ConversationState
from psview_agent.domain.enums import ConversationStage
from psview_agent.retrieval.lexical_retriever import LexicalEvidenceRetriever
from psview_agent.services.conversation_start import ConversationStartService
from psview_agent.services.conversation_turn import ConversationTurnService


async def _started_service_and_session(
    config_path: Path,
) -> tuple[ConversationTurnService, ConversationSession]:
    _ = config_path
    settings = get_settings()
    gateway = FakeModelGateway()
    retriever = LexicalEvidenceRetriever(settings.retrieval)
    graph = build_conversation_graph(
        settings=settings,
        gateway=gateway,
        retriever=retriever,
    )
    start_service = ConversationStartService(
        gateway=gateway,
        retriever=retriever,
        retrieval_limit=settings.retrieval.top_k,
    )
    session, _trace = await start_service.start_conversation(
        configuration=sample_configuration(),
        candidate=sample_candidate(),
        target_role="Senior Backend Engineer",
        target_role_description=(
            "Own backend services, integrations, and early platform decisions."
        ),
    )
    turn_service = ConversationTurnService(
        gateway=gateway,
        retriever=retriever,
        graph=graph,
        settings=settings,
    )
    return turn_service, session


@pytest.mark.asyncio
async def test_conversation_turn_processes_stateless_session(config_path: Path) -> None:
    service, session = await _started_service_and_session(config_path)

    response = await service.process_turn(
        session=session,
        candidate_reply="Can you tell me more about the role?",
    )

    assert response.candidate_message.content == "Can you tell me more about the role?"
    assert response.agent_message.content
    assert response.updated_state.turn_count == 1
    assert response.decision_trace.objective
    assert response.evaluation.passed


@pytest.mark.asyncio
async def test_conversation_turn_rejects_closed_session(config_path: Path) -> None:
    service, session = await _started_service_and_session(config_path)
    closed_session = session.model_copy(
        update={
            "state": ConversationState(
                stage=ConversationStage.CLOSED,
                is_closed=True,
                close_reason="opt_out",
            )
        }
    )

    with pytest.raises(ConversationClosedError):
        await service.process_turn(
            session=closed_session,
            candidate_reply="Actually, tell me more.",
        )


@pytest.mark.asyncio
async def test_conversation_turn_fallback_on_gateway_error(config_path: Path) -> None:
    from psview_agent.core.errors import ModelInvalidOutputError
    
    settings = get_settings()
    # Configure a FakeModelGateway to raise ModelInvalidOutputError on next action or response generation
    gateway = FakeModelGateway(scenarios={
        "analyze_candidate_reply": ModelInvalidOutputError("mock validation failure")
    })
    retriever = LexicalEvidenceRetriever(settings.retrieval)
    graph = build_conversation_graph(
        settings=settings,
        gateway=gateway,
        retriever=retriever,
    )
    
    start_service = ConversationStartService(
        gateway=FakeModelGateway(), # start works fine
        retriever=retriever,
        retrieval_limit=settings.retrieval.top_k,
    )
    session, _trace = await start_service.start_conversation(
        configuration=sample_configuration(),
        candidate=sample_candidate(),
        target_role="Senior Backend Engineer",
        target_role_description=(
            "Own backend services, integrations, and early platform decisions."
        ),
    )
    
    # Process turn with the failing gateway
    turn_service = ConversationTurnService(
        gateway=gateway,
        retriever=retriever,
        graph=graph,
        settings=settings,
    )
    
    response = await turn_service.process_turn(
        session=session,
        candidate_reply="Let's continue.",
    )
    
    # Verify fallback response properties
    assert response.agent_message.content == (
        "Thanks for your reply. I want to make sure I provide accurate information. "
        "Let me check the details and get back to you shortly."
    )
    assert response.updated_state.turn_count == session.state.turn_count + 1
    assert response.evaluation.fallback_used is True

