"""Conversation routes."""

from typing import Annotated

from fastapi import APIRouter, Body, Depends

from psview_agent.api.dependencies import (
    get_agent_configuration_service,
    get_conversation_start_service,
    get_conversation_turn_service,
)
from psview_agent.domain.api import (
    ConversationTurnRequest,
    ConversationTurnResponse,
    StartConversationRequest,
    StartConversationResponse,
)
from psview_agent.services.agent_configuration import AgentConfigurationService
from psview_agent.services.conversation_start import ConversationStartService
from psview_agent.services.conversation_turn import ConversationTurnService

router = APIRouter(prefix="/api/v1/conversations", tags=["conversations"])

START_CONVERSATION_EXAMPLES = {
    "start_from_company_context": {
        "summary": "Easy start request",
        "description": (
            "Let the backend configure the agent and start the conversation in one call."
        ),
        "value": {
            "company_context": {
                "company_name": "Acme AI",
                "company_description": (
                    "Acme builds AI workflow software for teams that need reliable "
                    "automation and thoughtful product delivery."
                ),
                "culture_and_values": (
                    "The team values ownership, clarity, curiosity, and respectful collaboration."
                ),
                "hiring_profiles": (
                    "We hire builders who can ship product, communicate well, and work "
                    "across functions."
                ),
                "communication_tone": "Clear, warm, direct, and specific.",
                "recruiting_intent": (
                    "We want to engage strong product-minded engineers who may be a fit "
                    "for our growth plans."
                ),
                "additional_context": (
                    "The team is focused on real customer problems and thoughtful "
                    "long-term execution."
                ),
            },
            "candidate": {
                "name": "Casey",
                "current_role": "Senior Software Engineer",
                "background_summary": (
                    "Casey has built backend systems, product integrations, and "
                    "internal AI tooling."
                ),
            },
            "target_role": "Senior Engineer",
            "target_role_description": (
                "Own backend architecture, build product integrations, collaborate directly "
                "with product and design, and help shape the engineering foundation for a "
                "growing AI software company."
            ),
        },
    }
}


@router.post(
    "/start",
    response_model=StartConversationResponse,
    summary="Start a conversation",
    description=(
        "Provide either a full configuration object or raw company_context. "
        "Using company_context is the easiest option in the API docs because the "
        "backend will configure the recruiting agent for you before starting the "
        "conversation."
    ),
)
async def start_conversation(
    request: Annotated[
        StartConversationRequest,
        Body(openapi_examples=START_CONVERSATION_EXAMPLES),
    ],
    service: Annotated[
        ConversationStartService,
        Depends(get_conversation_start_service),
    ],
    configuration_service: Annotated[
        AgentConfigurationService,
        Depends(get_agent_configuration_service),
    ],
) -> StartConversationResponse:
    configuration = request.configuration
    if configuration is None:
        assert request.company_context is not None
        configuration = await configuration_service.configure_agent(context=request.company_context)
    session, trace = await service.start_conversation(
        configuration=configuration,
        candidate=request.candidate,
        target_role=request.target_role,
        target_role_description=request.target_role_description,
    )
    return StartConversationResponse(session=session, initial_decision_trace=trace)


@router.post("/turn", response_model=ConversationTurnResponse)
async def conversation_turn(
    request: ConversationTurnRequest,
    service: Annotated[
        ConversationTurnService,
        Depends(get_conversation_turn_service),
    ],
) -> ConversationTurnResponse:
    return await service.process_turn(
        session=request.session,
        candidate_reply=request.candidate_reply,
    )
