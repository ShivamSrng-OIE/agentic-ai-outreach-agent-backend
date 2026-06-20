"""FastAPI dependencies."""

from typing import cast

from fastapi import Request

from psview_agent.core.config import Settings
from psview_agent.integrations.models.protocol import ModelGateway
from psview_agent.retrieval.protocol import EvidenceRetriever
from psview_agent.services.agent_configuration import AgentConfigurationService
from psview_agent.services.conversation_start import ConversationStartService
from psview_agent.services.conversation_turn import ConversationTurnService


def get_settings(request: Request) -> Settings:
    return cast(Settings, request.app.state.settings)


def get_model_gateway(request: Request) -> ModelGateway:
    return cast(ModelGateway, request.app.state.model_gateway)


def get_retriever(request: Request) -> EvidenceRetriever:
    return cast(EvidenceRetriever, request.app.state.retriever)


def get_agent_configuration_service(request: Request) -> AgentConfigurationService:
    return cast(AgentConfigurationService, request.app.state.agent_configuration_service)


def get_conversation_start_service(request: Request) -> ConversationStartService:
    return cast(ConversationStartService, request.app.state.conversation_start_service)


def get_conversation_turn_service(request: Request) -> ConversationTurnService:
    return cast(ConversationTurnService, request.app.state.conversation_turn_service)
