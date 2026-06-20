"""Agent configuration routes."""

from typing import Annotated

from fastapi import APIRouter, Depends

from psview_agent.api.dependencies import get_agent_configuration_service
from psview_agent.domain.api import ConfigureAgentRequest, ConfigureAgentResponse
from psview_agent.services.agent_configuration import AgentConfigurationService

router = APIRouter(prefix="/api/v1/agents", tags=["agents"])


@router.post("/configure", response_model=ConfigureAgentResponse)
async def configure_agent(
    request: ConfigureAgentRequest,
    service: Annotated[
        AgentConfigurationService,
        Depends(get_agent_configuration_service),
    ],
) -> ConfigureAgentResponse:
    configuration = await service.configure_agent(context=request.company_context)
    return ConfigureAgentResponse(configuration=configuration)
