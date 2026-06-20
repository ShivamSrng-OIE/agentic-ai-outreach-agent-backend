"""Agent configuration routes."""

import os
import json
import logging
import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Header, Request

from psview_agent.api.dependencies import get_agent_configuration_service
from psview_agent.domain.api import ConfigureAgentRequest, ConfigureAgentResponse
from psview_agent.services.agent_configuration import AgentConfigurationService
from psview_agent.api.routes.conversations import get_client_ip, get_ip_location

LOGGER = logging.getLogger("psview_agent.api.routes.agents")

router = APIRouter(prefix="/api/v1/agents", tags=["agents"])


@router.post("/configure", response_model=ConfigureAgentResponse)
async def configure_agent(
    request: ConfigureAgentRequest,
    service: Annotated[
        AgentConfigurationService,
        Depends(get_agent_configuration_service),
    ],
    fastapi_request: Request,
    x_user_id: Annotated[str | None, Header(alias="X-User-ID")] = None,
    x_user_location: Annotated[str | None, Header(alias="X-User-Location")] = None,
) -> ConfigureAgentResponse:
    configuration = await service.configure_agent(context=request.company_context)

    db = fastapi_request.app.state.mongodb
    if db is not None:
        try:
            location = None
            if x_user_location:
                try:
                    location = json.loads(x_user_location)
                except Exception:
                    location = {"raw": x_user_location}

            if not location or not location.get("city"):
                ip = get_client_ip(fastapi_request)
                api_key = os.getenv("IPAPI_KEY")
                resolved_location = await get_ip_location(ip, api_key)
                if resolved_location:
                    location = resolved_location

            await db.interactions.insert_one({
                "user_id": x_user_id,
                "location": location,
                "timestamp": datetime.datetime.now(datetime.timezone.utc),
                "action": "configure",
                "company_context": request.company_context.model_dump(mode="json"),
                "configuration": configuration.model_dump(mode="json")
            })
        except Exception as err:
            LOGGER.warning(f"Failed to log configure interaction: {err}")

    return ConfigureAgentResponse(configuration=configuration)
