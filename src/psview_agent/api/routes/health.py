"""Health and readiness routes."""

from typing import Annotated

from fastapi import APIRouter, Depends, Request

from psview_agent.api.dependencies import get_settings
from psview_agent.core.config import Settings
from psview_agent.domain.api import HealthResponse, ReadyResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health(
    settings: Annotated[Settings, Depends(get_settings)],
) -> HealthResponse:
    return HealthResponse(
        status="ok",
        service=settings.app.name,
        version=settings.app.version,
        environment=settings.app.env.value,
    )


@router.get("/ready", response_model=ReadyResponse)
async def ready(request: Request) -> ReadyResponse:
    if not getattr(request.app.state, "ready", False):
        return ReadyResponse(status="starting")
    return ReadyResponse(status="ready")
