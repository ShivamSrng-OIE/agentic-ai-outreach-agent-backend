"""Top-level API router."""

from fastapi import APIRouter

from psview_agent.api.routes import agents, conversations, health

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(agents.router)
api_router.include_router(conversations.router)
