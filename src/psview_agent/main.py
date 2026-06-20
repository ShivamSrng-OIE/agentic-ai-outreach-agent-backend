"""FastAPI application entrypoint."""

from __future__ import annotations

import uvicorn
from fastapi import FastAPI

from psview_agent.api.router import api_router
from psview_agent.core.exception_handlers import install_exception_handlers
from psview_agent.core.lifespan import build_lifespan
from psview_agent.core.middleware import install_http_middleware


def create_app() -> FastAPI:
    """Create the FastAPI application."""
    app = FastAPI(title="PSVIEW Recruiting Agent API", lifespan=build_lifespan())
    install_http_middleware(app, default_max_request_body_bytes=100000)
    install_exception_handlers(app)
    app.include_router(api_router)
    return app


app = create_app()


def run() -> None:
    """Run the development server."""
    uvicorn.run("psview_agent.main:app", host="0.0.0.0", port=8000, reload=False)
