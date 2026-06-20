"""Request middleware for request IDs, size limits, and logging."""

from __future__ import annotations

import logging
import re
import time
from contextvars import ContextVar
from typing import Final, cast

from fastapi import FastAPI
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from psview_agent.core.config import Settings
from psview_agent.core.errors import RequestTooLargeError
from psview_agent.domain.api import ErrorResponse
from psview_agent.utils.identifiers import new_request_id

REQUEST_ID_HEADER: Final[str] = "X-Request-ID"
REQUEST_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{8,64}$")
REQUEST_ID_CONTEXT: ContextVar[str] = ContextVar("request_id", default="")
LOGGER = logging.getLogger(__name__)


def set_request_id(request_id: str) -> None:
    REQUEST_ID_CONTEXT.set(request_id)


def get_request_id() -> str:
    request_id = REQUEST_ID_CONTEXT.get()
    return request_id or new_request_id()


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assign a safe request ID and structured request logging."""

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        incoming = request.headers.get(REQUEST_ID_HEADER, "")
        request_id = incoming if REQUEST_ID_PATTERN.fullmatch(incoming) else new_request_id()
        set_request_id(request_id)
        start = time.perf_counter()
        response = await call_next(request)
        response.headers[REQUEST_ID_HEADER] = request_id
        LOGGER.info(
            "request complete",
            extra={
                "method": request.method,
                "route": request.url.path,
                "status_code": response.status_code,
                "duration_ms": round((time.perf_counter() - start) * 1000, 2),
            },
        )
        return response


class MaxRequestBodySizeMiddleware:
    """Enforce a maximum request size without relying only on Content-Length."""

    def __init__(self, app: ASGIApp, *, default_max_bytes: int) -> None:
        self.app = app
        self.default_max_bytes = default_max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        max_bytes = self.default_max_bytes
        app_obj = scope.get("app")
        if isinstance(app_obj, FastAPI) and hasattr(app_obj.state, "settings"):
            settings = cast(Settings, app_obj.state.settings)
            max_bytes = settings.runtime.max_request_body_bytes

        content_length = next(
            (
                int(value.decode("latin-1"))
                for key, value in scope["headers"]
                if key == b"content-length" and value.decode("latin-1").isdigit()
            ),
            None,
        )
        if content_length is not None and content_length > max_bytes:
            request_id = get_request_id()
            payload = ErrorResponse(
                code="request_too_large",
                message="request body exceeds configured limit",
                request_id=request_id,
            )
            response = JSONResponse(status_code=413, content=payload.model_dump())
            response.headers[REQUEST_ID_HEADER] = request_id
            await response(scope, receive, send)
            return

        total = 0

        async def limited_receive() -> Message:
            nonlocal total
            message = await receive()
            if message["type"] == "http.request":
                body = message.get("body", b"")
                total += len(body)
                if total > max_bytes:
                    raise RequestTooLargeError()
            return message

        try:
            await self.app(scope, limited_receive, send)
        except RequestTooLargeError:
            request_id = get_request_id()
            payload = ErrorResponse(
                code="request_too_large",
                message="request body exceeds configured limit",
                request_id=request_id,
            )
            response = JSONResponse(status_code=413, content=payload.model_dump())
            response.headers[REQUEST_ID_HEADER] = request_id
            await response(scope, receive, send)


class DynamicCORSMiddleware(BaseHTTPMiddleware):
    """Apply simple dynamic CORS headers based on loaded settings."""

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        settings = getattr(request.app.state, "settings", None)
        if settings is None:
            return await call_next(request)
        origin = request.headers.get("origin")
        if request.method == "OPTIONS" and origin and origin in settings.runtime.allowed_origins:
            incoming = request.headers.get(REQUEST_ID_HEADER, "")
            request_id = incoming if REQUEST_ID_PATTERN.fullmatch(incoming) else new_request_id()
            set_request_id(request_id)
            response = Response(status_code=204)
            response.headers[REQUEST_ID_HEADER] = request_id
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Credentials"] = "true"
            response.headers["Access-Control-Allow-Methods"] = "*"
            response.headers["Access-Control-Allow-Headers"] = "*"
            response.headers["Vary"] = "Origin"
            return response

        response = await call_next(request)
        if origin and origin in settings.runtime.allowed_origins:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Credentials"] = "true"
            response.headers["Vary"] = "Origin"
        if request.method == "OPTIONS":
            response.headers["Access-Control-Allow-Methods"] = "*"
            response.headers["Access-Control-Allow-Headers"] = "*"
        return response


def install_http_middleware(app: FastAPI, *, default_max_request_body_bytes: int) -> None:
    """Install core HTTP middleware."""
    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(DynamicCORSMiddleware)
    app.add_middleware(
        MaxRequestBodySizeMiddleware,
        default_max_bytes=default_max_request_body_bytes,
    )
