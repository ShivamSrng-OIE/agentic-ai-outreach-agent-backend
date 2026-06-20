"""Global exception handlers."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import cast

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from psview_agent.core.errors import AppError
from psview_agent.core.middleware import get_request_id
from psview_agent.domain.api import ErrorDetail, ErrorResponse

LOGGER = logging.getLogger(__name__)


def _error_response(app_error: AppError) -> ErrorResponse:
    return ErrorResponse(
        code=app_error.code,
        message=app_error.message,
        request_id=get_request_id(),
        details=[ErrorDetail(field=item.field, message=item.message) for item in app_error.details],
    )


async def handle_app_error(_: Request, exc: AppError) -> JSONResponse:
    LOGGER.warning(
        "application error",
        extra={"error_category": exc.code, "status_code": exc.status_code},
    )
    return JSONResponse(status_code=exc.status_code, content=_error_response(exc).model_dump())


async def handle_validation_error(
    _: Request, exc: RequestValidationError | ValidationError
) -> JSONResponse:
    details = [
        ErrorDetail(
            field=".".join(str(part) for part in error["loc"]),
            message=error["msg"],
        )
        for error in exc.errors()
    ]
    payload = ErrorResponse(
        code="validation_error",
        message="request validation failed",
        request_id=get_request_id(),
        details=details,
    )
    return JSONResponse(status_code=422, content=payload.model_dump())


async def handle_unexpected_error(_: Request, exc: Exception) -> JSONResponse:
    LOGGER.exception("unexpected error", extra={"error_category": "internal_error"})
    payload = ErrorResponse(
        code="internal_error",
        message="internal server error",
        request_id=get_request_id(),
    )
    return JSONResponse(status_code=500, content=payload.model_dump())


def install_exception_handlers(app: FastAPI) -> None:
    """Register global exception handlers."""
    request_handler = cast(
        Callable[[Request, Exception], Awaitable[JSONResponse]],
        handle_app_error,
    )
    validation_handler = cast(
        Callable[[Request, Exception], Awaitable[JSONResponse]],
        handle_validation_error,
    )
    unexpected_handler = cast(
        Callable[[Request, Exception], Awaitable[JSONResponse]],
        handle_unexpected_error,
    )
    app.add_exception_handler(AppError, request_handler)
    app.add_exception_handler(RequestValidationError, validation_handler)
    app.add_exception_handler(ValidationError, validation_handler)
    app.add_exception_handler(Exception, unexpected_handler)
