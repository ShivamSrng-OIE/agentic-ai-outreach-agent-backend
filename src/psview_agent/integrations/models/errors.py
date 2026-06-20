"""Provider error mapping helpers."""

from __future__ import annotations

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    RateLimitError,
)

from psview_agent.core.errors import (
    ModelAuthenticationError,
    ModelConnectionError,
    ModelIncompleteResponseError,
    ModelRateLimitError,
    ModelRefusalError,
    ModelTimeoutError,
    ModelUnavailableError,
    ModelUnsupportedFeatureError,
)


def map_openai_error(exc: Exception) -> Exception:
    """Map provider exceptions into application errors."""
    if isinstance(exc, AuthenticationError):
        return ModelAuthenticationError()
    if isinstance(exc, RateLimitError):
        return ModelRateLimitError()
    if isinstance(exc, APITimeoutError):
        return ModelTimeoutError()
    if isinstance(exc, APIConnectionError):
        return ModelConnectionError()
    if isinstance(exc, BadRequestError):
        return ModelUnsupportedFeatureError(str(exc))
    if isinstance(exc, APIStatusError):
        if exc.status_code == 502:
            return ModelIncompleteResponseError()
        if exc.status_code in {503, 504}:
            return ModelUnavailableError()
        return ModelRefusalError(str(exc))
    return exc
