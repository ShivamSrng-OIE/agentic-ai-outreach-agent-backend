"""Application error hierarchy."""

from __future__ import annotations

from dataclasses import dataclass, field
from http import HTTPStatus


@dataclass(slots=True)
class ErrorDetail:
    field: str | None = None
    message: str = ""


@dataclass(slots=True)
class AppError(Exception):
    message: str
    code: str
    status_code: int = HTTPStatus.INTERNAL_SERVER_ERROR
    details: list[ErrorDetail] = field(default_factory=list)

    def __str__(self) -> str:
        return self.message


class ConfigurationError(AppError):
    def __init__(self, message: str, code: str = "invalid_configuration") -> None:
        super().__init__(message=message, code=code, status_code=HTTPStatus.INTERNAL_SERVER_ERROR)


class ConfigFileNotFoundError(ConfigurationError):
    def __init__(self, message: str) -> None:
        super().__init__(message=message, code="config_file_not_found")


class ConfigYamlParseError(ConfigurationError):
    def __init__(self, message: str) -> None:
        super().__init__(message=message, code="config_yaml_parse_error")


class UnresolvedEnvironmentPlaceholderError(ConfigurationError):
    def __init__(self, message: str) -> None:
        super().__init__(message=message, code="unresolved_environment_placeholder")


class InvalidConfigurationError(ConfigurationError):
    def __init__(self, message: str) -> None:
        super().__init__(message=message, code="invalid_configuration")


class RequestTooLargeError(AppError):
    def __init__(self, message: str = "request body exceeds configured limit") -> None:
        super().__init__(
            message=message,
            code="request_too_large",
            status_code=HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
        )


class ConversationClosedError(AppError):
    def __init__(self, message: str = "conversation is already closed") -> None:
        super().__init__(
            message=message,
            code="conversation_closed",
            status_code=HTTPStatus.CONFLICT,
        )


class TurnLimitReachedError(AppError):
    def __init__(self, message: str = "conversation turn limit reached") -> None:
        super().__init__(
            message=message,
            code="turn_limit_reached",
            status_code=HTTPStatus.CONFLICT,
        )


class InvalidConversationStateError(AppError):
    def __init__(self, message: str) -> None:
        super().__init__(
            message=message,
            code="invalid_conversation_state",
            status_code=HTTPStatus.CONFLICT,
        )


class InvalidCompanyEvidenceError(AppError):
    def __init__(self, message: str) -> None:
        super().__init__(
            message=message,
            code="invalid_company_evidence",
            status_code=HTTPStatus.CONFLICT,
        )


class RetrievalError(AppError):
    def __init__(self, message: str) -> None:
        super().__init__(
            message=message,
            code="retrieval_error",
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
        )


class ModelGatewayError(AppError):
    def __init__(self, message: str, code: str, status_code: int) -> None:
        super().__init__(message=message, code=code, status_code=status_code)


class ModelAuthenticationError(ModelGatewayError):
    def __init__(self, message: str = "model authentication failed") -> None:
        super().__init__(message, "model_authentication_failed", HTTPStatus.BAD_GATEWAY)


class ModelRateLimitError(ModelGatewayError):
    def __init__(self, message: str = "model provider rate limit exceeded") -> None:
        super().__init__(message, "model_rate_limited", HTTPStatus.TOO_MANY_REQUESTS)


class ModelTimeoutError(ModelGatewayError):
    def __init__(self, message: str = "model request timed out") -> None:
        super().__init__(message, "model_timeout", HTTPStatus.SERVICE_UNAVAILABLE)


class ModelConnectionError(ModelGatewayError):
    def __init__(self, message: str = "model connection failed") -> None:
        super().__init__(message, "model_connection_failed", HTTPStatus.SERVICE_UNAVAILABLE)


class ModelUnavailableError(ModelGatewayError):
    def __init__(self, message: str = "model provider unavailable") -> None:
        super().__init__(message, "model_unavailable", HTTPStatus.SERVICE_UNAVAILABLE)


class ModelRefusalError(ModelGatewayError):
    def __init__(self, message: str = "model refused the request") -> None:
        super().__init__(message, "model_refusal", HTTPStatus.BAD_GATEWAY)


class ModelIncompleteResponseError(ModelGatewayError):
    def __init__(self, message: str = "model response was incomplete") -> None:
        super().__init__(message, "model_incomplete_response", HTTPStatus.BAD_GATEWAY)


class ModelInvalidOutputError(ModelGatewayError):
    def __init__(self, message: str = "model output was invalid") -> None:
        super().__init__(message, "model_invalid_output", HTTPStatus.BAD_GATEWAY)


class ModelUnsupportedFeatureError(ModelGatewayError):
    def __init__(self, message: str = "model does not support the requested feature") -> None:
        super().__init__(message, "model_unsupported_feature", HTTPStatus.BAD_GATEWAY)
