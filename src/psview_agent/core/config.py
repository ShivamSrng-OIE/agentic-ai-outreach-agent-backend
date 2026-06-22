"""Typed settings models and cached accessors."""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from contextvars import ContextVar

from pydantic import AnyHttpUrl, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from psview_agent.domain.base import JsonValue, StrictModel


class AppEnvironment(StrEnum):
    DEVELOPMENT = "development"
    TEST = "test"
    PRODUCTION = "production"


class ModelProvider(StrEnum):
    OPENROUTER = "openrouter"
    NVIDIA = "nvidia"
    GEMINI = "gemini"
    OPENAI = "openai"


class StructuredOutputMode(StrEnum):
    AUTO = "auto"
    JSON_SCHEMA = "json_schema"
    JSON_OBJECT = "json_object"
    PROMPT_JSON = "prompt_json"


class ModelOverride(StrictModel):
    provider: ModelProvider
    api_key: str
    model_name: str
    general_chat_model_name: str | None = None
    structured_json_model_name: str | None = None
    coding_backend_model_name: str | None = None
    resume_parsing_model_name: str | None = None


model_override_var: ContextVar[ModelOverride | None] = ContextVar("model_override", default=None)


class AppSettings(StrictModel):
    name: str = Field(min_length=1, max_length=120)
    env: AppEnvironment
    version: str = Field(min_length=1, max_length=50)
    api_v1_prefix: str = Field(min_length=1, max_length=50)
    log_level: str = Field(min_length=1, max_length=20)


class ModelSettings(StrictModel):
    provider: ModelProvider
    api_key: SecretStr
    base_url: AnyHttpUrl
    model_name: str = Field(min_length=1, max_length=200)
    general_chat_model_name: str | None = Field(default=None, min_length=1, max_length=200)
    structured_json_model_name: str | None = Field(default=None, min_length=1, max_length=200)
    coding_backend_model_name: str | None = Field(default=None, min_length=1, max_length=200)
    resume_parsing_model_name: str | None = Field(default=None, min_length=1, max_length=200)
    structured_output_mode: StructuredOutputMode
    timeout_seconds: float = Field(gt=0, le=300)
    max_retries: int = Field(ge=0, le=10)
    max_output_tokens: int = Field(ge=100, le=8192)
    temperature: float = Field(ge=0, le=1.5)
    repair_attempts: int = Field(ge=0, le=3)
    concurrency_limit: int = Field(ge=1, le=32)
    extra_body: dict[str, JsonValue] = Field(default_factory=dict)


class OpenRouterSettings(StrictModel):
    site_url: AnyHttpUrl | None = None
    app_name: str | None = None


class RuntimeSettings(StrictModel):
    allowed_origins: list[str] = Field(min_length=1)
    max_request_body_bytes: int = Field(ge=1024, le=5_000_000)
    max_history_messages: int = Field(ge=1, le=200)
    max_conversation_turns: int = Field(ge=1, le=100)
    max_response_characters: int = Field(ge=50, le=5000)
    max_revision_attempts: int = Field(ge=0, le=3)
    langgraph_recursion_limit: int = Field(ge=5, le=100)


class RetrievalSettings(StrictModel):
    enabled: bool = True
    top_k: int = Field(ge=1, le=20)
    min_score: float = Field(ge=0, le=10)
    reuse_penalty: float = Field(ge=0, le=2)
    max_fact_candidates: int = Field(ge=1, le=100)


class DatabaseSettings(StrictModel):
    mongodb_uri: str = Field(min_length=1)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="forbid", frozen=True)

    app: AppSettings
    model: ModelSettings
    openrouter: OpenRouterSettings
    runtime: RuntimeSettings
    retrieval: RetrievalSettings
    database: DatabaseSettings



    @model_validator(mode="after")
    def validate_provider_consistency(self) -> Settings:
        if self.model.provider is ModelProvider.OPENROUTER and "openrouter.ai" not in str(
            self.model.base_url
        ):
            raise ValueError("OpenRouter provider requires an OpenRouter-compatible base URL")
        if self.app.env is AppEnvironment.PRODUCTION and "*" in self.runtime.allowed_origins:
            raise ValueError("wildcard CORS origin is not allowed in production")
        return self


def default_settings_dict() -> dict[str, object]:
    """Return hardcoded application defaults."""
    return {
        "app": {
            "name": "PSVIEW Recruiting Agent API",
            "env": "development",
            "version": "0.1.0",
            "api_v1_prefix": "/api/v1",
            "log_level": "INFO",
        },
        "model": {
            "provider": "openrouter",
            "api_key": "${MODEL_API_KEY}",
            "base_url": "https://openrouter.ai/api/v1",
            "model_name": "${MODEL_NAME}",
            "general_chat_model_name": "openai/gpt-oss-120b:free",
            "structured_json_model_name": "openai/gpt-oss-120b:free",
            "coding_backend_model_name": "openai/gpt-oss-120b:free",
            "resume_parsing_model_name": "openai/gpt-oss-120b:free",
            "structured_output_mode": "auto",
            "timeout_seconds": 45,
            "max_retries": 2,
            "max_output_tokens": 1800,
            "temperature": 0.2,
            "repair_attempts": 1,
            "concurrency_limit": 4,
            "extra_body": {},
        },
        "openrouter": {
            "site_url": None,
            "app_name": "PSVIEW Recruiting Agent",
        },
        "runtime": {
            "allowed_origins": ["http://localhost:5173"],
            "max_request_body_bytes": 100000,
            "max_history_messages": 20,
            "max_conversation_turns": 16,
            "max_response_characters": 1000,
            "max_revision_attempts": 1,
            "langgraph_recursion_limit": 20,
        },
        "retrieval": {
            "enabled": True,
            "top_k": 5,
            "min_score": 0.05,
            "reuse_penalty": 0.15,
            "max_fact_candidates": 20,
        },
        "database": {
            "mongodb_uri": "mongodb://localhost:27017/hirewire",
        },
    }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load and cache settings."""
    from psview_agent.core.config_loader import load_settings

    return load_settings().settings


def clear_settings_cache() -> None:
    """Clear the cached settings instance."""
    get_settings.cache_clear()
