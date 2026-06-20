"""Model client factory."""

from __future__ import annotations

from openai import AsyncOpenAI

from psview_agent.core.config import ModelProvider, Settings


def build_provider_headers(settings: Settings) -> dict[str, str]:
    """Build provider-specific default headers."""
    if settings.model.provider is ModelProvider.OPENROUTER:
        headers: dict[str, str] = {}
        if settings.openrouter.site_url is not None:
            headers["HTTP-Referer"] = str(settings.openrouter.site_url)
        if settings.openrouter.app_name:
            headers["X-OpenRouter-Title"] = settings.openrouter.app_name
        return headers
    return {}


def create_model_client(settings: Settings) -> AsyncOpenAI:
    """Create the shared AsyncOpenAI-compatible client."""
    return AsyncOpenAI(
        api_key=settings.model.api_key.get_secret_value(),
        base_url=str(settings.model.base_url).rstrip("/"),
        timeout=settings.model.timeout_seconds,
        max_retries=settings.model.max_retries,
        default_headers=build_provider_headers(settings),
    )
