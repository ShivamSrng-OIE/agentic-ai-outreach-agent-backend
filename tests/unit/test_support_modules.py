"""Tests for support modules with deterministic behavior."""

from __future__ import annotations

import asyncio
from typing import cast

from openai import AsyncOpenAI

from psview_agent.agent.fallbacks import build_safe_fallback
from psview_agent.core.config import (
    ModelProvider,
    Settings,
    StructuredOutputMode,
    default_settings_dict,
)
from psview_agent.core.errors import (
    ConfigFileNotFoundError,
    ConfigYamlParseError,
    ConversationClosedError,
    InvalidCompanyEvidenceError,
    InvalidConfigurationError,
    InvalidConversationStateError,
    ModelAuthenticationError,
    ModelConnectionError,
    ModelIncompleteResponseError,
    ModelInvalidOutputError,
    ModelRateLimitError,
    ModelRefusalError,
    ModelTimeoutError,
    ModelUnavailableError,
    ModelUnsupportedFeatureError,
    RequestTooLargeError,
    RetrievalError,
    TurnLimitReachedError,
    UnresolvedEnvironmentPlaceholderError,
)
from psview_agent.domain.decisions import AgentDecision
from psview_agent.domain.enums import (
    AgentAction,
    CandidateIntent,
    ConversationStage,
    EngagementLevel,
    Sentiment,
)
from psview_agent.integrations.models.client import build_provider_headers, create_model_client
from psview_agent.integrations.models.structured_output import (
    build_response_format,
    is_unsupported_format_error,
    mode_sequence,
    prompt_json_instructions,
)
from psview_agent.utils.collections import (
    dedupe_casefold,
    dedupe_preserving_order,
    unique_sequence,
)
from psview_agent.utils.text import (
    normalize_whitespace,
    remove_empty_strings,
    repair_common_mojibake,
    safe_truncate,
    sanitize_generated_text,
    split_paragraphs,
    split_sentences,
)


def _settings(*, provider: ModelProvider = ModelProvider.OPENROUTER) -> Settings:
    payload = default_settings_dict()
    model = dict(cast(dict[str, object], payload["model"]))
    model["api_key"] = "test-key"
    model["model_name"] = "test-model"
    model["provider"] = provider.value
    if provider is ModelProvider.NVIDIA:
        model["base_url"] = "https://integrate.api.nvidia.com/v1"
    payload["model"] = model
    if provider is ModelProvider.NVIDIA:
        payload["openrouter"] = {"site_url": None, "app_name": None}
    else:
        payload["openrouter"] = {
            "site_url": "https://example.com",
            "app_name": "PSVIEW Recruiting Agent",
        }
    return Settings.model_validate(payload)


def _decision(
    *,
    action: AgentAction = AgentAction.ANSWER_CANDIDATE_QUESTION,
    missing_information: list[str] | None = None,
) -> AgentDecision:
    return AgentDecision(
        candidate_intent=CandidateIntent.ASKS_ABOUT_ROLE,
        sentiment=Sentiment.NEUTRAL,
        engagement_level=EngagementLevel.MEDIUM,
        current_stage=ConversationStage.INFORMATION_EXCHANGE,
        next_stage=ConversationStage.INFORMATION_EXCHANGE,
        objective="Answer the candidate clearly.",
        selected_action=action,
        observed_signals=[],
        candidate_concerns=[],
        retrieved_evidence=[],
        company_fact_ids_to_use=[],
        missing_information=missing_information or [],
        should_continue=True,
        should_ask_question=False,
        confidence=0.9,
        rationale_summary="Keep the reply grounded.",
        policy_overrides=[],
    )


def test_build_safe_fallback_covers_all_branches() -> None:
    exit_response = build_safe_fallback(
        decision=_decision(action=AgentAction.GRACEFULLY_EXIT),
    )
    disclose_response = build_safe_fallback(
        decision=_decision(action=AgentAction.DISCLOSE_AI_IDENTITY),
    )
    missing_response = build_safe_fallback(
        decision=_decision(missing_information=["compensation"]),
    )
    generic_response = build_safe_fallback(decision=_decision())

    assert "not contact you again" in exit_response.message
    assert "AI recruiting assistant" in disclose_response.message
    assert "compensation" in missing_response.message
    assert "avoid guessing" in generic_response.message


def test_core_error_types_expose_expected_codes() -> None:
    errors = [
        (ConfigFileNotFoundError("missing"), "config_file_not_found"),
        (ConfigYamlParseError("bad yaml"), "config_yaml_parse_error"),
        (
            UnresolvedEnvironmentPlaceholderError("missing variable"),
            "unresolved_environment_placeholder",
        ),
        (InvalidConfigurationError("invalid"), "invalid_configuration"),
        (RequestTooLargeError(), "request_too_large"),
        (ConversationClosedError(), "conversation_closed"),
        (TurnLimitReachedError(), "turn_limit_reached"),
        (
            InvalidConversationStateError("bad state"),
            "invalid_conversation_state",
        ),
        (InvalidCompanyEvidenceError("bad evidence"), "invalid_company_evidence"),
        (RetrievalError("boom"), "retrieval_error"),
        (ModelAuthenticationError(), "model_authentication_failed"),
        (ModelRateLimitError(), "model_rate_limited"),
        (ModelTimeoutError(), "model_timeout"),
        (ModelConnectionError(), "model_connection_failed"),
        (ModelUnavailableError(), "model_unavailable"),
        (ModelRefusalError(), "model_refusal"),
        (ModelIncompleteResponseError(), "model_incomplete_response"),
        (ModelInvalidOutputError(), "model_invalid_output"),
        (ModelUnsupportedFeatureError("unsupported"), "model_unsupported_feature"),
    ]

    for error, expected_code in errors:
        assert error.code == expected_code
        assert str(error)


def test_structured_output_helpers_cover_modes_and_formats() -> None:
    openrouter_modes = mode_sequence(ModelProvider.OPENROUTER, StructuredOutputMode.AUTO)
    nvidia_modes = mode_sequence(ModelProvider.NVIDIA, StructuredOutputMode.AUTO)
    explicit_mode = mode_sequence(
        ModelProvider.OPENROUTER,
        StructuredOutputMode.JSON_OBJECT,
    )

    assert openrouter_modes == [
        StructuredOutputMode.JSON_SCHEMA,
        StructuredOutputMode.JSON_OBJECT,
        StructuredOutputMode.PROMPT_JSON,
    ]
    assert nvidia_modes == [
        StructuredOutputMode.JSON_OBJECT,
        StructuredOutputMode.PROMPT_JSON,
    ]
    assert explicit_mode == [StructuredOutputMode.JSON_OBJECT]

    schema_format = build_response_format(
        StructuredOutputMode.JSON_SCHEMA,
        "agent_decision",
        AgentDecision,
    )
    object_format = build_response_format(
        StructuredOutputMode.JSON_OBJECT,
        "agent_decision",
        AgentDecision,
    )
    prompt_format = build_response_format(
        StructuredOutputMode.PROMPT_JSON,
        "agent_decision",
        AgentDecision,
    )

    assert schema_format is not None
    assert schema_format["type"] == "json_schema"
    assert object_format == {"type": "json_object"}
    assert prompt_format is None
    assert "Return exactly one JSON object" in prompt_json_instructions(AgentDecision)
    assert is_unsupported_format_error(
        "response_format json_schema unsupported",
        mode=StructuredOutputMode.JSON_SCHEMA,
    )
    assert is_unsupported_format_error(
        "json_object is unsupported",
        mode=StructuredOutputMode.JSON_OBJECT,
    )
    assert not is_unsupported_format_error(
        "generic network error",
        mode=StructuredOutputMode.PROMPT_JSON,
    )


def test_model_client_helpers_build_expected_headers_and_client() -> None:
    openrouter_settings = _settings(provider=ModelProvider.OPENROUTER)
    nvidia_settings = _settings(provider=ModelProvider.NVIDIA)

    headers = build_provider_headers(openrouter_settings)
    assert headers == {
        "HTTP-Referer": "https://example.com/",
        "X-OpenRouter-Title": "PSVIEW Recruiting Agent",
    }
    assert build_provider_headers(nvidia_settings) == {}

    client = create_model_client(openrouter_settings)
    assert isinstance(client, AsyncOpenAI)
    asyncio.run(client.close())


def test_collection_helpers_are_deterministic() -> None:
    assert dedupe_preserving_order(["a", "b", "a", "c"]) == ["a", "b", "c"]
    assert dedupe_casefold(["Role", "role", "ROLE", "Compensation"]) == [
        "Role",
        "Compensation",
    ]
    assert unique_sequence(["x", "x", "y"]) == ["x", "y"]


def test_text_helpers_normalize_and_split_content() -> None:
    assert normalize_whitespace("  hello   world  ") == "hello world"
    assert remove_empty_strings([" hi ", " ", "\n", "there"]) == ["hi", "there"]
    assert safe_truncate("abcdefgh", 7) == "abcd..."
    assert safe_truncate("abcdef", 3) == "abc"
    assert split_paragraphs("First paragraph.\n\n Second paragraph. ") == [
        "First paragraph.",
        "Second paragraph.",
    ]
    assert split_sentences("One. Two? Three!") == ["One.", "Two?", "Three!"]
    assert repair_common_mojibake("I\u00e2\u20ac\u2122m ready") == "I\u2019m ready"
    assert sanitize_generated_text("Hi\u00a0there\u2014I\u2019m ready\u2026") == (
        "Hi there-I'm ready..."
    )
