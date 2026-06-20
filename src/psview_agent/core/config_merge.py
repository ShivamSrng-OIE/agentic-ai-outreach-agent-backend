"""Configuration deep-merge and environment override helpers."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping

type Parser = Callable[[str], object]


def _parse_str(value: str) -> object:
    return value


def _parse_int(value: str) -> object:
    return int(value)


def _parse_float(value: str) -> object:
    return float(value)


def _parse_bool(value: str) -> object:
    lowered = value.strip().casefold()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"invalid boolean value: {value}")


def _parse_csv(value: str) -> object:
    return [item.strip() for item in value.split(",") if item.strip()]


ENV_FIELD_MAP: dict[str, tuple[tuple[str, ...], Parser]] = {
    "APP_NAME": (("app", "name"), _parse_str),
    "APP_ENV": (("app", "env"), _parse_str),
    "APP_VERSION": (("app", "version"), _parse_str),
    "API_V1_PREFIX": (("app", "api_v1_prefix"), _parse_str),
    "LOG_LEVEL": (("app", "log_level"), _parse_str),
    "MODEL_PROVIDER": (("model", "provider"), _parse_str),
    "MODEL_API_KEY": (("model", "api_key"), _parse_str),
    "MODEL_BASE_URL": (("model", "base_url"), _parse_str),
    "MODEL_NAME": (("model", "model_name"), _parse_str),
    "MODEL_GENERAL_CHAT_NAME": (("model", "general_chat_model_name"), _parse_str),
    "MODEL_STRUCTURED_JSON_NAME": (("model", "structured_json_model_name"), _parse_str),
    "MODEL_CODING_BACKEND_NAME": (("model", "coding_backend_model_name"), _parse_str),
    "MODEL_STRUCTURED_OUTPUT_MODE": (("model", "structured_output_mode"), _parse_str),
    "MODEL_TIMEOUT_SECONDS": (("model", "timeout_seconds"), _parse_float),
    "MODEL_MAX_RETRIES": (("model", "max_retries"), _parse_int),
    "MODEL_MAX_OUTPUT_TOKENS": (("model", "max_output_tokens"), _parse_int),
    "MODEL_TEMPERATURE": (("model", "temperature"), _parse_float),
    "MODEL_REPAIR_ATTEMPTS": (("model", "repair_attempts"), _parse_int),
    "MODEL_CONCURRENCY_LIMIT": (("model", "concurrency_limit"), _parse_int),
    "OPENROUTER_SITE_URL": (("openrouter", "site_url"), _parse_str),
    "OPENROUTER_APP_NAME": (("openrouter", "app_name"), _parse_str),
    "ALLOWED_ORIGINS": (("runtime", "allowed_origins"), _parse_csv),
    "MAX_REQUEST_BODY_BYTES": (("runtime", "max_request_body_bytes"), _parse_int),
    "MAX_HISTORY_MESSAGES": (("runtime", "max_history_messages"), _parse_int),
    "MAX_CONVERSATION_TURNS": (("runtime", "max_conversation_turns"), _parse_int),
    "MAX_RESPONSE_CHARACTERS": (("runtime", "max_response_characters"), _parse_int),
    "MAX_REVISION_ATTEMPTS": (("runtime", "max_revision_attempts"), _parse_int),
    "LANGGRAPH_RECURSION_LIMIT": (("runtime", "langgraph_recursion_limit"), _parse_int),
    "RETRIEVAL_ENABLED": (("retrieval", "enabled"), _parse_bool),
    "RETRIEVAL_TOP_K": (("retrieval", "top_k"), _parse_int),
    "RETRIEVAL_MIN_SCORE": (("retrieval", "min_score"), _parse_float),
    "RETRIEVAL_REUSE_PENALTY": (("retrieval", "reuse_penalty"), _parse_float),
    "RETRIEVAL_MAX_FACT_CANDIDATES": (("retrieval", "max_fact_candidates"), _parse_int),
}


def _assign_nested(container: dict[str, object], path: tuple[str, ...], value: object) -> None:
    cursor = container
    for part in path[:-1]:
        existing = cursor.get(part)
        if not isinstance(existing, dict):
            new_child: dict[str, object] = {}
            cursor[part] = new_child
            cursor = new_child
        else:
            cursor = existing
    cursor[path[-1]] = value


def build_environment_overrides(environ: Mapping[str, str] | None = None) -> dict[str, object]:
    """Build nested overrides from supported flat environment aliases."""
    source = environ or os.environ
    overrides: dict[str, object] = {}
    for env_name, (path, parser) in ENV_FIELD_MAP.items():
        raw = source.get(env_name)
        if raw is None or raw.strip() == "":
            continue
        _assign_nested(overrides, path, parser(raw))
    return overrides


def deep_merge(base: Mapping[str, object], overlay: Mapping[str, object]) -> dict[str, object]:
    """Recursively merge overlay onto base."""
    result: dict[str, object] = dict(base)
    for key, value in overlay.items():
        existing = result.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            result[key] = deep_merge(
                {str(child_key): child_value for child_key, child_value in existing.items()},
                {str(child_key): child_value for child_key, child_value in value.items()},
            )
        else:
            result[key] = value
    return result
