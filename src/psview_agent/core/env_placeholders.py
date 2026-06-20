"""Exact environment placeholder resolution."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence

from psview_agent.core.errors import UnresolvedEnvironmentPlaceholderError

EXACT_PLACEHOLDER_RE = re.compile(r"^\$\{([A-Z0-9_]+)\}$")


def resolve_placeholders(value: object) -> object:
    """Recursively resolve exact ${ENV_VAR} placeholders."""
    if isinstance(value, str):
        match = EXACT_PLACEHOLDER_RE.fullmatch(value)
        if match:
            env_name = match.group(1)
            resolved = os.getenv(env_name)
            if resolved is None or resolved.strip() == "":
                raise UnresolvedEnvironmentPlaceholderError(
                    f"required environment variable {env_name} is missing or empty"
                )
            return resolved
        if "${" in value or value.startswith("$"):
            raise UnresolvedEnvironmentPlaceholderError(f"unsupported placeholder syntax: {value}")
        return value
    if isinstance(value, Mapping):
        return {str(key): resolve_placeholders(child) for key, child in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [resolve_placeholders(child) for child in value]
    return value
