"""Redaction helpers for configuration diagnostics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from pydantic import SecretStr

SECRET_FIELD_NAMES = {"api_key", "token", "secret", "password", "authorization"}
REDACTED = "**********"


def redact_value(value: object, *, path: tuple[str, ...] = ()) -> object:
    """Recursively redact sensitive values."""
    if isinstance(value, SecretStr):
        return REDACTED
    if path and path[-1].casefold() in SECRET_FIELD_NAMES:
        return REDACTED
    if isinstance(value, Mapping):
        return {
            str(key): redact_value(child, path=(*path, str(key))) for key, child in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [redact_value(child, path=path) for child in value]
    return value


def redacted_mapping(mapping: Mapping[str, object]) -> dict[str, object]:
    """Return a redacted copy of a mapping."""
    return {key: redact_value(value, path=(key,)) for key, value in mapping.items()}
