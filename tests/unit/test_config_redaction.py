"""Tests for configuration redaction."""

from pydantic import SecretStr

from psview_agent.core.config_redaction import REDACTED, redacted_mapping


def test_redacts_secret_fields_and_secretstr() -> None:
    payload = {
        "api_key": "secret",
        "nested": {"token": "another"},
        "plain": "value",
        "secret_obj": SecretStr("top-secret"),
    }
    redacted = redacted_mapping(payload)
    assert redacted["api_key"] == REDACTED
    assert redacted["nested"]["token"] == REDACTED  # type: ignore[index]
    assert redacted["secret_obj"] == REDACTED
    assert redacted["plain"] == "value"
