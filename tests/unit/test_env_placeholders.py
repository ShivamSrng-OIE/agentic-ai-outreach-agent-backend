"""Tests for environment placeholder resolution."""

import pytest

from psview_agent.core.env_placeholders import resolve_placeholders
from psview_agent.core.errors import UnresolvedEnvironmentPlaceholderError


def test_resolve_exact_placeholder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_API_KEY", "secret-value")
    assert resolve_placeholders("${MODEL_API_KEY}") == "secret-value"


def test_resolve_nested_mapping_and_list(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FIRST", "one")
    monkeypatch.setenv("SECOND", "two")
    resolved = resolve_placeholders({"a": ["${FIRST}", {"b": "${SECOND}"}]})
    assert resolved == {"a": ["one", {"b": "two"}]}


def test_reject_partial_placeholder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_API_KEY", "secret-value")
    with pytest.raises(UnresolvedEnvironmentPlaceholderError):
        resolve_placeholders("prefix-${MODEL_API_KEY}")


def test_reject_missing_or_empty_placeholder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    with pytest.raises(UnresolvedEnvironmentPlaceholderError):
        resolve_placeholders("${MODEL_API_KEY}")
    monkeypatch.setenv("MODEL_API_KEY", "")
    with pytest.raises(UnresolvedEnvironmentPlaceholderError):
        resolve_placeholders("${MODEL_API_KEY}")
