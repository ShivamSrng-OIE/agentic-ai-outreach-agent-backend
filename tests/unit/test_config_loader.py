"""Tests for settings loading."""

from pathlib import Path

import pytest
from tests.conftest import write_test_config

from psview_agent.core.config import AppEnvironment
from psview_agent.core.config_loader import load_settings
from psview_agent.core.errors import (
    InvalidConfigurationError,
    UnresolvedEnvironmentPlaceholderError,
)


def test_load_settings_from_yaml_and_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = tmp_path / "config.yaml"
    write_test_config(config)
    monkeypatch.setenv("CONFIG_FILE", str(config))
    monkeypatch.setenv("MODEL_API_KEY", "key-123")
    monkeypatch.setenv("MODEL_NAME", "model-123")
    monkeypatch.chdir(tmp_path)

    loaded = load_settings()

    assert loaded.settings.model.api_key.get_secret_value() == "key-123"
    assert loaded.settings.model.model_name == "model-123"
    assert loaded.settings.app.env is AppEnvironment.TEST


def test_shell_env_wins_over_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = tmp_path / "config.yaml"
    write_test_config(config)
    (tmp_path / ".env").write_text("MODEL_NAME=dotenv-model\n", encoding="utf-8")
    monkeypatch.setenv("CONFIG_FILE", str(config))
    monkeypatch.setenv("MODEL_API_KEY", "key-123")
    monkeypatch.setenv("MODEL_NAME", "shell-model")
    monkeypatch.chdir(tmp_path)

    loaded = load_settings()

    assert loaded.settings.model.model_name == "shell-model"


def test_missing_placeholder_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = tmp_path / "config.yaml"
    write_test_config(config)
    monkeypatch.setenv("CONFIG_FILE", str(config))
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    monkeypatch.delenv("MODEL_NAME", raising=False)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(UnresolvedEnvironmentPlaceholderError):
        load_settings()


def test_invalid_yaml_shape_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("- not-a-mapping\n", encoding="utf-8")
    monkeypatch.setenv("CONFIG_FILE", str(config))
    monkeypatch.setenv("MODEL_API_KEY", "key-123")
    monkeypatch.setenv("MODEL_NAME", "model-123")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(InvalidConfigurationError):
        load_settings()


def test_production_wildcard_cors_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        (
            "app:\n"
            "  name: PSVIEW Recruiting Agent API\n"
            "  env: production\n"
            "  version: 0.1.0\n"
            "  api_v1_prefix: /api/v1\n"
            "  log_level: INFO\n"
            "model:\n"
            "  provider: openrouter\n"
            "  api_key: ${MODEL_API_KEY}\n"
            "  base_url: https://openrouter.ai/api/v1\n"
            "  model_name: ${MODEL_NAME}\n"
            "  structured_output_mode: auto\n"
            "  timeout_seconds: 5\n"
            "  max_retries: 1\n"
            "  max_output_tokens: 600\n"
            "  temperature: 0.2\n"
            "  repair_attempts: 1\n"
            "  concurrency_limit: 2\n"
            "  extra_body: {}\n"
            "openrouter:\n"
            "  site_url: null\n"
            "  app_name: PSVIEW Recruiting Agent\n"
            "runtime:\n"
            "  allowed_origins: ['*']\n"
            "  max_request_body_bytes: 100000\n"
            "  max_history_messages: 20\n"
            "  max_conversation_turns: 16\n"
            "  max_response_characters: 1000\n"
            "  max_revision_attempts: 1\n"
            "  langgraph_recursion_limit: 20\n"
            "retrieval:\n"
            "  enabled: true\n"
            "  top_k: 5\n"
            "  min_score: 0.05\n"
            "  reuse_penalty: 0.15\n"
            "  max_fact_candidates: 20\n"
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CONFIG_FILE", str(config))
    monkeypatch.setenv("MODEL_API_KEY", "key-123")
    monkeypatch.setenv("MODEL_NAME", "model-123")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(InvalidConfigurationError):
        load_settings()
