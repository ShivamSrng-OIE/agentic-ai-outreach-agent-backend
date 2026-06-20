"""Load configuration from YAML plus environment variables."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import ValidationError
from yaml.nodes import MappingNode, Node, ScalarNode, SequenceNode

from psview_agent.core.config import Settings, default_settings_dict
from psview_agent.core.config_merge import build_environment_overrides, deep_merge
from psview_agent.core.config_redaction import redacted_mapping
from psview_agent.core.env_placeholders import resolve_placeholders
from psview_agent.core.errors import (
    ConfigFileNotFoundError,
    ConfigYamlParseError,
    InvalidConfigurationError,
)


def _validate_unique_keys(node: Node) -> None:
    if isinstance(node, MappingNode):
        seen: set[str] = set()
        for key_node, value_node in node.value:
            if not isinstance(key_node, ScalarNode):
                raise ConfigYamlParseError("YAML keys must be strings")
            key = str(key_node.value)
            if key in seen:
                raise ConfigYamlParseError(f"duplicate YAML key: {key}")
            seen.add(key)
            _validate_unique_keys(value_node)
        return
    if isinstance(node, SequenceNode):
        for item in node.value:
            _validate_unique_keys(item)


@dataclass(frozen=True, slots=True)
class LoadedSettings:
    settings: Settings
    config_path: Path
    diagnostics: dict[str, object]


def determine_config_path() -> Path:
    """Determine the active YAML configuration file path."""
    raw = os.getenv("CONFIG_FILE", "config.yaml")
    return Path(raw).expanduser()


def _read_yaml_mapping(config_path: Path) -> Mapping[str, object]:
    if not config_path.exists():
        raise ConfigFileNotFoundError(f"configuration file not found: {config_path}")
    raw_text = config_path.read_text(encoding="utf-8")
    try:
        parsed_node = yaml.compose(raw_text, Loader=yaml.SafeLoader)
    except yaml.YAMLError as exc:
        raise ConfigYamlParseError(f"could not parse configuration YAML: {exc}") from exc
    if parsed_node is None:
        raise InvalidConfigurationError("configuration YAML must not be empty")
    _validate_unique_keys(parsed_node)
    try:
        raw_data = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        raise ConfigYamlParseError(f"could not parse configuration YAML: {exc}") from exc
    if not isinstance(raw_data, Mapping):
        raise InvalidConfigurationError("configuration YAML root must be a mapping")
    return {str(key): value for key, value in raw_data.items()}


def load_settings() -> LoadedSettings:
    """Load, resolve, merge, and validate settings."""
    config_path = determine_config_path()
    yaml_mapping = _read_yaml_mapping(config_path)
    dotenv_path = config_path.parent / ".env"
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path, override=False)
    resolved_yaml = resolve_placeholders(yaml_mapping)
    if not isinstance(resolved_yaml, Mapping):
        raise InvalidConfigurationError("resolved configuration YAML root must be a mapping")
    merged = deep_merge(
        default_settings_dict(),
        {str(key): value for key, value in resolved_yaml.items()},
    )
    merged = deep_merge(merged, build_environment_overrides())
    try:
        settings = Settings.model_validate(merged)
    except ValidationError as exc:
        raise InvalidConfigurationError(str(exc)) from exc
    diagnostics = redacted_mapping(
        {
            "config_path": str(config_path),
            "settings": settings.model_dump(),
        }
    )
    return LoadedSettings(settings=settings, config_path=config_path, diagnostics=diagnostics)
