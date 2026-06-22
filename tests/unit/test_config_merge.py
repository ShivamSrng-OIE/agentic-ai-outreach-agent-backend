"""Tests for environment override merging."""

from psview_agent.core.config_merge import build_environment_overrides, deep_merge


def test_build_environment_overrides_parses_csv_and_ignores_empty() -> None:
    overrides = build_environment_overrides(
        {
            "ALLOWED_ORIGINS": "http://localhost:5173,http://localhost:3000",
            "MODEL_PROVIDER": "nvidia",
            "MODEL_NAME": "",
        }
    )
    runtime = overrides["runtime"]
    model = overrides["model"]
    assert isinstance(runtime, dict)
    assert isinstance(model, dict)
    assert runtime["allowed_origins"] == [
        "http://localhost:5173",
        "http://localhost:3000",
    ]
    assert model["provider"] == "nvidia"
    assert "model_name" not in model


def test_deep_merge_prefers_overlay_values() -> None:
    merged = deep_merge(
        {"a": {"b": 1, "c": 2}, "d": 3},
        {"a": {"c": 4}, "e": 5},
    )
    assert merged == {"a": {"b": 1, "c": 4}, "d": 3, "e": 5}


def test_build_environment_overrides_types() -> None:
    import pytest
    overrides = build_environment_overrides(
        {
            "MODEL_TIMEOUT_SECONDS": "15.5",
            "MODEL_MAX_RETRIES": "3",
            "RETRIEVAL_ENABLED": "true",
            "MODEL_RESUME_PARSING_NAME": "meta-llama/llama-3-8b-instruct:free",
        }
    )
    model = overrides["model"]
    retrieval = overrides["retrieval"]
    assert model["timeout_seconds"] == 15.5
    assert model["max_retries"] == 3
    assert model["resume_parsing_model_name"] == "meta-llama/llama-3-8b-instruct:free"
    assert retrieval["enabled"] is True

    # Test boolean off/false variations
    overrides_false = build_environment_overrides({"RETRIEVAL_ENABLED": "off"})
    assert overrides_false["retrieval"]["enabled"] is False

    # Test invalid boolean raises ValueError
    with pytest.raises(ValueError, match="invalid boolean value"):
        build_environment_overrides({"RETRIEVAL_ENABLED": "not-a-boolean"})

