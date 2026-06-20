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
