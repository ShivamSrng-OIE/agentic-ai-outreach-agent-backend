"""Structured output mode helpers."""

from pydantic import BaseModel

from psview_agent.core.config import ModelProvider, StructuredOutputMode


def mode_sequence(
    provider: ModelProvider,
    preferred: StructuredOutputMode,
) -> list[StructuredOutputMode]:
    """Return the allowed fallback sequence."""
    if preferred is not StructuredOutputMode.AUTO:
        return [preferred]
    if provider is ModelProvider.OPENROUTER:
        return [
            StructuredOutputMode.JSON_SCHEMA,
            StructuredOutputMode.JSON_OBJECT,
            StructuredOutputMode.PROMPT_JSON,
        ]
    return [StructuredOutputMode.JSON_OBJECT, StructuredOutputMode.PROMPT_JSON]


def build_response_format(
    mode: StructuredOutputMode,
    schema_name: str,
    output_model: type[BaseModel],
) -> dict[str, object] | None:
    """Return the OpenAI-compatible response_format payload."""
    if mode is StructuredOutputMode.JSON_SCHEMA:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": True,
                "schema": output_model.model_json_schema(),
            },
        }
    if mode is StructuredOutputMode.JSON_OBJECT:
        return {"type": "json_object"}
    return None


def prompt_json_instructions(output_model: type[BaseModel]) -> str:
    """Return prompt instructions for explicit JSON output."""
    return (
        "Return exactly one JSON object that matches this schema. "
        "Do not include markdown, prose, comments, or extra fields.\n"
        f"{output_model.model_json_schema()}"
    )


def is_unsupported_format_error(message: str, *, mode: StructuredOutputMode) -> bool:
    """Detect whether an error is a mode-support issue."""
    lowered = message.casefold()
    if mode is StructuredOutputMode.JSON_SCHEMA:
        return "json_schema" in lowered or "response_format" in lowered or "unsupported" in lowered
    if mode is StructuredOutputMode.JSON_OBJECT:
        return "json_object" in lowered or "response_format" in lowered or "unsupported" in lowered
    return False
