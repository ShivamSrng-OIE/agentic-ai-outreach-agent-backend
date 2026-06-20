"""Base model and shared type aliases."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

type JsonPrimitive = None | bool | int | float | str
type JsonValue = JsonPrimitive | list["JsonValue"] | dict[str, "JsonValue"]


class StrictModel(BaseModel):
    """Shared strict Pydantic base model."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
        use_enum_values=False,
    )
