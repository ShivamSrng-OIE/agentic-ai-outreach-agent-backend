"""Text normalization and segmentation helpers."""

from __future__ import annotations

import re
from collections.abc import Iterable

from pydantic import BaseModel

from psview_agent.domain.base import JsonValue

WHITESPACE_RE = re.compile(r"\s+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200f\u2060\ufeff]")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")

PUNCTUATION_REPLACEMENTS = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2022": "-",
        "\u00a0": " ",
    }
)
MOJIBAKE_REPLACEMENTS = {
    "\u00c2\u00a0": " ",
    "\u00e2\u20ac\u2122": "\u2019",
    "\u00e2\u20ac\u02dc": "\u2018",
    "\u00e2\u20ac\u0153": "\u201c",
    "\u00e2\u20ac\u009d": "\u201d",
    "\u00e2\u20ac\u201c": "\u2013",
    "\u00e2\u20ac\u201d": "\u2014",
    "\u00e2\u20ac\u00a6": "\u2026",
}


def normalize_whitespace(value: str) -> str:
    """Collapse repeated whitespace into single spaces."""
    return WHITESPACE_RE.sub(" ", value).strip()


def remove_empty_strings(values: Iterable[str]) -> list[str]:
    """Remove blank values after normalization."""
    return [normalized for value in values if (normalized := normalize_whitespace(value))]


def safe_truncate(value: str, max_length: int) -> str:
    """Trim text to a safe maximum length."""
    if len(value) <= max_length:
        return value
    if max_length <= 3:
        return value[:max_length]
    return value[: max_length - 3].rstrip() + "..."


def split_paragraphs(value: str) -> list[str]:
    """Split text into normalized paragraphs."""
    paragraphs = [normalize_whitespace(part) for part in re.split(r"\n\s*\n", value)]
    return [part for part in paragraphs if part]


def split_sentences(value: str) -> list[str]:
    """Split text into coarse sentences for safe segmentation."""
    sentences = [normalize_whitespace(part) for part in SENTENCE_SPLIT_RE.split(value)]
    return [part for part in sentences if part]


def repair_common_mojibake(value: str) -> str:
    """Repair common UTF-8-as-Latin-1 mojibake when safely detectable."""
    if not any(marker in value for marker in ("\u00c3", "\u00c2", "\u00e2")):
        return value
    repaired = value
    for broken, fixed in MOJIBAKE_REPLACEMENTS.items():
        repaired = repaired.replace(broken, fixed)
    if repaired != value:
        return repaired
    for encoding in ("cp1252", "latin-1"):
        try:
            repaired = value.encode(encoding).decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
        return repaired
    return value


def sanitize_generated_text(value: str) -> str:
    """Normalize generated text into a plain, API-safe form."""
    sanitized = repair_common_mojibake(value)
    sanitized = sanitized.translate(PUNCTUATION_REPLACEMENTS)
    sanitized = sanitized.replace("\u2026", "...")
    sanitized = ZERO_WIDTH_RE.sub("", sanitized)
    sanitized = CONTROL_RE.sub("", sanitized)
    return normalize_whitespace(sanitized)


def sanitize_json_strings(value: JsonValue) -> JsonValue:
    """Recursively sanitize all string values inside a JSON-like payload."""
    if isinstance(value, str):
        return sanitize_generated_text(value)
    if isinstance(value, list):
        return [sanitize_json_strings(item) for item in value]
    if isinstance(value, dict):
        return {key: sanitize_json_strings(item) for key, item in value.items()}
    return value


def sanitize_model_strings[TModel: BaseModel](model: TModel) -> TModel:
    """Return a revalidated copy of a Pydantic model with sanitized string fields."""
    payload = model.model_dump(mode="json")
    sanitized_payload = sanitize_json_strings(payload)
    return model.__class__.model_validate(sanitized_payload)
