"""Time helpers."""

from __future__ import annotations

from datetime import UTC, datetime


def utc_now() -> datetime:
    """Return the current UTC-aware timestamp."""
    return datetime.now(tz=UTC)


def ensure_utc(value: datetime) -> datetime:
    """Normalize a datetime to UTC and require timezone awareness."""
    if value.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(UTC)
