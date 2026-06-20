"""Identifier helpers."""

from uuid import UUID, uuid4


def new_uuid() -> UUID:
    """Return a new UUID4."""
    return uuid4()


def prefixed_sequence_id(prefix: str, ordinal: int) -> str:
    """Build a zero-padded deterministic sequence identifier."""
    return f"{prefix}_{ordinal:03d}"


def new_request_id() -> str:
    """Build a compact request identifier."""
    return uuid4().hex
