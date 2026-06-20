"""Collection helpers with deterministic ordering."""

from collections.abc import Iterable, Sequence


def dedupe_preserving_order(values: Iterable[str]) -> list[str]:
    """Return normalized unique values in first-seen order."""
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def dedupe_casefold(values: Iterable[str]) -> list[str]:
    """Deduplicate case-insensitively while preserving first-seen casing."""
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def unique_sequence(values: Sequence[str]) -> list[str]:
    """Return a unique copy of a string sequence."""
    return dedupe_preserving_order(values)
