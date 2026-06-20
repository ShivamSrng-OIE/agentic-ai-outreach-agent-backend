"""Tokenization utilities for lexical retrieval."""

from __future__ import annotations

import re

STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "across",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "may",
    "of",
    "our",
    "on",
    "or",
    "that",
    "the",
    "to",
    "with",
    "we",
    "well",
    "who",
    "can",
    "want",
    "work",
    "you",
    "your",
}
TOKEN_RE = re.compile(r"[0-9A-Za-z]+(?:'[0-9A-Za-z]+)?")


def tokenize_text(value: str) -> list[str]:
    """Tokenize text into useful lowercase terms."""
    tokens: list[str] = []
    for raw in TOKEN_RE.findall(value.lower()):
        token = raw.removesuffix("'s")
        if token in STOP_WORDS:
            continue
        tokens.append(token)
    return tokens
