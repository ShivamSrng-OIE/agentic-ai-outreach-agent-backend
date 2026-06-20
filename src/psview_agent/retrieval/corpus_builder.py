"""Build source segments and evidence corpora."""

from __future__ import annotations

from collections.abc import Iterable

from psview_agent.core.errors import InvalidCompanyEvidenceError
from psview_agent.domain.company import (
    CompanyContextInput,
    EvidenceCorpus,
    EvidenceFact,
    EvidenceFactDraft,
    SourceSegment,
)
from psview_agent.domain.enums import SourceField
from psview_agent.utils.identifiers import prefixed_sequence_id
from psview_agent.utils.text import normalize_whitespace, split_paragraphs, split_sentences

MAX_SEGMENT_LENGTH = 700
MIN_SEGMENT_LENGTH = 20

FIELD_ORDER: list[tuple[SourceField, str]] = [
    (SourceField.COMPANY_NAME, "company_name"),
    (SourceField.COMPANY_DESCRIPTION, "company_description"),
    (SourceField.CULTURE_AND_VALUES, "culture_and_values"),
    (SourceField.HIRING_PROFILES, "hiring_profiles"),
    (SourceField.COMMUNICATION_TONE, "communication_tone"),
    (SourceField.RECRUITING_INTENT, "recruiting_intent"),
    (SourceField.ADDITIONAL_CONTEXT, "additional_context"),
]


def _segment_text(source_field: SourceField, text: str) -> list[str]:
    normalized = normalize_whitespace(text)
    if not normalized:
        return []
    if source_field is SourceField.COMPANY_NAME:
        return [normalized]

    segments: list[str] = []
    for paragraph in split_paragraphs(text):
        if len(paragraph) <= MAX_SEGMENT_LENGTH:
            segments.append(paragraph)
            continue
        buffer = ""
        for sentence in split_sentences(paragraph):
            candidate = f"{buffer} {sentence}".strip() if buffer else sentence
            if len(candidate) <= MAX_SEGMENT_LENGTH:
                buffer = candidate
                continue
            if buffer:
                segments.append(buffer)
            if len(sentence) <= MAX_SEGMENT_LENGTH:
                buffer = sentence
                continue
            start = 0
            while start < len(sentence):
                chunk = sentence[start : start + MAX_SEGMENT_LENGTH].strip()
                if chunk:
                    segments.append(chunk)
                start += MAX_SEGMENT_LENGTH
            buffer = ""
        if buffer:
            segments.append(buffer)
    return [segment for segment in segments if len(segment) >= MIN_SEGMENT_LENGTH]


def segment_company_context(context: CompanyContextInput) -> list[SourceSegment]:
    """Create deterministic source segments from the company context."""
    segments: list[SourceSegment] = []
    ordinal = 1
    for source_field, attribute in FIELD_ORDER:
        raw_value = getattr(context, attribute)
        for chunk in _segment_text(source_field, raw_value):
            segments.append(
                SourceSegment(
                    id=prefixed_sequence_id("segment", ordinal),
                    source_field=source_field,
                    text=chunk,
                    ordinal=ordinal,
                )
            )
            ordinal += 1
    return segments


def build_evidence_corpus(
    *,
    source_segments: list[SourceSegment],
    evidence_drafts: Iterable[EvidenceFactDraft],
) -> EvidenceCorpus:
    """Validate drafts and assign Python-owned evidence IDs."""
    segment_ids = {segment.id for segment in source_segments}
    facts: list[EvidenceFact] = []
    seen_facts: set[str] = set()
    for ordinal, draft in enumerate(evidence_drafts, start=1):
        if draft.fact.casefold() in seen_facts:
            raise InvalidCompanyEvidenceError("duplicate evidence fact draft")
        seen_facts.add(draft.fact.casefold())
        unknown_ids = [
            segment_id for segment_id in draft.source_segment_ids if segment_id not in segment_ids
        ]
        if unknown_ids:
            raise InvalidCompanyEvidenceError(
                f"evidence draft references unknown source segments: {', '.join(unknown_ids)}"
            )
        facts.append(
            EvidenceFact(
                id=prefixed_sequence_id("fact", ordinal),
                fact=draft.fact,
                kind=draft.kind,
                source_segment_ids=draft.source_segment_ids,
                retrieval_tags=draft.retrieval_tags,
            )
        )
    return EvidenceCorpus(source_segments=source_segments, evidence_facts=facts)
