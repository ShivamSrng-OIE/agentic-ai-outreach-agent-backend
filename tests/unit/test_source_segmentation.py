"""Tests for source segmentation."""

from tests.fixtures.domain import sample_company_context

from psview_agent.retrieval.corpus_builder import segment_company_context


def test_source_segmentation_stable_ids_and_order() -> None:
    segments = segment_company_context(sample_company_context())
    assert segments[0].id == "segment_001"
    assert [segment.ordinal for segment in segments] == list(range(1, len(segments) + 1))
    assert segments[0].text == "Acme AI"


def test_source_segmentation_preserves_content_on_large_paragraph() -> None:
    context = sample_company_context()
    context.company_description = "Sentence one. " + ("Large content. " * 80)
    segments = segment_company_context(context)
    combined = " ".join(
        segment.text for segment in segments if segment.source_field.value == "company_description"
    )
    assert "Sentence one." in combined
    assert len(segments) > 1
