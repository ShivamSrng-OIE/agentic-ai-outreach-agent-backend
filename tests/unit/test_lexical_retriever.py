"""Tests for lexical retrieval."""

from tests.fixtures.domain import sample_candidate, sample_configuration

from psview_agent.core.config import RetrievalSettings
from psview_agent.domain.retrieval import RetrievalQuery
from psview_agent.retrieval.lexical_retriever import LexicalEvidenceRetriever
from psview_agent.retrieval.query_builder import build_initial_retrieval_query


def test_lexical_retriever_ranks_relevant_fact_first() -> None:
    configuration = sample_configuration()
    retriever = LexicalEvidenceRetriever(
        RetrievalSettings(
            enabled=True,
            top_k=5,
            min_score=0,
            reuse_penalty=0.1,
            max_fact_candidates=20,
        )
    )
    query = RetrievalQuery(
        text="product minded engineers hiring",
        target_role="Senior Engineer",
        topics=["engineering", "hiring"],
    )
    results = retriever.retrieve(
        corpus=configuration.evidence_corpus,
        query=query,
        already_used_fact_ids=[],
        limit=3,
    )
    assert results
    assert results[0].evidence.id == "fact_001"
    assert results[0].rank == 1
    assert results[0].normalized_relevance == 1.0


def test_lexical_retriever_filters_non_candidate_facing_evidence() -> None:
    configuration = sample_configuration()
    retriever = LexicalEvidenceRetriever(
        RetrievalSettings(
            enabled=True,
            top_k=5,
            min_score=0,
            reuse_penalty=0.1,
            max_fact_candidates=20,
        )
    )
    query = RetrievalQuery(
        text="growth intent recruiting",
        target_role="Senior Engineer",
        topics=["growth", "recruiting"],
    )
    results = retriever.retrieve(
        corpus=configuration.evidence_corpus,
        query=query,
        already_used_fact_ids=[],
        limit=5,
    )
    assert all(item.evidence.id != "fact_003" for item in results)


def test_lexical_retriever_respects_threshold() -> None:
    configuration = sample_configuration()
    retriever = LexicalEvidenceRetriever(
        RetrievalSettings(
            enabled=True,
            top_k=5,
            min_score=10,
            reuse_penalty=0.1,
            max_fact_candidates=20,
        )
    )
    query = RetrievalQuery(text="unrelated", target_role="Designer", topics=["nothing"])
    results = retriever.retrieve(
        corpus=configuration.evidence_corpus,
        query=query,
        already_used_fact_ids=[],
        limit=3,
    )
    assert results == []


def test_initial_query_includes_target_role_description() -> None:
    configuration = sample_configuration()
    query = build_initial_retrieval_query(
        configuration_context=configuration.company_context,
        candidate=sample_candidate(),
        target_role="Founding Engineer",
        target_role_description=(
            "Own backend systems, ship integrations, and shape the engineering foundation."
        ),
    )
    assert "shape the engineering foundation" in query.text
    assert query.target_role_description is not None
