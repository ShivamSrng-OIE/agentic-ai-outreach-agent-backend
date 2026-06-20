"""Deterministic lexical scoring."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from math import log

from psview_agent.domain.company import EvidenceCorpus, EvidenceFact
from psview_agent.domain.retrieval import RetrievalQuery
from psview_agent.retrieval.tokenization import tokenize_text


def build_fact_document(corpus: EvidenceCorpus, fact: EvidenceFact) -> str:
    """Build the text indexed for a fact."""
    segment_lookup = {segment.id: segment.text for segment in corpus.source_segments}
    segment_text = " ".join(
        segment_lookup[segment_id]
        for segment_id in fact.source_segment_ids
        if segment_id in segment_lookup
    )
    tags = " ".join(fact.retrieval_tags)
    return f"{fact.fact} {tags} {segment_text}".strip()


def document_frequencies(corpus: EvidenceCorpus) -> Counter[str]:
    """Compute document frequencies for all tokens in the corpus."""
    frequencies: Counter[str] = Counter()
    for fact in corpus.evidence_facts:
        frequencies.update(set(tokenize_text(build_fact_document(corpus, fact))))
    return frequencies


def idf_weight(document_count: int, document_frequency: int) -> float:
    """Compute the deterministic IDF-style weight."""
    return log((document_count + 1) / (document_frequency + 1)) + 1


def build_query_tokens(query: RetrievalQuery) -> list[str]:
    """Build the token set used for scoring a query."""
    parts = [query.text, query.target_role, *query.topics]
    if query.target_role_description:
        parts.append(query.target_role_description)
    combined = " ".join(parts)
    if query.action is not None:
        combined += f" {query.action.value}"
    return tokenize_text(combined)


def score_fact(
    *,
    corpus: EvidenceCorpus,
    fact: EvidenceFact,
    query: RetrievalQuery,
    frequencies: Counter[str],
    document_count: int,
    already_used_fact_ids: Sequence[str],
    reuse_penalty: float,
) -> tuple[float, list[str]]:
    """Score an evidence fact deterministically."""
    document_text = build_fact_document(corpus, fact)
    document_tokens = tokenize_text(document_text)
    token_counts = Counter(document_tokens)
    query_tokens = build_query_tokens(query)
    matched_terms = [token for token in query_tokens if token in token_counts]
    score = 0.0
    for token in matched_terms:
        score += idf_weight(document_count, frequencies[token]) * token_counts[token]
    lowered_document = document_text.lower()
    if query.text.lower() in lowered_document:
        score += 1.5
    for tag in fact.retrieval_tags:
        if tag.casefold() in {topic.casefold() for topic in query.topics}:
            score += 0.5
    role_tokens = set(tokenize_text(query.target_role))
    score += 0.25 * sum(1 for token in document_tokens if token in role_tokens)
    topic_tokens = set(tokenize_text(" ".join(query.topics)))
    score += 0.35 * sum(1 for token in document_tokens if token in topic_tokens)
    if query.action is not None and query.action.value.replace("_", " ") in lowered_document:
        score += 0.4
    if fact.id in already_used_fact_ids:
        score = max(0.0, score - reuse_penalty)
    return score, sorted(set(matched_terms))


def rank_facts(
    *,
    corpus: EvidenceCorpus,
    query: RetrievalQuery,
    already_used_fact_ids: Sequence[str],
    reuse_penalty: float,
) -> list[tuple[EvidenceFact, float, list[str]]]:
    """Rank facts for a query."""
    frequencies = document_frequencies(corpus)
    document_count = len(corpus.evidence_facts)
    scored: list[tuple[EvidenceFact, float, list[str]]] = []
    for fact in corpus.evidence_facts:
        score, matched_terms = score_fact(
            corpus=corpus,
            fact=fact,
            query=query,
            frequencies=frequencies,
            document_count=document_count,
            already_used_fact_ids=already_used_fact_ids,
            reuse_penalty=reuse_penalty,
        )
        scored.append((fact, score, matched_terms))
    scored.sort(key=lambda item: (-item[1], item[0].id))
    return scored
