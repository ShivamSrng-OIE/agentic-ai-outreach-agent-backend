"""Deterministic lexical evidence retriever."""

from collections.abc import Sequence

from psview_agent.core.config import RetrievalSettings
from psview_agent.domain.company import EvidenceCorpus
from psview_agent.domain.enums import EvidenceKind
from psview_agent.domain.retrieval import RetrievalQuery, RetrievedEvidence
from psview_agent.retrieval.scoring import rank_facts

CANDIDATE_FACING_EVIDENCE_KINDS = {
    EvidenceKind.COMPANY_IDENTITY,
    EvidenceKind.PRODUCT,
    EvidenceKind.CULTURE,
    EvidenceKind.WORKING_STYLE,
    EvidenceKind.HIRING_PROFILE,
    EvidenceKind.ROLE_INFORMATION,
    EvidenceKind.COMPENSATION,
    EvidenceKind.VISA_OR_SPONSORSHIP,
    EvidenceKind.LOCATION,
    EvidenceKind.WORK_MODE,
    EvidenceKind.BENEFITS,
    EvidenceKind.FUNDING,
}


class LexicalEvidenceRetriever:
    """Pure-Python lexical retrieval over the in-memory evidence corpus."""

    def __init__(self, settings: RetrievalSettings) -> None:
        self.settings = settings

    def retrieve(
        self,
        *,
        corpus: EvidenceCorpus,
        query: RetrievalQuery,
        already_used_fact_ids: Sequence[str],
        limit: int,
    ) -> list[RetrievedEvidence]:
        ranked = rank_facts(
            corpus=corpus,
            query=query,
            already_used_fact_ids=already_used_fact_ids,
            reuse_penalty=self.settings.reuse_penalty,
        )
        results: list[RetrievedEvidence] = []
        candidate_rows = [
            (fact, score, matched_terms)
            for fact, score, matched_terms in ranked
            if fact.kind in CANDIDATE_FACING_EVIDENCE_KINDS and score >= self.settings.min_score
        ]
        max_score = max((score for _, score, _ in candidate_rows), default=0.0)
        for rank, (fact, score, matched_terms) in enumerate(candidate_rows, start=1):
            if score < self.settings.min_score:
                continue
            results.append(
                RetrievedEvidence(
                    evidence=fact,
                    rank=rank,
                    raw_relevance_score=score,
                    normalized_relevance=(score / max_score) if max_score > 0 else 0.0,
                    matched_terms=matched_terms,
                )
            )
            if len(results) >= min(limit, self.settings.top_k):
                break
        return results
