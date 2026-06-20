"""Retrieval protocol."""

from collections.abc import Sequence
from typing import Protocol

from psview_agent.domain.company import EvidenceCorpus
from psview_agent.domain.retrieval import RetrievalQuery, RetrievedEvidence


class EvidenceRetriever(Protocol):
    """Deterministic evidence retriever."""

    def retrieve(
        self,
        *,
        corpus: EvidenceCorpus,
        query: RetrievalQuery,
        already_used_fact_ids: Sequence[str],
        limit: int,
    ) -> list[RetrievedEvidence]:
        """Return the best matching evidence for a query."""
