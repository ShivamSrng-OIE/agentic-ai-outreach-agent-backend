"""LangGraph state definition."""

from __future__ import annotations

from typing import NotRequired, TypedDict

from psview_agent.domain.agent import AgentConfiguration
from psview_agent.domain.candidate import CandidateProfile
from psview_agent.domain.conversation import ConversationMessage, ConversationState
from psview_agent.domain.decisions import AgentDecision, AgentDecisionDraft, CandidateAnalysis
from psview_agent.domain.evaluation import (
    DeterministicResponseCheck,
    GeneratedResponseDraft,
    ResponseEvaluation,
)
from psview_agent.domain.retrieval import RetrievalQuery, RetrievedEvidence


class ConversationGraphState(TypedDict):
    configuration: AgentConfiguration
    candidate: CandidateProfile
    target_role: str
    target_role_description: str | None
    conversation_state: ConversationState
    message_history: list[ConversationMessage]
    candidate_reply: str
    analysis: NotRequired[CandidateAnalysis]
    retrieval_query: NotRequired[RetrievalQuery]
    retrieved_evidence: NotRequired[list[RetrievedEvidence]]
    decision_draft: NotRequired[AgentDecisionDraft]
    decision: NotRequired[AgentDecision]
    response_draft: NotRequired[GeneratedResponseDraft]
    deterministic_check: NotRequired[DeterministicResponseCheck]
    evaluation: NotRequired[ResponseEvaluation]
    revision_count: int
    fallback_used: bool
    final_response: NotRequired[GeneratedResponseDraft]
    final_state: NotRequired[ConversationState]
