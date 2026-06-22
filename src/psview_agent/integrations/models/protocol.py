"""Provider-neutral model gateway protocol."""

from collections.abc import Sequence
from typing import Protocol

from psview_agent.domain.agent import (
    AgentConfiguration,
    CompanyAgentConfigurationDraft,
    OutreachPlanDraft,
)
from psview_agent.domain.candidate import CandidateProfile, ExtractedCandidateProfile
from psview_agent.domain.company import CompanyContextInput, SourceSegment
from psview_agent.domain.conversation import ConversationMessage, ConversationState
from psview_agent.domain.decisions import AgentDecision, AgentDecisionDraft, CandidateAnalysis
from psview_agent.domain.evaluation import GeneratedResponseDraft, ResponseEvaluation
from psview_agent.domain.retrieval import RetrievedEvidence


class ModelGateway(Protocol):
    """Provider-neutral protocol used by services and graph nodes."""

    async def configure_company_agent(
        self,
        *,
        context: CompanyContextInput,
        source_segments: Sequence[SourceSegment],
    ) -> CompanyAgentConfigurationDraft: ...

    async def generate_outreach_plan(
        self,
        *,
        configuration: AgentConfiguration,
        candidate: CandidateProfile,
        target_role: str,
        target_role_description: str | None,
        retrieved_evidence: Sequence[RetrievedEvidence],
    ) -> OutreachPlanDraft: ...

    async def analyze_candidate_reply(
        self,
        *,
        configuration: AgentConfiguration,
        candidate: CandidateProfile,
        target_role: str,
        target_role_description: str | None,
        state: ConversationState,
        history: Sequence[ConversationMessage],
        candidate_reply: str,
    ) -> CandidateAnalysis: ...

    async def plan_next_action(
        self,
        *,
        configuration: AgentConfiguration,
        candidate: CandidateProfile,
        target_role: str,
        target_role_description: str | None,
        state: ConversationState,
        analysis: CandidateAnalysis,
        history: Sequence[ConversationMessage],
        retrieved_evidence: Sequence[RetrievedEvidence],
    ) -> AgentDecisionDraft: ...

    async def generate_candidate_response(
        self,
        *,
        configuration: AgentConfiguration,
        candidate: CandidateProfile,
        target_role: str,
        target_role_description: str | None,
        state: ConversationState,
        decision: AgentDecision,
        history: Sequence[ConversationMessage],
        candidate_reply: str,
    ) -> GeneratedResponseDraft: ...

    async def evaluate_candidate_response(
        self,
        *,
        configuration: AgentConfiguration,
        candidate: CandidateProfile,
        target_role: str,
        target_role_description: str | None,
        state: ConversationState,
        decision: AgentDecision,
        history: Sequence[ConversationMessage],
        candidate_reply: str,
        response: GeneratedResponseDraft,
    ) -> ResponseEvaluation: ...

    async def revise_candidate_response(
        self,
        *,
        configuration: AgentConfiguration,
        candidate: CandidateProfile,
        target_role: str,
        target_role_description: str | None,
        state: ConversationState,
        decision: AgentDecision,
        history: Sequence[ConversationMessage],
        candidate_reply: str,
        response: GeneratedResponseDraft,
        evaluation: ResponseEvaluation | None,
        deterministic_violations: Sequence[str],
    ) -> GeneratedResponseDraft: ...

    async def extract_profile_from_resume(
        self,
        *,
        resume_text: str,
    ) -> ExtractedCandidateProfile: ...
