"""Build retrieval queries for outreach and conversation turns."""

from psview_agent.domain.candidate import CandidateProfile
from psview_agent.domain.company import CompanyContextInput
from psview_agent.domain.conversation import ConversationState
from psview_agent.domain.decisions import CandidateAnalysis
from psview_agent.domain.retrieval import RetrievalQuery
from psview_agent.utils.text import normalize_whitespace


def _role_topics(target_role_description: str | None) -> list[str]:
    if not target_role_description:
        return []
    normalized = normalize_whitespace(target_role_description)
    if not normalized:
        return []
    return [normalized[:200]]


def build_initial_retrieval_query(
    *,
    configuration_context: CompanyContextInput,
    candidate: CandidateProfile,
    target_role: str,
    target_role_description: str | None,
) -> RetrievalQuery:
    """Build the initial candidate-matching retrieval query."""
    parts = [
        target_role,
        candidate.current_role,
        candidate.background_summary,
        configuration_context.hiring_profiles,
    ]
    if target_role_description:
        parts.append(target_role_description)
    if candidate.resume_text:
        parts.append(candidate.resume_text)
    text = " ".join(parts)
    return RetrievalQuery(
        text=text,
        target_role=target_role,
        target_role_description=target_role_description,
        topics=[
            candidate.current_role,
            target_role,
            "candidate relevance",
            *_role_topics(target_role_description),
        ],
    )


def build_turn_retrieval_query(
    *,
    candidate_reply: str,
    analysis: CandidateAnalysis,
    target_role: str,
    target_role_description: str | None,
    state: ConversationState,
) -> RetrievalQuery:
    """Build the retrieval query for a new conversation turn."""
    parts = [
        candidate_reply,
        " ".join(analysis.questions_or_topics),
        " ".join(analysis.candidate_concerns),
        " ".join(analysis.expressed_motivations),
        target_role,
        state.stage.value,
    ]
    if target_role_description:
        parts.append(target_role_description)
    text = " ".join(parts)
    return RetrievalQuery(
        text=text,
        target_role=target_role,
        target_role_description=target_role_description,
        topics=[
            *analysis.retrieval_topics,
            *analysis.questions_or_topics,
            *analysis.candidate_concerns,
            *_role_topics(target_role_description),
        ],
        action=state.last_action,
    )
