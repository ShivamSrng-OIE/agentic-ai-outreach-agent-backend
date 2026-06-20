"""Safe fallback responses."""

from __future__ import annotations

from psview_agent.domain.decisions import AgentDecision
from psview_agent.domain.enums import AgentAction
from psview_agent.domain.evaluation import GeneratedResponseDraft


def build_safe_fallback(*, decision: AgentDecision) -> GeneratedResponseDraft:
    """Return a safe deterministic fallback response."""
    if decision.selected_action is AgentAction.GRACEFULLY_EXIT:
        return GeneratedResponseDraft(
            message="Understood. I will not contact you again. Thank you for the reply.",
            company_fact_ids_used=[],
        )
    if decision.selected_action is AgentAction.DISCLOSE_AI_IDENTITY:
        return GeneratedResponseDraft(
            message=(
                "I am an AI recruiting assistant working on the company's behalf. "
                "I can share the information included in the supplied context."
            ),
            company_fact_ids_used=[],
        )
    if decision.missing_information:
        topic = decision.missing_information[0]
        return GeneratedResponseDraft(
            message=(
                f"The supplied context does not include confirmed information "
                f"about {topic}. I do not want to guess."
            ),
            company_fact_ids_used=[],
        )
    return GeneratedResponseDraft(
        message=(
            "Thanks for the reply. I want to stay accurate, so I will keep this "
            "brief and avoid guessing beyond the supplied company context."
        ),
        company_fact_ids_used=[],
    )
