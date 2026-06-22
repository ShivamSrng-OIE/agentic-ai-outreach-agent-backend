"""Prompt for action planning."""

from collections.abc import Sequence

from psview_agent.domain.agent import AgentConfiguration
from psview_agent.domain.candidate import CandidateProfile
from psview_agent.domain.conversation import ConversationMessage, ConversationState
from psview_agent.domain.decisions import CandidateAnalysis
from psview_agent.domain.retrieval import RetrievedEvidence
from psview_agent.prompts.common import (
    PROMPT_SECURITY_INSTRUCTION,
    role_context_payload,
    untrusted_json_block,
)


def build_action_planning_prompts(
    *,
    configuration: AgentConfiguration,
    candidate: CandidateProfile,
    target_role: str,
    target_role_description: str | None,
    state: ConversationState,
    analysis: CandidateAnalysis,
    history: Sequence[ConversationMessage],
    retrieved_evidence: Sequence[RetrievedEvidence],
) -> tuple[str, str]:
    system_prompt = (
        "Plan the recruiting agent's next objective and action.\n"
        f"{PROMPT_SECURITY_INSTRUCTION}\n"
        "Select only evidence IDs from the retrieved evidence. Note that the target role and its description "
        "(in role_context) are fully trusted sources of truth for the role's details, requirements, qualifications, "
        "and responsibilities. Do NOT treat role details or qualifications explicitly stated in the role description "
        "as missing information. Identify missing information only when a topic is not supported by either the "
        "retrieved evidence or the role description. The rationale_summary must be specific to the candidate's latest reply "
        "and must name the selected evidence IDs, role context details used, or the missing information that blocks a stronger answer. "
        "Avoid generic wording such as 'keep it grounded' or unsupported fit language."
    )
    user_prompt = untrusted_json_block(
        {
            "persona": configuration.persona.model_dump(),
            "candidate": candidate.model_dump(),
            "role_context": role_context_payload(
                target_role=target_role,
                target_role_description=target_role_description,
            ),
            "state": state.model_dump(),
            "analysis": analysis.model_dump(),
            "history": [message.model_dump(mode="json") for message in history],
            "retrieved_evidence": [item.model_dump() for item in retrieved_evidence],
        }
    )
    return system_prompt, user_prompt
