"""Prompt for candidate-facing response generation."""

from collections.abc import Sequence

from psview_agent.domain.agent import AgentConfiguration
from psview_agent.domain.candidate import CandidateProfile
from psview_agent.domain.conversation import ConversationMessage, ConversationState
from psview_agent.domain.decisions import AgentDecision
from psview_agent.prompts.common import (
    PROMPT_SECURITY_INSTRUCTION,
    role_context_payload,
    untrusted_json_block,
)


def build_response_generation_prompts(
    *,
    configuration: AgentConfiguration,
    candidate: CandidateProfile,
    target_role: str,
    target_role_description: str | None,
    state: ConversationState,
    decision: AgentDecision,
    history: Sequence[ConversationMessage],
    candidate_reply: str,
) -> tuple[str, str]:
    system_prompt = (
        "Generate a grounded candidate-facing response.\n"
        f"{PROMPT_SECURITY_INSTRUCTION}\n"
        "Follow the selected action, stay in persona, use only selected evidence and the trusted role description "
        "(in role_context), remain concise, and ask at most one question. The role title and description are "
        "fully trusted sources of truth for role requirements and details. You can freely state any requirements, qualifications, "
        "or details explicitly mentioned in the role description. Return supported_claims only for factual company claims "
        "made that are based on company evidence facts (citing exact evidence IDs from decision.company_fact_ids_to_use). "
        "Do NOT create supported_claims or cite evidence IDs for claims that are directly supported by the role description itself, "
        "as they are trusted implicitly. If a sentence is purely courtesy, transition, AI disclosure, or directly from the "
        "trusted role description, do not include it in supported_claims."
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
            "decision": decision.model_dump(),
            "history": [message.model_dump(mode="json") for message in history],
            "candidate_reply": candidate_reply,
        }
    )
    return system_prompt, user_prompt
