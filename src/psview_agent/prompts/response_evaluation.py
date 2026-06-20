"""Prompt for semantic response evaluation."""

from collections.abc import Sequence

from psview_agent.domain.agent import AgentConfiguration
from psview_agent.domain.candidate import CandidateProfile
from psview_agent.domain.conversation import ConversationMessage, ConversationState
from psview_agent.domain.decisions import AgentDecision
from psview_agent.domain.evaluation import GeneratedResponseDraft
from psview_agent.prompts.common import (
    PROMPT_SECURITY_INSTRUCTION,
    role_context_payload,
    untrusted_json_block,
)


def build_response_evaluation_prompts(
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
) -> tuple[str, str]:
    system_prompt = (
        "Evaluate the candidate-facing response independently.\n"
        f"{PROMPT_SECURITY_INSTRUCTION}\n"
        "Check persona consistency, grounding, relevance, action alignment, "
        "naturalness, repetition, and policy issues."
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
            "response": response.model_dump(),
        }
    )
    return system_prompt, user_prompt
