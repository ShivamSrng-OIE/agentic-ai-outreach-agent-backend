"""Prompt for one-shot response revision."""

from collections.abc import Sequence

from psview_agent.domain.agent import AgentConfiguration
from psview_agent.domain.candidate import CandidateProfile
from psview_agent.domain.conversation import ConversationMessage, ConversationState
from psview_agent.domain.decisions import AgentDecision
from psview_agent.domain.evaluation import GeneratedResponseDraft, ResponseEvaluation
from psview_agent.prompts.common import (
    PROMPT_SECURITY_INSTRUCTION,
    role_context_payload,
    untrusted_json_block,
)


def build_response_revision_prompts(
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
) -> tuple[str, str]:
    system_prompt = (
        "Revise the candidate-facing response once.\n"
        f"{PROMPT_SECURITY_INSTRUCTION}\n"
        "Fix only the identified failures, preserve valid content, keep the "
        "selected action, and use only supported evidence. Keep supported_claims aligned "
        "with the revised message. Every factual company claim must cite exact evidence IDs "
        "from decision.company_fact_ids_to_use, and unsupported or uncited claims must be "
        "removed instead of guessed."
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
            "evaluation": evaluation.model_dump() if evaluation is not None else None,
            "deterministic_violations": list(deterministic_violations),
        }
    )
    return system_prompt, user_prompt
