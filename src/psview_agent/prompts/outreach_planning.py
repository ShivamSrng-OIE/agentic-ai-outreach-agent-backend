"""Prompt for candidate-specific outreach planning."""

from psview_agent.domain.agent import AgentConfiguration
from psview_agent.domain.candidate import CandidateProfile
from psview_agent.domain.retrieval import RetrievedEvidence
from psview_agent.prompts.common import (
    PROMPT_SECURITY_INSTRUCTION,
    role_context_payload,
    untrusted_json_block,
)


def build_outreach_planning_prompts(
    *,
    configuration: AgentConfiguration,
    candidate: CandidateProfile,
    target_role: str,
    target_role_description: str | None,
    retrieved_evidence: list[RetrievedEvidence],
) -> tuple[str, str]:
    system_prompt = (
        "You are creating an outreach preview for a recruiting agent.\n"
        f"{PROMPT_SECURITY_INSTRUCTION}\n"
        "Use only retrieved evidence facts. Produce exactly three messages: "
        "initial outreach, follow-up, and final closeout. Messages must be "
        "concise, candidate-specific, and ask at most one question. "
        "The initial message must begin with a personalized introduction where the recruiter addresses the candidate directly by name "
        "and introduces themselves by their exact persona name (provided in persona.name), e.g., 'Hi [Candidate Name], I'm [Persona Name], recruiting for the [Role] role at [Company Name].' "
        "The initial message should be three to four concise sentences: this personalized intro/opener, a role or job-description "
        "relevance sentence, one grounded company evidence sentence, and one soft call to "
        "action. The follow-up and closeout must stay under three sentences. Write like a "
        "thoughtful recruiter, not a sequence template. Ensure you use paragraphs, spaces, proper grammar, "
        "and a natural conversational cadence. The agent must be fully aware of its name throughout the messages. "
        "Keep the copy human, specific, and calm. Do not paste raw candidate.background_summary "
        "or raw role_context.description into the message; paraphrase naturally. Do not mention "
        "security, privacy, infrastructure, encryption, GDPR, hosting, or compliance evidence "
        "unless the candidate or role context explicitly makes those topics relevant. Avoid "
        "robotic phrasing, repeated openers, duplicated calls to action, and generic metadata "
        "values. Each message should have a meaningful objective and trigger that explain why "
        "that message exists. For each factual company statement, return supported_claims with "
        "the exact retrieved evidence IDs that support it; company_fact_ids_used is derived "
        "from those claims. Do not leave factual outreach statements unattributed. Avoid saying "
        "a candidate is a strong fit, perfect fit, or ideal fit unless the retrieved evidence "
        "directly supports that conclusion. When explaining why the candidate stood out, "
        "paraphrase concrete candidate details into a short natural phrase."
    )
    user_prompt = untrusted_json_block(
        {
            "persona": configuration.persona.model_dump(),
            "candidate": candidate.model_dump(),
            "role_context": role_context_payload(
                target_role=target_role,
                target_role_description=target_role_description,
            ),
            "retrieved_evidence": [item.model_dump() for item in retrieved_evidence],
        }
    )
    return system_prompt, user_prompt
