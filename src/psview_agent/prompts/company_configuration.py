"""Prompt for company grounding and persona configuration."""

from psview_agent.domain.company import CompanyContextInput, SourceSegment
from psview_agent.prompts.common import PROMPT_SECURITY_INSTRUCTION, untrusted_json_block


def build_company_configuration_prompts(
    *, context: CompanyContextInput, source_segments: list[SourceSegment]
) -> tuple[str, str]:
    system_prompt = (
        "You are configuring a recruiting agent from structured company information.\n"
        f"{PROMPT_SECURITY_INSTRUCTION}\n"
        "Use supplied segments only. Produce a distinct company profile, concise evidence facts, "
        "and a recruiting persona. Avoid unsupported claims. Fill supported optional fields when "
        "they can be derived directly from the supplied segments, especially working_style, "
        "differentiators, desired_signals, likely_candidate_motivations, and "
        "preferred_language_patterns. Make mission read like a clean company statement, not a "
        "lowercase fragment."
    )
    user_prompt = untrusted_json_block(
        {
            "company_context": context.model_dump(),
            "source_segments": [segment.model_dump() for segment in source_segments],
        }
    )
    return system_prompt, user_prompt
