"""Deterministic response validation."""

from __future__ import annotations

import re
from collections.abc import Sequence
from difflib import SequenceMatcher

from psview_agent.core.config import Settings
from psview_agent.domain.conversation import ConversationMessage
from psview_agent.domain.decisions import AgentDecision
from psview_agent.domain.evaluation import DeterministicResponseCheck, GeneratedResponseDraft

HTML_RE = re.compile(r"<[^>]+>")
UNSUPPORTED_FIT_LANGUAGE = (
    "strong fit",
    "perfect fit",
    "ideal fit",
)
CLAIM_ATTRIBUTION_ACTIONS = {
    "introduce_opportunity",
    "explain_candidate_relevance",
    "answer_candidate_question",
    "address_objection",
    "suggest_next_step",
}


def run_response_checks(
    *,
    settings: Settings,
    decision: AgentDecision,
    response: GeneratedResponseDraft,
    history: Sequence[ConversationMessage],
) -> DeterministicResponseCheck:
    """Validate a generated response against deterministic rules."""
    violations: list[str] = []
    message = response.message.strip()
    question_count = message.count("?")
    character_count = len(message)
    retrieved_ids = {item.evidence.id for item in decision.retrieved_evidence}
    cited_ids = {
        fact_id
        for claim in response.supported_claims
        for fact_id in claim.evidence_fact_ids
    }
    if not message:
        violations.append("empty_response")
    if character_count > settings.runtime.max_response_characters:
        violations.append("character_limit")
    if question_count > 1:
        violations.append("multiple_questions")
    if any(fact_id not in retrieved_ids for fact_id in response.company_fact_ids_used):
        violations.append("unretrieved_evidence_id")
    if any(fact_id not in retrieved_ids for fact_id in cited_ids):
        violations.append("unretrieved_supported_claim_id")
    if (
        response.company_fact_ids_used
        and not response.supported_claims
        and decision.selected_action.value in CLAIM_ATTRIBUTION_ACTIONS
    ):
        violations.append("missing_supported_claims")
    if response.supported_claims and (
        cited_ids != set(response.company_fact_ids_used)
    ):
        violations.append("claim_id_mismatch")
    if any(phrase in message.casefold() for phrase in UNSUPPORTED_FIT_LANGUAGE):
        violations.append("unsupported_fit_language")
    if HTML_RE.search(message):
        violations.append("html_content")
    if message.startswith("#"):
        violations.append("markdown_heading")
    if "|" in message and "\n" in message:
        violations.append("markdown_table")
    if "```" in message:
        violations.append("code_fence")
    if decision.selected_action.value == "disclose_ai_identity" and "ai" not in message.casefold():
        violations.append("missing_ai_disclosure")
    if decision.selected_action.value == "gracefully_exit" and question_count > 0:
        violations.append("question_after_opt_out")
    if decision.candidate_intent.value == "clear_rejection" and "schedule" in message.casefold():
        violations.append("persuasion_after_rejection")
    for previous in history[-3:]:
        similarity = SequenceMatcher(a=previous.content.casefold(), b=message.casefold()).ratio()
        if similarity > 0.92:
            violations.append("strong_repetition")
            break
    return DeterministicResponseCheck(
        passed=not violations,
        violations=violations,
        question_count=question_count,
        character_count=character_count,
    )
