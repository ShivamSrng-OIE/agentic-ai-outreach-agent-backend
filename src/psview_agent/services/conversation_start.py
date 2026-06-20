"""Conversation-start service."""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence

from psview_agent.core.errors import (
    InvalidCompanyEvidenceError,
    ModelIncompleteResponseError,
    ModelInvalidOutputError,
)
from psview_agent.domain.agent import AgentConfiguration, OutreachMessage, OutreachPlan
from psview_agent.domain.candidate import CandidateProfile
from psview_agent.domain.company import EvidenceFact
from psview_agent.domain.conversation import (
    ConversationMessage,
    ConversationSession,
    ConversationState,
)
from psview_agent.domain.decisions import DecisionTrace
from psview_agent.domain.enums import (
    AgentAction,
    CandidateIntent,
    ConversationStage,
    EvidenceKind,
    MessageRole,
    OutreachStage,
    Sentiment,
)
from psview_agent.domain.evaluation import SupportedClaim
from psview_agent.domain.retrieval import RetrievedEvidence
from psview_agent.integrations.models.protocol import ModelGateway
from psview_agent.retrieval.protocol import EvidenceRetriever
from psview_agent.retrieval.query_builder import build_initial_retrieval_query
from psview_agent.retrieval.scoring import build_fact_document
from psview_agent.retrieval.tokenization import tokenize_text
from psview_agent.utils.identifiers import new_uuid, prefixed_sequence_id
from psview_agent.utils.text import normalize_whitespace, safe_truncate, sanitize_generated_text
from psview_agent.utils.time import utc_now

LOGGER = logging.getLogger(__name__)
UNSUPPORTED_FIT_LANGUAGE = ("strong fit", "perfect fit", "ideal fit")
CLAIM_SUPPORT_THRESHOLD = 0.6
MAX_INITIAL_OUTREACH_CHARS = 1440
MAX_FOLLOW_UP_CHARS = 720
MAX_CLOSEOUT_CHARS = 560
OUTREACH_ALLOWED_EVIDENCE_KINDS = {
    EvidenceKind.COMPANY_IDENTITY,
    EvidenceKind.PRODUCT,
    EvidenceKind.CULTURE,
    EvidenceKind.WORKING_STYLE,
    EvidenceKind.HIRING_PROFILE,
    EvidenceKind.ROLE_INFORMATION,
}
SECURITY_FACT_TERMS = {
    "aes",
    "aws",
    "compliance",
    "encryption",
    "encrypted",
    "gdpr",
    "privacy",
    "security",
}
SECURITY_CONTEXT_TERMS = {
    "compliance",
    "encryption",
    "gdpr",
    "privacy",
    "risk",
    "secure",
    "security",
    "trust",
}
BROKEN_BACKGROUND_PATTERNS = (
    re.compile(r"\byour experience with (?:he|she|they) has\b", re.IGNORECASE),
    re.compile(r"\byour background in (?:he|she|they) has\b", re.IGNORECASE),
    re.compile(r"\.\s+stood out\b", re.IGNORECASE),
)


class ConversationStartService:
    """Build the outreach preview and initial session state."""

    def __init__(
        self,
        *,
        gateway: ModelGateway,
        retriever: EvidenceRetriever,
        retrieval_limit: int = 5,
    ) -> None:
        self._gateway = gateway
        self._retriever = retriever
        self._retrieval_limit = retrieval_limit

    async def start_conversation(
        self,
        *,
        configuration: AgentConfiguration,
        candidate: CandidateProfile,
        target_role: str,
        target_role_description: str | None = None,
    ) -> tuple[ConversationSession, DecisionTrace]:
        query = build_initial_retrieval_query(
            configuration_context=configuration.company_context,
            candidate=candidate,
            target_role=target_role,
            target_role_description=target_role_description,
        )
        retrieved = self._retriever.retrieve(
            corpus=configuration.evidence_corpus,
            query=query,
            already_used_fact_ids=(),
            limit=self._retrieval_limit,
        )
        plan = await self._build_outreach_plan(
            configuration=configuration,
            candidate=candidate,
            target_role=target_role,
            target_role_description=target_role_description,
            retrieved_evidence=retrieved,
        )
        now = utc_now()
        initial_message = ConversationMessage(
            id=new_uuid(),
            role=MessageRole.AGENT,
            content=plan.messages[0].message,
            created_at=now,
        )
        state = ConversationState(
            stage=ConversationStage.INITIAL_OUTREACH,
            company_fact_ids_already_used=plan.messages[0].company_fact_ids_used,
            last_action=AgentAction.INTRODUCE_OPPORTUNITY,
        )
        session = ConversationSession(
            conversation_id=new_uuid(),
            configuration=configuration,
            candidate=candidate,
            target_role=target_role,
            target_role_description=target_role_description,
            outreach_plan=plan,
            state=state,
            messages=[initial_message],
            created_at=now,
            updated_at=now,
        )
        used_facts = [
            fact
            for fact in configuration.evidence_corpus.evidence_facts
            if fact.id in plan.messages[0].company_fact_ids_used
        ]
        confidence = self._initial_confidence(
            candidate=candidate,
            target_role=target_role,
            target_role_description=target_role_description,
            used_facts=used_facts,
            retrieved_evidence=retrieved,
        )
        trace = DecisionTrace(
            candidate_intent=CandidateIntent.UNCLEAR,
            sentiment=Sentiment.NEUTRAL,
            engagement_level=state.engagement_level,
            current_stage=ConversationStage.INITIAL_OUTREACH,
            next_stage=ConversationStage.INITIAL_OUTREACH,
            objective=plan.messages[0].objective,
            selected_action=AgentAction.INTRODUCE_OPPORTUNITY,
            observed_signals=[],
            candidate_concerns=[],
            retrieved_company_facts=retrieved,
            company_facts_used=used_facts,
            missing_information=[],
            should_continue=True,
            confidence=confidence,
            rationale_summary=self._build_initial_rationale(
                candidate=candidate,
                target_role=target_role,
                target_role_description=target_role_description,
                used_facts=used_facts,
                retrieved_evidence=retrieved,
                confidence=confidence,
            ),
            policy_overrides=[],
        )
        return session, trace

    async def _build_outreach_plan(
        self,
        *,
        configuration: AgentConfiguration,
        candidate: CandidateProfile,
        target_role: str,
        target_role_description: str | None,
        retrieved_evidence: Sequence[RetrievedEvidence],
    ) -> OutreachPlan:
        try:
            curated_evidence = self._curate_outreach_evidence(
                configuration=configuration,
                candidate=candidate,
                target_role=target_role,
                target_role_description=target_role_description,
                retrieved_evidence=retrieved_evidence,
            )
            plan_draft = await self._gateway.generate_outreach_plan(
                configuration=configuration,
                candidate=candidate,
                target_role=target_role,
                target_role_description=target_role_description,
                retrieved_evidence=curated_evidence,
            )
            repaired_messages = [
                self._repair_message_supported_claims(
                    configuration=configuration,
                    retrieved_evidence=curated_evidence,
                    message=OutreachMessage(
                        id=prefixed_sequence_id("outreach", index),
                        **message.model_dump(),
                    ),
                )
                for index, message in enumerate(plan_draft.messages, start=1)
            ]
            return self._validate_and_materialize_plan(
                configuration=configuration,
                candidate=candidate,
                target_role=target_role,
                target_role_description=target_role_description,
                overall_intent=plan_draft.overall_intent,
                messages=repaired_messages,
            )
        except (InvalidCompanyEvidenceError, ModelIncompleteResponseError, ModelInvalidOutputError):
            LOGGER.warning(
                "using deterministic outreach fallback",
                extra={"error_category": "outreach_plan_fallback"},
            )
            return self._build_fallback_outreach_plan(
                configuration=configuration,
                candidate=candidate,
                target_role=target_role,
                target_role_description=target_role_description,
                retrieved_evidence=retrieved_evidence,
            )

    def _validate_and_materialize_plan(
        self,
        *,
        configuration: AgentConfiguration,
        candidate: CandidateProfile,
        target_role: str,
        target_role_description: str | None,
        overall_intent: str,
        messages: Sequence[OutreachMessage],
    ) -> OutreachPlan:
        expected_order = [
            OutreachStage.INITIAL_OUTREACH,
            OutreachStage.FOLLOW_UP,
            OutreachStage.FINAL_CLOSEOUT,
        ]
        if len(messages) != 3 or [item.stage for item in messages] != expected_order:
            raise InvalidCompanyEvidenceError(
                "outreach plan must contain exactly three ordered messages"
            )
        valid_fact_ids = {fact.id for fact in configuration.evidence_corpus.evidence_facts}
        polished_messages: list[OutreachMessage] = []
        for message in messages:
            message = message.model_copy(
                update={"message": self._polish_outreach_text(message.message)}
            )
            if (
                message.stage in {OutreachStage.INITIAL_OUTREACH, OutreachStage.FOLLOW_UP}
                and not message.supported_claims
            ):
                raise InvalidCompanyEvidenceError(
                    "factual outreach messages must include supported_claims"
                )
            unknown_ids = [
                fact_id
                for fact_id in message.company_fact_ids_used
                if fact_id not in valid_fact_ids
            ]
            if unknown_ids:
                raise InvalidCompanyEvidenceError(
                    f"outreach message references unknown fact IDs: {', '.join(unknown_ids)}"
                )
            for claim in message.supported_claims:
                unsupported_ids = [
                    fact_id
                    for fact_id in claim.evidence_fact_ids
                    if not self._claim_is_supported_by_fact(
                        configuration=configuration,
                        claim=claim.claim,
                        fact_id=fact_id,
                    )
                ]
                if unsupported_ids:
                    raise InvalidCompanyEvidenceError(
                        "outreach claim cites evidence that does not support it"
                    )
                disallowed_ids = [
                    fact_id
                    for fact_id in claim.evidence_fact_ids
                    if not self._fact_allowed_for_outreach(
                        configuration=configuration,
                        candidate=candidate,
                        target_role=target_role,
                        target_role_description=target_role_description,
                        fact_id=fact_id,
                    )
                ]
                if disallowed_ids:
                    raise InvalidCompanyEvidenceError(
                        "outreach claim cites evidence that is not appropriate for outreach"
                    )
            if any(phrase in message.message.casefold() for phrase in UNSUPPORTED_FIT_LANGUAGE):
                raise InvalidCompanyEvidenceError("outreach message uses unsupported fit language")
            if self._has_broken_candidate_phrasing(message.message):
                raise InvalidCompanyEvidenceError(
                    "outreach message contains broken candidate phrasing"
                )
            if self._has_raw_text_injection(
                message=message.message,
                candidate=candidate,
                target_role_description=target_role_description,
            ):
                raise InvalidCompanyEvidenceError(
                    "outreach message contains raw pasted candidate or role text"
                )
            if self._question_count(message.message) > 1:
                raise InvalidCompanyEvidenceError("outreach message asks more than one question")
            sentence_limit = 4 if message.stage is OutreachStage.INITIAL_OUTREACH else 3
            if self._sentence_count(message.message) > sentence_limit:
                raise InvalidCompanyEvidenceError("outreach message exceeds the sentence limit")
            if len(message.message) > self._max_message_length(message.stage):
                raise InvalidCompanyEvidenceError("outreach message is too long")
            polished_messages.append(message)
        return OutreachPlan(overall_intent=overall_intent, messages=polished_messages)

    def _build_fallback_outreach_plan(
        self,
        *,
        configuration: AgentConfiguration,
        candidate: CandidateProfile,
        target_role: str,
        target_role_description: str | None,
        retrieved_evidence: Sequence[RetrievedEvidence],
    ) -> OutreachPlan:
        curated_evidence = self._curate_outreach_evidence(
            configuration=configuration,
            candidate=candidate,
            target_role=target_role,
            target_role_description=target_role_description,
            retrieved_evidence=retrieved_evidence,
        )
        support_facts = [item.evidence for item in curated_evidence[:2]]
        if not support_facts:
            support_facts = [
                fact
                for fact in configuration.evidence_corpus.evidence_facts
                if self._fact_allowed_for_outreach(
                    configuration=configuration,
                    candidate=candidate,
                    target_role=target_role,
                    target_role_description=target_role_description,
                    fact_id=fact.id,
                )
            ][:2]
        company_name = configuration.company_context.company_name
        support_ids = [fact.id for fact in support_facts[:2]]
        candidate_experience = self._candidate_experience_summary(candidate)
        role_phrase = self._role_relevance_phrase(target_role_description)
        primary_fact = (
            support_facts[0].fact
            if support_facts
            else configuration.company_context.company_description
        )
        primary_sentence = self._candidate_facing_fact_sentence(
            company_name=company_name,
            fact=primary_fact,
        )
        intro = sanitize_generated_text(
            f"Hi {candidate.name}, I noticed your experience {candidate_experience} "
            "and thought it was worth reaching out. "
            f"For {company_name}'s {target_role} role, that background seems especially "
            f"relevant to {role_phrase}. "
            f"{primary_sentence} "
            "Would you be open to a brief conversation?"
        )
        follow_up = sanitize_generated_text(
            f"Following up in case my earlier note got buried. {primary_sentence} "
            "Happy to share more context if helpful."
        )
        closeout = sanitize_generated_text(
            f"I'll close the loop for now. If {company_name}'s work and the {target_role} "
            "opportunity become relevant later, I'd be glad to reconnect."
        )
        messages = [
            OutreachMessage(
                id=prefixed_sequence_id("outreach", 1),
                stage=OutreachStage.INITIAL_OUTREACH,
                objective="Open a relevant conversation with a specific reason for reaching out.",
                trigger="candidate background appears relevant",
                message=intro,
                supported_claims=[
                    {
                        "claim": primary_sentence.rstrip("."),
                        "evidence_fact_ids": support_ids[:1],
                    }
                ],
            ),
            OutreachMessage(
                id=prefixed_sequence_id("outreach", 2),
                stage=OutreachStage.FOLLOW_UP,
                objective="Follow up briefly and politely after no response.",
                trigger="no reply to the initial note",
                message=follow_up,
                supported_claims=[
                    {
                        "claim": primary_sentence.rstrip("."),
                        "evidence_fact_ids": support_ids[:1],
                    }
                ],
            ),
            OutreachMessage(
                id=prefixed_sequence_id("outreach", 3),
                stage=OutreachStage.FINAL_CLOSEOUT,
                objective="Close the loop gracefully while leaving the door open.",
                trigger="no reply after the follow-up",
                message=closeout,
                supported_claims=[],
            ),
        ]
        return self._validate_and_materialize_plan(
            configuration=configuration,
            candidate=candidate,
            target_role=target_role,
            target_role_description=target_role_description,
            overall_intent="Introduce a grounded opportunity and leave the door open politely.",
            messages=messages,
        )

    def _build_initial_rationale(
        self,
        *,
        candidate: CandidateProfile,
        target_role: str,
        target_role_description: str | None,
        used_facts: Sequence[EvidenceFact],
        retrieved_evidence: Sequence[RetrievedEvidence],
        confidence: float,
    ) -> str:
        confidence_percent = round(confidence * 100)
        role_match = self._candidate_role_overlap(
            candidate=candidate,
            target_role=target_role,
            target_role_description=target_role_description,
        )
        if used_facts:
            experience = self._candidate_experience_summary(candidate)
            return (
                f"{candidate.name}'s experience {experience} connects to the role context "
                f"and {self._summarize_used_facts(used_facts)}. Confidence is "
                f"{confidence_percent}% because {len(used_facts)} grounded fact(s) were used, "
                f"the top retrieval relevance was {self._top_relevance(retrieved_evidence):.0%}, "
                f"and candidate-role overlap was {role_match:.0%}."
            )
        return (
            f"{candidate.name}'s {candidate.current_role} background is being used to open a "
            f"careful {target_role} outreach without unsupported company claims. Confidence is "
            f"{confidence_percent}% because no candidate-facing company fact was selected."
        )

    def _candidate_experience_summary(self, candidate: CandidateProfile) -> str:
        summary = normalize_whitespace(candidate.background_summary).strip()
        summary = sanitize_generated_text(summary).strip(" .")
        if not summary:
            return "in relevant product and engineering work"
        name_pattern = re.compile(rf"^{re.escape(candidate.name)}\b[\s,:-]*", re.IGNORECASE)
        summary = name_pattern.sub("", summary).strip()
        summary = re.split(r"(?<=[.!?])\s+", summary, maxsplit=1)[0].strip(" .")
        replacements = (
            (r"^i have built\b", "building"),
            (r"^i've built\b", "building"),
            (r"^i build\b", "building"),
            (r"^i am building\b", "building"),
            (r"^i'm building\b", "building"),
            (r"^i have developed\b", "developing"),
            (r"^i developed\b", "developing"),
            (r"^i develop\b", "developing"),
            (r"^i have shipped\b", "shipping"),
            (r"^i shipped\b", "shipping"),
            (r"^i ship\b", "shipping"),
            (r"^has built\b", "building"),
            (r"^built\b", "building"),
            (r"^has developed\b", "developing"),
            (r"^developed\b", "developing"),
            (r"^has led\b", "leading"),
            (r"^led\b", "leading"),
            (r"^has shipped\b", "shipping"),
            (r"^shipped\b", "shipping"),
            (r"^has worked on\b", "working on"),
            (r"^worked on\b", "working on"),
            (r"^has created\b", "creating"),
            (r"^created\b", "creating"),
            (r"^has owned\b", "owning"),
            (r"^owned\b", "owning"),
        )
        for pattern, replacement in replacements:
            if re.match(pattern, summary, re.IGNORECASE):
                summary = re.sub(pattern, replacement, summary, count=1, flags=re.IGNORECASE)
                break
        summary = normalize_whitespace(safe_truncate(summary, 120)).strip(" .")
        if not summary:
            return "in relevant product and engineering work"
        if summary.casefold().startswith(("building ", "developing ", "leading ", "shipping ")):
            return summary
        if summary.casefold().startswith(("working on ", "creating ", "owning ")):
            return summary
        if summary.casefold().startswith(("with ", "across ", "in ")):
            return summary
        if len(summary) > 1:
            return f"in {summary[0].lower()}{summary[1:]}"
        return f"in {summary.lower()}"

    def _role_relevance_phrase(self, target_role_description: str | None) -> str:
        if not target_role_description:
            return "the role's core engineering needs"
        tokens = tokenize_text(target_role_description)
        priority_terms = [
            token
            for token in tokens
            if token
            not in {
                "founding",
                "engineer",
                "engineering",
                "role",
                "startup",
                "systems",
            }
        ]
        chosen = []
        for token in priority_terms:
            if token not in chosen:
                chosen.append(token)
            if len(chosen) == 3:
                break
        if not chosen:
            return "the role's core engineering needs"
        if len(chosen) == 1:
            return chosen[0]
        if len(chosen) == 2:
            return f"{chosen[0]} and {chosen[1]}"
        return f"{chosen[0]}, {chosen[1]}, and {chosen[2]}"

    def _initial_confidence(
        self,
        *,
        candidate: CandidateProfile,
        target_role: str,
        target_role_description: str | None,
        used_facts: Sequence[EvidenceFact],
        retrieved_evidence: Sequence[RetrievedEvidence],
    ) -> float:
        fact_score = min(len(used_facts), 2) * 0.12
        relevance_score = 0.18 * self._top_relevance(retrieved_evidence)
        role_overlap = self._candidate_role_overlap(
            candidate=candidate,
            target_role=target_role,
            target_role_description=target_role_description,
        )
        role_score = 0.2 * role_overlap
        description_score = 0.08 if target_role_description else 0.0
        confidence = 0.48 + fact_score + relevance_score + role_score + description_score
        return round(min(0.95, max(0.35, confidence)), 2)

    def _top_relevance(self, retrieved_evidence: Sequence[RetrievedEvidence]) -> float:
        if not retrieved_evidence:
            return 0.0
        return max(item.normalized_relevance for item in retrieved_evidence)

    def _candidate_role_overlap(
        self,
        *,
        candidate: CandidateProfile,
        target_role: str,
        target_role_description: str | None,
    ) -> float:
        candidate_tokens = set(
            tokenize_text(
                " ".join(
                    [
                        candidate.current_role,
                        candidate.background_summary,
                    ]
                )
            )
        )
        role_tokens = set(tokenize_text(" ".join([target_role, target_role_description or ""])))
        if not candidate_tokens or not role_tokens:
            return 0.0
        return len(candidate_tokens & role_tokens) / len(role_tokens)

    def _has_broken_candidate_phrasing(self, message: str) -> bool:
        normalized = normalize_whitespace(message)
        if any(pattern.search(normalized) for pattern in BROKEN_BACKGROUND_PATTERNS):
            return True
        return bool(
            re.search(
                r"\b(?:your experience with|your background in)\s+[A-Z][a-z]+\s+has\b",
                normalized,
            )
        )

    def _curate_outreach_evidence(
        self,
        *,
        configuration: AgentConfiguration,
        candidate: CandidateProfile,
        target_role: str,
        target_role_description: str | None,
        retrieved_evidence: Sequence[RetrievedEvidence],
    ) -> list[RetrievedEvidence]:
        return [
            item
            for item in retrieved_evidence
            if self._fact_allowed_for_outreach(
                configuration=configuration,
                candidate=candidate,
                target_role=target_role,
                target_role_description=target_role_description,
                fact_id=item.evidence.id,
            )
        ]

    def _fact_allowed_for_outreach(
        self,
        *,
        configuration: AgentConfiguration,
        candidate: CandidateProfile,
        target_role: str,
        target_role_description: str | None,
        fact_id: str,
    ) -> bool:
        fact_lookup = {fact.id: fact for fact in configuration.evidence_corpus.evidence_facts}
        fact = fact_lookup.get(fact_id)
        if fact is None or fact.kind not in OUTREACH_ALLOWED_EVIDENCE_KINDS:
            return False
        fact_terms = set(tokenize_text(fact.fact))
        if SECURITY_FACT_TERMS & fact_terms:
            context = " ".join(
                [
                    candidate.current_role,
                    candidate.background_summary,
                    target_role,
                    target_role_description or "",
                ]
            )
            if not (SECURITY_CONTEXT_TERMS & set(tokenize_text(context))):
                return False
        return True

    def _candidate_facing_fact_sentence(self, *, company_name: str, fact: str) -> str:
        text = sanitize_generated_text(fact).strip(" .")
        signals_prefix = "strong candidate signals likely include "
        lowered = text.casefold()
        if lowered.startswith(signals_prefix):
            signals = text[len(signals_prefix) :].strip(" .")
            return f"{company_name} appears to value {signals}."
        if lowered.startswith("we hire "):
            hiring = text[len("we hire ") :].strip(" .")
            return f"{company_name} hires {hiring}."
        if lowered.startswith(company_name.casefold()):
            return text + "."
        return safe_truncate(text, 180).rstrip(".") + "."

    def _polish_outreach_text(self, message: str) -> str:
        polished = sanitize_generated_text(message)
        polished = re.sub(r"\s+([?.!,])", r"\1", polished)
        polished = re.sub(r"([?.!]){2,}", r"\1", polished)
        return polished.strip()

    def _has_raw_text_injection(
        self,
        *,
        message: str,
        candidate: CandidateProfile,
        target_role_description: str | None,
    ) -> bool:
        return self._contains_raw_word_window(message, candidate.background_summary) or (
            target_role_description is not None
            and self._contains_raw_word_window(message, target_role_description)
        )

    def _contains_raw_word_window(self, message: str, raw_text: str) -> bool:
        message_words = " ".join(tokenize_text(sanitize_generated_text(message)))
        raw_words = tokenize_text(sanitize_generated_text(raw_text))
        if len(raw_words) < 8:
            return False
        for window_size in (12, 10, 8):
            if len(raw_words) < window_size:
                continue
            for index in range(len(raw_words) - window_size + 1):
                window = " ".join(raw_words[index : index + window_size])
                if window and window in message_words:
                    return True
        return False

    def _sentence_count(self, message: str) -> int:
        sentences = [
            sentence for sentence in re.split(r"(?<=[.!?])\s+", message.strip()) if sentence.strip()
        ]
        return max(1, len(sentences))

    def _question_count(self, message: str) -> int:
        return message.count("?")

    def _max_message_length(self, stage: OutreachStage) -> int:
        if stage is OutreachStage.INITIAL_OUTREACH:
            return MAX_INITIAL_OUTREACH_CHARS
        if stage is OutreachStage.FOLLOW_UP:
            return MAX_FOLLOW_UP_CHARS
        return MAX_CLOSEOUT_CHARS

    def _repair_message_supported_claims(
        self,
        *,
        configuration: AgentConfiguration,
        retrieved_evidence: Sequence[RetrievedEvidence],
        message: OutreachMessage,
    ) -> OutreachMessage:
        available_facts = {item.evidence.id: item.evidence for item in retrieved_evidence}
        repaired_claims: list[SupportedClaim] = []
        for claim in message.supported_claims:
            repaired_ids = self._repair_claim_evidence_ids(
                configuration=configuration,
                available_facts=available_facts,
                claim=claim.claim,
                cited_ids=claim.evidence_fact_ids,
            )
            if not repaired_ids:
                continue
            repaired_claims.append(claim.model_copy(update={"evidence_fact_ids": repaired_ids}))
        return message.model_copy(update={"supported_claims": repaired_claims})

    def _repair_claim_evidence_ids(
        self,
        *,
        configuration: AgentConfiguration,
        available_facts: dict[str, EvidenceFact],
        claim: str,
        cited_ids: Sequence[str],
    ) -> list[str]:
        supported_cited_ids = [
            fact_id
            for fact_id in cited_ids
            if fact_id in available_facts
            and self._claim_is_supported_by_fact(
                configuration=configuration,
                claim=claim,
                fact_id=fact_id,
            )
        ]
        if supported_cited_ids:
            return supported_cited_ids
        best_fact_id, best_score = self._best_supporting_fact(
            configuration=configuration,
            claim=claim,
            available_facts=available_facts,
        )
        if best_fact_id is None or best_score < CLAIM_SUPPORT_THRESHOLD:
            return []
        return [best_fact_id]

    def _best_supporting_fact(
        self,
        *,
        configuration: AgentConfiguration,
        claim: str,
        available_facts: dict[str, EvidenceFact],
    ) -> tuple[str | None, float]:
        best_fact_id: str | None = None
        best_score = 0.0
        for fact_id, fact in available_facts.items():
            score = self._claim_support_score(
                configuration=configuration,
                claim=claim,
                fact=fact,
            )
            if score > best_score or (
                score == best_score and best_fact_id is not None and fact_id < best_fact_id
            ):
                best_fact_id = fact_id
                best_score = score
        return best_fact_id, best_score

    def _claim_is_supported_by_fact(
        self,
        *,
        configuration: AgentConfiguration,
        claim: str,
        fact_id: str,
    ) -> bool:
        fact_lookup = {fact.id: fact for fact in configuration.evidence_corpus.evidence_facts}
        fact = fact_lookup.get(fact_id)
        if fact is None:
            return False
        return (
            self._claim_support_score(
                configuration=configuration,
                claim=claim,
                fact=fact,
            )
            >= CLAIM_SUPPORT_THRESHOLD
        )

    def _claim_support_score(
        self,
        *,
        configuration: AgentConfiguration,
        claim: str,
        fact: EvidenceFact,
    ) -> float:
        normalized_claim = normalize_whitespace(claim).casefold()
        normalized_fact = normalize_whitespace(fact.fact).casefold()
        if normalized_claim == normalized_fact:
            return 1.0
        if normalized_claim and normalized_claim in normalized_fact:
            return 0.95
        claim_tokens = set(tokenize_text(normalized_claim))
        if not claim_tokens:
            return 0.0
        document_tokens = set(
            tokenize_text(build_fact_document(configuration.evidence_corpus, fact))
        )
        if not document_tokens:
            return 0.0
        overlap = claim_tokens & document_tokens
        coverage = len(overlap) / len(claim_tokens)
        precision = len(overlap) / len(document_tokens)
        return round((0.8 * coverage) + (0.2 * precision), 4)

    def _summarize_used_facts(self, used_facts: Sequence[EvidenceFact]) -> str:
        summaries: list[str] = []
        for fact in used_facts[:2]:
            if fact.kind is EvidenceKind.PRODUCT:
                summaries.append("the company's AI workflow product")
            elif fact.kind is EvidenceKind.HIRING_PROFILE:
                summaries.append("its preference for engineers who ship across functions")
            else:
                summaries.append(safe_truncate(fact.fact, 80))
        if not summaries:
            return "the retrieved company evidence"
        if len(summaries) == 1:
            return summaries[0]
        return f"{summaries[0]} and {summaries[1]}"
