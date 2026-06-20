"""Configurable fake model gateway for tests."""

import re
from collections.abc import Callable, Sequence
from typing import cast

from psview_agent.domain.agent import (
    AgentConfiguration,
    AgentPersonaDraft,
    CompanyAgentConfigurationDraft,
    OutreachMessageDraft,
    OutreachPlanDraft,
)
from psview_agent.domain.candidate import CandidateProfile
from psview_agent.domain.company import (
    CommunicationProfile,
    CompanyContextInput,
    CompanyCulture,
    CompanyIdentity,
    CompanyProfileDraft,
    EvidenceFactDraft,
    HiringProfile,
    SourceSegment,
)
from psview_agent.domain.conversation import ConversationMessage, ConversationState
from psview_agent.domain.decisions import AgentDecision, AgentDecisionDraft, CandidateAnalysis
from psview_agent.domain.enums import (
    AgentAction,
    CandidateIntent,
    ConversationStage,
    EngagementLevel,
    EvidenceKind,
    OutreachStage,
    Sentiment,
)
from psview_agent.domain.evaluation import (
    GeneratedResponseDraft,
    ResponseEvaluation,
    SupportedClaim,
)
from psview_agent.domain.retrieval import RetrievedEvidence
from psview_agent.integrations.models.protocol import ModelGateway


class FakeModelGateway(ModelGateway):
    """A deterministic fake gateway with optional per-method overrides."""

    def __init__(self, scenarios: dict[str, object] | None = None) -> None:
        self.scenarios = scenarios or {}

    @staticmethod
    def _candidate_experience_summary(candidate: CandidateProfile) -> str:
        summary = candidate.background_summary.strip().rstrip(".")
        summary = re.sub(
            rf"^{re.escape(candidate.name)}\b[\s,:-]*",
            "",
            summary,
            flags=re.IGNORECASE,
        )
        summary = re.split(r"(?<=[.!?])\s+", summary, maxsplit=1)[0].strip(" .")
        replacements = (
            (r"^i have built\b", "building"),
            (r"^i've built\b", "building"),
            (r"^i build\b", "building"),
            (r"^i am building\b", "building"),
            (r"^i'm building\b", "building"),
            (r"^has built\b", "building"),
            (r"^built\b", "building"),
            (r"^has developed\b", "developing"),
            (r"^developed\b", "developing"),
        )
        for pattern, replacement in replacements:
            if re.match(pattern, summary, re.IGNORECASE):
                summary = re.sub(pattern, replacement, summary, count=1, flags=re.IGNORECASE)
                break
        summary = summary.strip()
        if not summary:
            return "in relevant product and engineering work"
        if summary.casefold().startswith(("building ", "developing ")):
            return summary
        if summary.casefold().startswith(("with ", "across ", "in ")):
            return summary
        if len(summary) > 1:
            return f"in {summary[0].lower()}{summary[1:]}"
        return f"in {summary.lower()}"

    async def configure_company_agent(
        self,
        *,
        context: CompanyContextInput,
        source_segments: Sequence[SourceSegment],
    ) -> CompanyAgentConfigurationDraft:
        scenario = self.scenarios.get("configure_company_agent")
        if isinstance(scenario, Exception):
            raise scenario
        if callable(scenario):
            result = cast(Callable[..., object], scenario)(
                context=context,
                source_segments=source_segments,
            )
            return CompanyAgentConfigurationDraft.model_validate(result)
        formal = (
            "bank" in context.company_name.casefold()
            or "financial" in context.company_name.casefold()
            or "financial" in context.company_description.casefold()
        )
        tone = ["formal", "careful"] if formal else ["fast-moving", "curious"]
        return CompanyAgentConfigurationDraft(
            company_profile=CompanyProfileDraft(
                identity=CompanyIdentity(
                    name=context.company_name,
                    summary=context.company_description[:160],
                    industry_or_category="financial services" if formal else "ai software",
                    mission=context.company_description[:160],
                ),
                culture=CompanyCulture(
                    values=["integrity", "quality"] if formal else ["speed", "experimentation"],
                    working_style=["structured"] if formal else ["iterative"],
                    differentiators=["domain expertise"] if formal else ["technical ambition"],
                ),
                hiring_profile=HiringProfile(
                    target_profiles=["builders", "engineers"],
                    desired_signals=["ownership", "communication"],
                    likely_candidate_motivations=["impact", "growth"],
                ),
                communication_profile=CommunicationProfile(
                    tone_attributes=tone,
                    preferred_language_patterns=["clear", "specific"],
                    language_to_avoid=["spammy"],
                ),
            ),
            evidence_facts=[
                EvidenceFactDraft(
                    fact=f"{context.company_name} is hiring for targeted roles.",
                    kind=EvidenceKind.HIRING_PROFILE,
                    source_segment_ids=[source_segments[3].id],
                    retrieval_tags=["hiring", "role"],
                ),
                EvidenceFactDraft(
                    fact=context.recruiting_intent[:150],
                    kind=EvidenceKind.RECRUITING_INSTRUCTION,
                    source_segment_ids=[source_segments[5].id],
                    retrieval_tags=["intent", "growth"],
                ),
                EvidenceFactDraft(
                    fact=context.communication_tone[:150],
                    kind=EvidenceKind.COMMUNICATION_GUIDANCE,
                    source_segment_ids=[source_segments[4].id],
                    retrieval_tags=["tone", "culture"],
                ),
            ],
            persona=AgentPersonaDraft(
                name="Ari",
                role_identity=f"{context.company_name} recruiting assistant",
                personality_summary="A clear and grounded recruiting guide.",
                traits=["thoughtful", "concise", "respectful"],
                communication_principles=["be specific", "stay grounded", "be respectful"],
                questioning_style="Ask at most one focused question when needed.",
                objection_handling_style="Acknowledge concerns without pressure.",
                boundaries=[
                    "Do not invent company facts.",
                    "Do not pressure a rejecting candidate.",
                ],
                language_to_avoid=["guaranteed", "urgent"],
            ),
        )

    async def generate_outreach_plan(
        self,
        *,
        configuration: AgentConfiguration,
        candidate: CandidateProfile,
        target_role: str,
        target_role_description: str | None,
        retrieved_evidence: Sequence[RetrievedEvidence],
    ) -> OutreachPlanDraft:
        scenario = self.scenarios.get("generate_outreach_plan")
        if isinstance(scenario, Exception):
            raise scenario
        if callable(scenario):
            result = cast(Callable[..., object], scenario)(
                configuration=configuration,
                candidate=candidate,
                target_role=target_role,
                target_role_description=target_role_description,
                retrieved_evidence=retrieved_evidence,
            )
            return OutreachPlanDraft.model_validate(result)
        fact_ids = [item.evidence.id for item in retrieved_evidence[:2]]
        candidate_experience = self._candidate_experience_summary(candidate)
        return OutreachPlanDraft(
            overall_intent="Introduce a relevant opportunity with grounded company context.",
            messages=[
                OutreachMessageDraft(
                    stage=OutreachStage.INITIAL_OUTREACH,
                    objective="Introduce the role and candidate relevance.",
                    trigger="Initial contact",
                    message=(
                        f"Hi {candidate.name}, your experience {candidate_experience} "
                        f"stood out for this {target_role} opening. The company is hiring "
                        "builders for targeted roles. Would you be open to a brief conversation?"
                    ),
                    supported_claims=[
                        SupportedClaim(
                            claim="the company is hiring builders for targeted roles",
                            evidence_fact_ids=fact_ids[:1],
                        )
                    ]
                    if fact_ids
                    else [],
                ),
                OutreachMessageDraft(
                    stage=OutreachStage.FOLLOW_UP,
                    objective="Follow up if there is no reply.",
                    trigger="No response",
                    message=(
                        "Following up in case the earlier note got buried. "
                        "The company is hiring builders for targeted roles and I can share more "
                        "context if helpful."
                    ),
                    supported_claims=[
                        SupportedClaim(
                            claim="the company is hiring builders for targeted roles",
                            evidence_fact_ids=fact_ids[:1],
                        )
                    ]
                    if fact_ids
                    else [],
                ),
                OutreachMessageDraft(
                    stage=OutreachStage.FINAL_CLOSEOUT,
                    objective="Close respectfully if there is still no reply.",
                    trigger="No response after follow-up",
                    message="I will close the loop here for now. Thanks for taking a look.",
                    supported_claims=[],
                ),
            ],
        )

    async def analyze_candidate_reply(
        self,
        *,
        configuration: AgentConfiguration,
        candidate: CandidateProfile,
        target_role: str,
        target_role_description: str | None,
        state: ConversationState,
        history: Sequence[ConversationMessage],
        candidate_reply: str,
    ) -> CandidateAnalysis:
        scenario = self.scenarios.get("analyze_candidate_reply")
        if isinstance(scenario, Exception):
            raise scenario
        if callable(scenario):
            result = cast(Callable[..., object], scenario)(
                configuration=configuration,
                candidate=candidate,
                target_role=target_role,
                target_role_description=target_role_description,
                state=state,
                history=history,
                candidate_reply=candidate_reply,
            )
            return CandidateAnalysis.model_validate(result)
        lowered = candidate_reply.casefold()
        intent = CandidateIntent.INTERESTED
        explicit_opt_out = False
        concerns: list[str] = []
        topics: list[str] = []
        motivations: list[str] = []
        if "salary" in lowered or "compensation" in lowered:
            intent = CandidateIntent.ASKS_ABOUT_COMPENSATION
            topics.append("compensation")
        elif "visa" in lowered:
            intent = CandidateIntent.ASKS_ABOUT_VISA_OR_ELIGIBILITY
            topics.append("visa")
        elif "where" in lowered or "remote" in lowered:
            intent = CandidateIntent.ASKS_ABOUT_LOCATION_OR_WORK_MODE
            topics.append("location")
        elif "why" in lowered and "contact" in lowered:
            intent = CandidateIntent.ASKS_WHY_CONTACTED
            concerns.append("generic outreach")
        elif "ai" in lowered or "automated" in lowered:
            intent = CandidateIntent.ASKS_IF_AI_OR_AUTOMATED
            topics.append("ai identity")
        elif "do not contact" in lowered:
            intent = CandidateIntent.DO_NOT_CONTACT
            explicit_opt_out = True
        elif "no thanks" in lowered or "not interested" in lowered:
            intent = CandidateIntent.CLEAR_REJECTION
        elif "right now" in lowered or "busy" in lowered:
            intent = CandidateIntent.BUSY_OR_NOT_READY
        elif "stop wasting my time" in lowered:
            intent = CandidateIntent.HOSTILE
        elif "role" in lowered:
            intent = CandidateIntent.ASKS_ABOUT_ROLE
            topics.append("role")
        if "interested" in lowered:
            motivations.append("interest")
        return CandidateAnalysis(
            intent=intent,
            sentiment=(
                Sentiment.NEUTRAL if intent is not CandidateIntent.HOSTILE else Sentiment.NEGATIVE
            ),
            engagement_level=EngagementLevel.MEDIUM,
            observed_signals=["keyword_match"],
            expressed_motivations=motivations,
            candidate_concerns=concerns,
            questions_or_topics=topics,
            explicit_opt_out=explicit_opt_out,
            reply_summary=candidate_reply[:120],
            retrieval_topics=topics or ["role"],
            confidence=0.82,
        )

    async def plan_next_action(
        self,
        *,
        configuration: AgentConfiguration,
        candidate: CandidateProfile,
        target_role: str,
        target_role_description: str | None,
        state: ConversationState,
        analysis: CandidateAnalysis,
        history: Sequence[ConversationMessage],
        retrieved_evidence: Sequence[RetrievedEvidence],
    ) -> AgentDecisionDraft:
        scenario = self.scenarios.get("plan_next_action")
        if isinstance(scenario, Exception):
            raise scenario
        if callable(scenario):
            result = cast(Callable[..., object], scenario)(
                configuration=configuration,
                candidate=candidate,
                target_role=target_role,
                target_role_description=target_role_description,
                state=state,
                analysis=analysis,
                history=history,
                retrieved_evidence=retrieved_evidence,
            )
            return AgentDecisionDraft.model_validate(result)
        action = (
            AgentAction.ANSWER_CANDIDATE_QUESTION
            if analysis.questions_or_topics
            else AgentAction.ASK_DISCOVERY_QUESTION
        )
        return AgentDecisionDraft(
            current_stage=state.stage,
            proposed_next_stage=ConversationStage.INFORMATION_EXCHANGE,
            objective="Answer the candidate directly and keep the conversation grounded.",
            proposed_action=action,
            company_fact_ids_to_use=[item.evidence.id for item in retrieved_evidence[:2]],
            missing_information=[],
            should_continue=True,
            should_ask_question=action is AgentAction.ASK_DISCOVERY_QUESTION,
            rationale_summary="Use relevant evidence and address the candidate's latest topic.",
            confidence=0.8,
        )

    async def generate_candidate_response(
        self,
        *,
        configuration: AgentConfiguration,
        candidate: CandidateProfile,
        target_role: str,
        target_role_description: str | None,
        state: ConversationState,
        decision: AgentDecision,
        history: Sequence[ConversationMessage],
        candidate_reply: str,
    ) -> GeneratedResponseDraft:
        scenario = self.scenarios.get("generate_candidate_response")
        if isinstance(scenario, Exception):
            raise scenario
        if callable(scenario):
            result = cast(Callable[..., object], scenario)(
                configuration=configuration,
                candidate=candidate,
                target_role=target_role,
                target_role_description=target_role_description,
                state=state,
                decision=decision,
                history=history,
                candidate_reply=candidate_reply,
            )
            return GeneratedResponseDraft.model_validate(result)
        if decision.selected_action is AgentAction.DISCLOSE_AI_IDENTITY:
            message = "I am an AI recruiting assistant working on the company's behalf."
            supported_claims: list[SupportedClaim] = []
        elif decision.selected_action is AgentAction.GRACEFULLY_EXIT:
            message = "Understood. I will close the loop here respectfully."
            supported_claims = []
        elif decision.selected_action is AgentAction.CLARIFY_MISSING_INFORMATION:
            missing = (
                decision.missing_information[0] if decision.missing_information else "that topic"
            )
            message = (
                f"The supplied context does not include confirmed information about {missing}."
            )
            supported_claims = []
        elif decision.selected_action is AgentAction.PAUSE_RESPECTFULLY:
            message = "Thanks for the note. I will pause here and leave this with you."
            supported_claims = []
        else:
            message = (
                "Thanks for the question. Based on the supplied company context, "
                "this role is aimed at strong builders with relevant experience."
            )
            supported_claims = []
            if decision.company_fact_ids_to_use:
                supported_claims = [
                    SupportedClaim(
                        claim="this role is aimed at strong builders with relevant experience",
                        evidence_fact_ids=decision.company_fact_ids_to_use[:1],
                    )
                ]
        return GeneratedResponseDraft(
            message=message,
            supported_claims=supported_claims,
        )

    async def evaluate_candidate_response(
        self,
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
    ) -> ResponseEvaluation:
        scenario = self.scenarios.get("evaluate_candidate_response")
        if isinstance(scenario, Exception):
            raise scenario
        if callable(scenario):
            result = cast(Callable[..., object], scenario)(
                configuration=configuration,
                candidate=candidate,
                target_role=target_role,
                target_role_description=target_role_description,
                state=state,
                decision=decision,
                history=history,
                candidate_reply=candidate_reply,
                response=response,
            )
            return ResponseEvaluation.model_validate(result)
        passed = "bad" not in response.message.casefold()
        return ResponseEvaluation(
            personality_consistency=0.9 if passed else 0.4,
            company_grounding=0.9 if passed else 0.4,
            candidate_relevance=0.9 if passed else 0.4,
            action_alignment=0.9 if passed else 0.4,
            conversational_naturalness=0.85 if passed else 0.4,
            repetition_risk=0.1 if passed else 0.8,
            unsupported_claims=[],
            personality_violations=[],
            policy_violations=[],
            passed=passed,
            revision_instructions=["be more grounded"] if not passed else [],
        )

    async def revise_candidate_response(
        self,
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
    ) -> GeneratedResponseDraft:
        scenario = self.scenarios.get("revise_candidate_response")
        if isinstance(scenario, Exception):
            raise scenario
        if callable(scenario):
            result = cast(Callable[..., object], scenario)(
                configuration=configuration,
                candidate=candidate,
                target_role=target_role,
                target_role_description=target_role_description,
                state=state,
                decision=decision,
                history=history,
                candidate_reply=candidate_reply,
                response=response,
                evaluation=evaluation,
                deterministic_violations=deterministic_violations,
            )
            return GeneratedResponseDraft.model_validate(result)
        return GeneratedResponseDraft(
            message=(
                "Thanks for the question. I will stay grounded in the supplied company context."
            ),
            supported_claims=response.supported_claims,
        )
