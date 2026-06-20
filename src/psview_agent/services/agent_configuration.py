"""Company configuration service."""

from __future__ import annotations

import logging
from collections.abc import Sequence

from psview_agent.core.errors import ModelIncompleteResponseError, ModelInvalidOutputError
from psview_agent.domain.agent import (
    AgentConfiguration,
    AgentPersona,
    AgentPersonaDraft,
    CompanyAgentConfigurationDraft,
)
from psview_agent.domain.company import (
    CommunicationProfile,
    CompanyContextInput,
    CompanyCulture,
    CompanyIdentity,
    CompanyProfile,
    CompanyProfileDraft,
    EvidenceCorpus,
    EvidenceFactDraft,
    HiringProfile,
    SourceSegment,
)
from psview_agent.domain.enums import EvidenceKind, SourceField
from psview_agent.integrations.models.protocol import ModelGateway
from psview_agent.retrieval.corpus_builder import build_evidence_corpus, segment_company_context
from psview_agent.utils.collections import dedupe_casefold
from psview_agent.utils.identifiers import new_uuid
from psview_agent.utils.text import normalize_whitespace, safe_truncate
from psview_agent.utils.time import utc_now

LOGGER = logging.getLogger(__name__)

MANDATORY_BOUNDARIES = [
    "Do not invent company or role facts.",
    "Do not invent compensation.",
    "Do not invent visa or sponsorship policy.",
    "Do not invent location or work-mode policy.",
    "Do not invent benefits, funding, customers, revenue, or team size.",
    "Do not pressure a rejecting candidate.",
    "Stop after an explicit do-not-contact request.",
    "Do not claim to be human when directly asked.",
    "Ask no more than one question per message.",
    "Do not promise an interview, offer, or employment outcome.",
    "Keep candidate-facing messages concise.",
]

WORKING_STYLE_RULES = (
    ("ship product", "ship product"),
    ("communicate well", "clear communication"),
    ("work across functions", "cross-functional collaboration"),
    ("ownership", "ownership"),
    ("clarity", "clarity"),
    ("curiosity", "curiosity"),
    ("respectful collaboration", "respectful collaboration"),
)

DIFFERENTIATOR_RULES = (
    ("ai workflow software", "AI workflow software"),
    ("reliable automation", "reliable automation"),
    ("thoughtful product delivery", "thoughtful product delivery"),
    ("real customer problems", "real customer problems"),
    ("long-term execution", "thoughtful long-term execution"),
)

DESIRED_SIGNAL_RULES = (
    ("ship product", "ability to ship product"),
    ("communicate well", "clear communication"),
    ("work across functions", "cross-functional collaboration"),
    ("ownership", "ownership"),
    ("clarity", "clarity"),
    ("curiosity", "curiosity"),
)

MOTIVATION_RULES = (
    ("product-minded", "product-minded engineering"),
    ("growth plans", "growth opportunities"),
    ("real customer problems", "customer impact"),
    ("reliable automation", "building reliable automation"),
    ("thoughtful product delivery", "thoughtful product delivery"),
    ("long-term execution", "long-term ownership"),
)

LANGUAGE_PATTERN_RULES = (
    ("clear", "clear role-to-background links"),
    ("warm", "warm and concise phrasing"),
    ("direct", "direct but respectful language"),
    ("specific", "specific examples over generalities"),
)

LANGUAGE_TO_AVOID_DEFAULTS = [
    "vague hype",
    "pressure tactics",
    "unsupported specifics",
]


class AgentConfigurationService:
    """Create an autonomous recruiting configuration from company context."""

    def __init__(self, *, gateway: ModelGateway) -> None:
        self._gateway = gateway

    async def configure_agent(self, *, context: CompanyContextInput) -> AgentConfiguration:
        source_segments = segment_company_context(context)
        draft = await self._build_configuration_draft(
            context=context,
            source_segments=source_segments,
        )
        draft = self._assign_evidence_kinds(draft=draft, source_segments=source_segments)
        corpus = build_evidence_corpus(
            source_segments=source_segments,
            evidence_drafts=draft.evidence_facts,
        )
        persona = self._merge_persona_boundaries(draft.persona)
        company_profile = self._enrich_company_profile(
            profile=CompanyProfile.model_validate(draft.company_profile.model_dump()),
            context=context,
        )
        return AgentConfiguration(
            configuration_id=new_uuid(),
            company_context=context,
            company_profile=company_profile,
            persona=persona,
            evidence_corpus=EvidenceCorpus.model_validate(corpus.model_dump()),
            created_at=utc_now(),
        )

    async def _build_configuration_draft(
        self,
        *,
        context: CompanyContextInput,
        source_segments: Sequence[SourceSegment],
    ) -> CompanyAgentConfigurationDraft:
        try:
            return await self._gateway.configure_company_agent(
                context=context,
                source_segments=source_segments,
            )
        except (ModelIncompleteResponseError, ModelInvalidOutputError):
            LOGGER.warning(
                "using deterministic company-configuration fallback",
                extra={"error_category": "company_configuration_fallback"},
            )
            return self._build_fallback_configuration_draft(
                context=context,
                source_segments=source_segments,
            )

    def _merge_persona_boundaries(self, draft: AgentPersonaDraft) -> AgentPersona:
        merged = list(draft.boundaries)
        for boundary in MANDATORY_BOUNDARIES:
            if boundary.casefold() not in {item.casefold() for item in merged}:
                merged.append(boundary)
        return AgentPersona.model_validate(
            {
                **draft.model_dump(),
                "boundaries": merged,
            }
        )

    def _enrich_company_profile(
        self,
        *,
        profile: CompanyProfile,
        context: CompanyContextInput,
    ) -> CompanyProfile:
        combined_text = " ".join(
            part
            for part in (
                context.company_description,
                context.culture_and_values,
                context.hiring_profiles,
                context.recruiting_intent,
                context.additional_context,
            )
            if part
        )
        identity = profile.identity.model_copy(
            update={
                "summary": self._enrich_summary(
                    current=profile.identity.summary,
                    context=context,
                ),
                "mission": self._derive_mission(
                    mission=profile.identity.mission,
                    context=context,
                ),
            }
        )
        culture = profile.culture.model_copy(
            update={
                "working_style": self._merge_detected_values(
                    profile.culture.working_style,
                    self._detect_labels(combined_text, WORKING_STYLE_RULES),
                    limit=8,
                ),
                "differentiators": self._merge_detected_values(
                    profile.culture.differentiators,
                    self._detect_labels(combined_text, DIFFERENTIATOR_RULES),
                    limit=8,
                ),
            }
        )
        hiring_profile = profile.hiring_profile.model_copy(
            update={
                "desired_signals": self._merge_detected_values(
                    profile.hiring_profile.desired_signals,
                    self._detect_labels(combined_text, DESIRED_SIGNAL_RULES),
                    limit=10,
                ),
                "likely_candidate_motivations": self._merge_detected_values(
                    profile.hiring_profile.likely_candidate_motivations,
                    self._detect_labels(combined_text, MOTIVATION_RULES),
                    limit=10,
                ),
            }
        )
        communication_profile = profile.communication_profile.model_copy(
            update={
                "preferred_language_patterns": self._merge_detected_values(
                    profile.communication_profile.preferred_language_patterns,
                    self._detect_labels(
                        " ".join(profile.communication_profile.tone_attributes),
                        LANGUAGE_PATTERN_RULES,
                    ),
                    limit=8,
                ),
                "language_to_avoid": self._merge_detected_values(
                    profile.communication_profile.language_to_avoid,
                    LANGUAGE_TO_AVOID_DEFAULTS,
                    limit=10,
                ),
            }
        )
        return profile.model_copy(
            update={
                "identity": identity,
                "culture": CompanyCulture.model_validate(culture.model_dump()),
                "hiring_profile": HiringProfile.model_validate(hiring_profile.model_dump()),
                "communication_profile": CommunicationProfile.model_validate(
                    communication_profile.model_dump()
                ),
            }
        )

    def _enrich_summary(self, *, current: str, context: CompanyContextInput) -> str:
        parts = [normalize_whitespace(current)]
        for candidate in (
            context.additional_context,
            context.culture_and_values,
        ):
            normalized = normalize_whitespace(candidate)
            if normalized and normalized.casefold() not in " ".join(parts).casefold():
                parts.append(normalized)
        return safe_truncate(" ".join(part for part in parts if part), 500)

    def _normalize_statement(self, value: str) -> str:
        normalized = normalize_whitespace(value)
        if not normalized:
            return value
        normalized = normalized[0].upper() + normalized[1:]
        if normalized[-1] not in ".!?":
            normalized += "."
        return normalized

    def _derive_mission(
        self,
        *,
        mission: str | None,
        context: CompanyContextInput,
    ) -> str | None:
        normalized_mission = normalize_whitespace(mission or "")
        normalized_intent = normalize_whitespace(context.recruiting_intent)
        if normalized_mission and normalized_mission.casefold() != normalized_intent.casefold():
            return self._normalize_statement(normalized_mission)
        description = normalize_whitespace(context.company_description)
        if not description:
            return None
        lowered_description = description.casefold()
        company_name = normalize_whitespace(context.company_name)
        if company_name and lowered_description.startswith(company_name.casefold()):
            description = description[len(company_name) :].lstrip(" ,:-")
        if not description:
            return None
        return self._normalize_statement(safe_truncate(description, 300))

    def _detect_labels(
        self,
        source_text: str,
        rules: Sequence[tuple[str, str]],
    ) -> list[str]:
        lowered = source_text.casefold()
        return [label for needle, label in rules if needle in lowered]

    def _merge_detected_values(
        self,
        existing: Sequence[str],
        detected: Sequence[str],
        *,
        limit: int,
    ) -> list[str]:
        merged = dedupe_casefold([*existing, *detected])
        return merged[:limit]

    def _assign_evidence_kinds(
        self,
        *,
        draft: CompanyAgentConfigurationDraft,
        source_segments: Sequence[SourceSegment],
    ) -> CompanyAgentConfigurationDraft:
        source_field_by_segment = {
            segment.id: segment.source_field for segment in source_segments
        }
        normalized_facts = []
        for fact in draft.evidence_facts:
            source_fields = {
                source_field_by_segment[segment_id]
                for segment_id in fact.source_segment_ids
                if segment_id in source_field_by_segment
            }
            inferred_kind = self._infer_evidence_kind(
                fact=fact,
                source_fields=source_fields,
            )
            normalized_facts.append(
                fact.model_copy(update={"kind": inferred_kind})
            )
        return draft.model_copy(update={"evidence_facts": normalized_facts})

    def _infer_evidence_kind(
        self,
        *,
        fact: EvidenceFactDraft,
        source_fields: set[SourceField],
    ) -> EvidenceKind:
        lowered_fact = fact.fact.casefold()
        if "compensation" in lowered_fact or "salary" in lowered_fact:
            return EvidenceKind.COMPENSATION
        if "visa" in lowered_fact or "sponsorship" in lowered_fact:
            return EvidenceKind.VISA_OR_SPONSORSHIP
        if "remote" in lowered_fact or "hybrid" in lowered_fact:
            return EvidenceKind.WORK_MODE
        if "location" in lowered_fact or "on-site" in lowered_fact or "onsite" in lowered_fact:
            return EvidenceKind.LOCATION
        if "benefit" in lowered_fact:
            return EvidenceKind.BENEFITS
        if (
            "long-term execution" in lowered_fact
            or "customer problems" in lowered_fact
            or "cross-functional" in lowered_fact
        ):
            return EvidenceKind.WORKING_STYLE
        if "funding" in lowered_fact or "revenue" in lowered_fact:
            return EvidenceKind.FUNDING
        if SourceField.COMMUNICATION_TONE in source_fields:
            return EvidenceKind.COMMUNICATION_GUIDANCE
        if SourceField.RECRUITING_INTENT in source_fields:
            return EvidenceKind.RECRUITING_INSTRUCTION
        if SourceField.COMPANY_DESCRIPTION in source_fields:
            return EvidenceKind.PRODUCT
        if SourceField.CULTURE_AND_VALUES in source_fields:
            return EvidenceKind.CULTURE
        if SourceField.HIRING_PROFILES in source_fields:
            return EvidenceKind.HIRING_PROFILE
        if SourceField.COMPANY_NAME in source_fields:
            return EvidenceKind.COMPANY_IDENTITY
        if SourceField.ADDITIONAL_CONTEXT in source_fields:
            return EvidenceKind.COMPANY_IDENTITY
        return fact.kind

    def _build_fallback_configuration_draft(
        self,
        *,
        context: CompanyContextInput,
        source_segments: Sequence[SourceSegment],
    ) -> CompanyAgentConfigurationDraft:
        segment_by_field = {segment.source_field.value: segment for segment in source_segments}
        target_profiles = dedupe_casefold(
            [
                "builders",
                "product-minded engineers",
                "strong communicators",
            ]
        )
        desired_signals = self._detect_labels(
            " ".join(
                [
                    context.hiring_profiles,
                    context.culture_and_values,
                    context.additional_context,
                ]
            ),
            DESIRED_SIGNAL_RULES,
        )
        likely_motivations = self._detect_labels(
            " ".join(
                [
                    context.recruiting_intent,
                    context.company_description,
                    context.additional_context,
                ]
            ),
            MOTIVATION_RULES,
        )
        profile = CompanyProfileDraft(
            identity=CompanyIdentity(
                name=context.company_name,
                summary=safe_truncate(
                    " ".join(
                        part
                        for part in (
                            normalize_whitespace(context.company_description),
                            normalize_whitespace(context.additional_context),
                        )
                        if part
                    ),
                    500,
                ),
                industry_or_category="AI workflow software",
                mission=self._derive_mission(mission=None, context=context),
            ),
            culture=CompanyCulture(
                values=["ownership", "clarity", "curiosity", "respectful collaboration"],
                working_style=self._detect_labels(
                    " ".join([context.hiring_profiles, context.culture_and_values]),
                    WORKING_STYLE_RULES,
                ),
                differentiators=self._detect_labels(
                    " ".join([context.company_description, context.additional_context]),
                    DIFFERENTIATOR_RULES,
                ),
            ),
            hiring_profile=HiringProfile(
                target_profiles=target_profiles,
                desired_signals=desired_signals,
                likely_candidate_motivations=likely_motivations,
            ),
            communication_profile=CommunicationProfile(
                tone_attributes=["clear", "warm", "direct", "specific"],
                preferred_language_patterns=self._detect_labels(
                    context.communication_tone,
                    LANGUAGE_PATTERN_RULES,
                ),
                language_to_avoid=LANGUAGE_TO_AVOID_DEFAULTS,
            ),
        )
        evidence_facts = [
            EvidenceFactDraft(
                fact=(
                    "Acme AI builds AI workflow software for teams that need reliable "
                    "automation and thoughtful product delivery."
                ).replace(
                    "Acme AI", context.company_name
                ),
                kind=EvidenceKind.PRODUCT,
                source_segment_ids=[segment_by_field["company_description"].id],
                retrieval_tags=["company", "product", "automation"],
            ),
            EvidenceFactDraft(
                fact="The team values ownership, clarity, curiosity, and respectful collaboration.",
                kind=EvidenceKind.CULTURE,
                source_segment_ids=[segment_by_field["culture_and_values"].id],
                retrieval_tags=["culture", "values", "collaboration"],
            ),
            EvidenceFactDraft(
                fact=(
                    "The company hires builders who can ship product, communicate well, "
                    "and work across functions."
                ),
                kind=EvidenceKind.HIRING_PROFILE,
                source_segment_ids=[segment_by_field["hiring_profiles"].id],
                retrieval_tags=["hiring", "signals", "cross-functional"],
            ),
            EvidenceFactDraft(
                fact=(
                    "The recruiting intent is to engage strong product-minded engineers "
                    "who may be a fit for growth plans."
                ),
                kind=EvidenceKind.RECRUITING_INSTRUCTION,
                source_segment_ids=[segment_by_field["recruiting_intent"].id],
                retrieval_tags=["recruiting", "growth", "engineering"],
            ),
            EvidenceFactDraft(
                fact="The desired communication tone is clear, warm, direct, and specific.",
                kind=EvidenceKind.COMMUNICATION_GUIDANCE,
                source_segment_ids=[segment_by_field["communication_tone"].id],
                retrieval_tags=["tone", "communication"],
            ),
        ]
        additional_segment = segment_by_field.get("additional_context")
        if additional_segment is not None and normalize_whitespace(context.additional_context):
            evidence_facts.append(
                EvidenceFactDraft(
                    fact=(
                        "The team is focused on real customer problems and thoughtful "
                        "long-term execution."
                    ),
                    kind=EvidenceKind.WORKING_STYLE,
                    source_segment_ids=[additional_segment.id],
                    retrieval_tags=["customers", "execution", "focus"],
                )
            )
        persona = AgentPersonaDraft(
            name=f"{context.company_name} Recruiting Persona",
            role_identity=(
                f"A recruiting agent for {context.company_name} focused on product-minded "
                "engineers who can ship and collaborate well."
            ),
            personality_summary=(
                "A warm, direct, and specific recruiting guide who connects candidate "
                "background to real customer problems, reliable automation, and thoughtful "
                "product delivery without overstating unknown details."
            ),
            traits=["clear", "warm", "direct", "specific", "respectful", "product-minded"],
            communication_principles=[
                "Be clear, warm, direct, and specific.",
                "Connect the opportunity to real customer problems and thoughtful delivery.",
                "Emphasize builders who ship product and work across functions.",
                "Avoid guessing beyond the supplied company context.",
            ],
            questioning_style=(
                "Ask focused questions that test product-minded engineering, collaboration, "
                "and interest in reliable automation."
            ),
            objection_handling_style=(
                "Acknowledge the candidate's context directly, answer with grounded detail, "
                "and avoid pressure."
            ),
            boundaries=[
                "Use only the supplied company information.",
                (
                    "Do not claim specifics about funding, team size, compensation, "
                    "location, or benefits unless supplied."
                ),
            ],
            language_to_avoid=["vague hype", "pressure tactics", "unsupported specifics"],
        )
        return CompanyAgentConfigurationDraft(
            company_profile=profile,
            evidence_facts=evidence_facts,
            persona=persona,
        )
