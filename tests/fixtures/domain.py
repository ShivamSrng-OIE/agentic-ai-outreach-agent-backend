"""Domain object builders for tests."""

from psview_agent.domain.agent import AgentConfiguration, AgentPersona
from psview_agent.domain.candidate import CandidateProfile
from psview_agent.domain.company import (
    CommunicationProfile,
    CompanyContextInput,
    CompanyCulture,
    CompanyIdentity,
    CompanyProfile,
    EvidenceFactDraft,
    HiringProfile,
)
from psview_agent.domain.enums import EvidenceKind
from psview_agent.retrieval.corpus_builder import build_evidence_corpus, segment_company_context
from psview_agent.utils.identifiers import new_uuid
from psview_agent.utils.time import utc_now


def sample_company_context(company_name: str = "Acme AI") -> CompanyContextInput:
    return CompanyContextInput(
        company_name=company_name,
        company_description=(
            "Acme builds AI workflow software for teams that need reliable automation "
            "and thoughtful product delivery."
        ),
        culture_and_values=(
            "The team values ownership, clarity, curiosity, and respectful collaboration."
        ),
        hiring_profiles=(
            "We hire builders who can ship product, communicate well, and work across functions."
        ),
        communication_tone="Clear, warm, direct, and specific.",
        recruiting_intent=(
            "We want to engage strong product-minded engineers who may be a fit "
            "for our growth plans."
        ),
        additional_context=(
            "The team is focused on real customer problems and thoughtful long-term execution."
        ),
    )


def sample_configuration(company_name: str = "Acme AI") -> AgentConfiguration:
    context = sample_company_context(company_name)
    segments = segment_company_context(context)
    corpus = build_evidence_corpus(
        source_segments=segments,
        evidence_drafts=[
            EvidenceFactDraft(
                fact=f"{context.company_name} is hiring product-minded engineers.",
                kind=EvidenceKind.HIRING_PROFILE,
                source_segment_ids=[segments[0].id],
                retrieval_tags=["hiring", "engineering"],
            ),
            EvidenceFactDraft(
                fact="The company values ownership and clear communication.",
                kind=EvidenceKind.CULTURE,
                source_segment_ids=[segments[2].id, segments[4].id],
                retrieval_tags=["culture", "communication"],
            ),
            EvidenceFactDraft(
                fact="The recruiting intent is to engage strong builders for growth.",
                kind=EvidenceKind.RECRUITING_INSTRUCTION,
                source_segment_ids=[segments[5].id],
                retrieval_tags=["growth", "intent"],
            ),
        ],
    )
    return AgentConfiguration(
        configuration_id=new_uuid(),
        company_context=context,
        company_profile=CompanyProfile(
            identity=CompanyIdentity(
                name=context.company_name,
                summary=context.company_description[:120],
                industry_or_category="ai software",
                mission=context.company_description[:120],
            ),
            culture=CompanyCulture(
                values=["ownership", "clarity"],
                working_style=["cross-functional"],
                differentiators=["customer focus"],
            ),
            hiring_profile=HiringProfile(
                target_profiles=["builders", "engineers"],
                desired_signals=["communication", "shipping"],
                likely_candidate_motivations=["impact", "growth"],
            ),
            communication_profile=CommunicationProfile(
                tone_attributes=["clear", "warm"],
                preferred_language_patterns=["specific", "direct"],
                language_to_avoid=["spammy"],
            ),
        ),
        persona=AgentPersona(
            name="Ari",
            role_identity=f"{context.company_name} recruiting assistant",
            personality_summary="A concise and grounded recruiting guide.",
            traits=["thoughtful", "specific", "respectful"],
            communication_principles=["stay grounded", "be clear", "be respectful"],
            questioning_style="Ask one focused question when needed.",
            objection_handling_style="Acknowledge concerns without pressure.",
            boundaries=[
                "Do not invent company or role facts.",
                "Ask no more than one question per message.",
            ],
            language_to_avoid=["pushy"],
        ),
        evidence_corpus=corpus,
        created_at=utc_now(),
    )


def sample_candidate() -> CandidateProfile:
    return CandidateProfile(
        name="Casey",
        current_role="Senior Software Engineer",
        background_summary=(
            "Casey has built backend systems, product integrations, and internal AI tooling."
        ),
    )
