"""Tests for the agent configuration service."""

import pytest
from tests.fakes.fake_model_gateway import FakeModelGateway
from tests.fixtures.domain import sample_company_context

from psview_agent.core.errors import ModelInvalidOutputError
from psview_agent.domain.agent import AgentPersonaDraft, CompanyAgentConfigurationDraft
from psview_agent.domain.company import (
    CommunicationProfile,
    CompanyCulture,
    CompanyIdentity,
    CompanyProfileDraft,
    EvidenceFactDraft,
    HiringProfile,
)
from psview_agent.domain.enums import EvidenceKind
from psview_agent.services.agent_configuration import (
    MANDATORY_BOUNDARIES,
    AgentConfigurationService,
)


@pytest.mark.asyncio
async def test_configuration_service_assigns_ids_and_boundaries() -> None:
    service = AgentConfigurationService(gateway=FakeModelGateway())
    configuration = await service.configure_agent(context=sample_company_context())
    assert configuration.configuration_id
    assert configuration.evidence_corpus.evidence_facts[0].id == "fact_001"
    assert any(
        boundary == "Do not invent company or role facts."
        for boundary in configuration.persona.boundaries
    )
    assert len(configuration.persona.boundaries) >= len(MANDATORY_BOUNDARIES)


@pytest.mark.asyncio
async def test_configuration_service_persona_changes_with_company_context() -> None:
    service = AgentConfigurationService(gateway=FakeModelGateway())
    financial = await service.configure_agent(
        context=sample_company_context("Formal Financial Group")
    )
    startup = await service.configure_agent(context=sample_company_context("Speedy AI Lab"))
    assert (
        financial.company_profile.communication_profile.tone_attributes
        != startup.company_profile.communication_profile.tone_attributes
    )


@pytest.mark.asyncio
async def test_configuration_service_enriches_sparse_company_profile_fields() -> None:
    context = sample_company_context()
    service = AgentConfigurationService(
        gateway=FakeModelGateway(
            scenarios={
                "configure_company_agent": lambda **kwargs: CompanyAgentConfigurationDraft(
                    company_profile=CompanyProfileDraft(
                        identity=CompanyIdentity(
                            name=context.company_name,
                            summary=context.company_description,
                            industry_or_category="AI workflow software",
                            mission="builds reliable automation for teams",
                        ),
                        culture=CompanyCulture(
                            values=["ownership", "clarity"],
                            working_style=[],
                            differentiators=[],
                        ),
                        hiring_profile=HiringProfile(
                            target_profiles=["builders"],
                            desired_signals=[],
                            likely_candidate_motivations=[],
                        ),
                        communication_profile=CommunicationProfile(
                            tone_attributes=["clear", "warm", "direct", "specific"],
                            preferred_language_patterns=[],
                            language_to_avoid=[],
                        ),
                    ),
                    evidence_facts=[
                        EvidenceFactDraft(
                            fact="Acme builds AI workflow software.",
                            source_segment_ids=[kwargs["source_segments"][1].id],
                            retrieval_tags=[],
                        ),
                        EvidenceFactDraft(
                            fact="The team values ownership and clarity.",
                            source_segment_ids=[kwargs["source_segments"][2].id],
                            retrieval_tags=[],
                        ),
                        EvidenceFactDraft(
                            fact="The company wants to engage product-minded engineers.",
                            source_segment_ids=[kwargs["source_segments"][5].id],
                            retrieval_tags=[],
                        ),
                    ],
                    persona=AgentPersonaDraft(
                        name="Ari",
                        role_identity="Acme AI recruiting assistant",
                        personality_summary="A concise and grounded recruiting guide.",
                        traits=["thoughtful", "specific", "respectful"],
                        communication_principles=["stay grounded", "be clear", "be respectful"],
                        questioning_style="Ask one focused question when needed.",
                        objection_handling_style="Acknowledge concerns without pressure.",
                        boundaries=[
                            "Do not invent company or role facts.",
                            "Ask no more than one question per message.",
                        ],
                        language_to_avoid=[],
                    ),
                )
            }
        )
    )

    configuration = await service.configure_agent(context=context)

    assert configuration.company_profile.identity.mission is not None
    assert configuration.company_profile.identity.mission.startswith("Builds")
    assert configuration.company_profile.culture.working_style
    assert configuration.company_profile.culture.differentiators
    assert configuration.company_profile.hiring_profile.desired_signals
    assert configuration.company_profile.hiring_profile.likely_candidate_motivations
    assert configuration.company_profile.communication_profile.preferred_language_patterns
    assert configuration.company_profile.communication_profile.language_to_avoid


@pytest.mark.asyncio
async def test_configuration_service_falls_back_when_model_output_is_invalid() -> None:
    service = AgentConfigurationService(
        gateway=FakeModelGateway(
            scenarios={"configure_company_agent": ModelInvalidOutputError()}
        )
    )

    configuration = await service.configure_agent(context=sample_company_context())

    assert configuration.configuration_id
    assert configuration.company_profile.identity.name == "Acme AI"
    assert configuration.company_profile.hiring_profile.desired_signals
    assert configuration.persona.boundaries
    assert len(configuration.evidence_corpus.evidence_facts) >= 5


@pytest.mark.asyncio
async def test_configuration_service_assigns_sensitive_and_working_style_kinds() -> None:
    context = sample_company_context()
    service = AgentConfigurationService(
        gateway=FakeModelGateway(
            scenarios={
                "configure_company_agent": lambda **kwargs: CompanyAgentConfigurationDraft(
                    company_profile=CompanyProfileDraft(
                        identity=CompanyIdentity(
                            name=context.company_name,
                            summary=context.company_description,
                            industry_or_category="AI workflow software",
                            mission=None,
                        ),
                        culture=CompanyCulture(
                            values=["ownership", "clarity"],
                            working_style=[],
                            differentiators=[],
                        ),
                        hiring_profile=HiringProfile(
                            target_profiles=["builders"],
                            desired_signals=[],
                            likely_candidate_motivations=[],
                        ),
                        communication_profile=CommunicationProfile(
                            tone_attributes=["clear", "warm"],
                            preferred_language_patterns=[],
                            language_to_avoid=[],
                        ),
                    ),
                    evidence_facts=[
                        EvidenceFactDraft(
                            fact="Compensation details are not included in the provided context.",
                            source_segment_ids=[kwargs["source_segments"][6].id],
                            retrieval_tags=[],
                        ),
                        EvidenceFactDraft(
                            fact=(
                                "Visa sponsorship details are not included in the "
                                "provided context."
                            ),
                            source_segment_ids=[kwargs["source_segments"][6].id],
                            retrieval_tags=[],
                        ),
                        EvidenceFactDraft(
                            fact="Remote work details are not included in the provided context.",
                            source_segment_ids=[kwargs["source_segments"][6].id],
                            retrieval_tags=[],
                        ),
                        EvidenceFactDraft(
                            fact=(
                                "The team is focused on real customer problems and "
                                "thoughtful long-term execution."
                            ),
                            source_segment_ids=[kwargs["source_segments"][6].id],
                            retrieval_tags=[],
                        ),
                    ],
                    persona=AgentPersonaDraft(
                        name="Ari",
                        role_identity="Acme AI recruiting assistant",
                        personality_summary="A concise and grounded recruiting guide.",
                        traits=["thoughtful", "specific", "respectful"],
                        communication_principles=["stay grounded", "be clear", "be respectful"],
                        questioning_style="Ask one focused question when needed.",
                        objection_handling_style="Acknowledge concerns without pressure.",
                        boundaries=[
                            "Do not invent company or role facts.",
                            "Ask no more than one question per message.",
                        ],
                        language_to_avoid=[],
                    ),
                )
            }
        )
    )

    configuration = await service.configure_agent(context=context)
    facts_by_text = {fact.fact: fact.kind for fact in configuration.evidence_corpus.evidence_facts}

    assert (
        facts_by_text["Compensation details are not included in the provided context."]
        is EvidenceKind.COMPENSATION
    )
    assert (
        facts_by_text["Visa sponsorship details are not included in the provided context."]
        is EvidenceKind.VISA_OR_SPONSORSHIP
    )
    assert (
        facts_by_text["Remote work details are not included in the provided context."]
        is EvidenceKind.WORK_MODE
    )
    assert (
        facts_by_text[
            "The team is focused on real customer problems and thoughtful long-term execution."
        ]
        is EvidenceKind.WORKING_STYLE
    )
