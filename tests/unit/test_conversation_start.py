"""Tests for conversation start."""

import pytest
from tests.fakes.fake_model_gateway import FakeModelGateway
from tests.fixtures.domain import sample_candidate, sample_company_context, sample_configuration

from psview_agent.core.config import RetrievalSettings
from psview_agent.core.errors import ModelInvalidOutputError
from psview_agent.domain.company import EvidenceCorpus, EvidenceFact
from psview_agent.domain.enums import EvidenceKind
from psview_agent.retrieval.lexical_retriever import LexicalEvidenceRetriever
from psview_agent.services.agent_configuration import AgentConfigurationService
from psview_agent.services.conversation_start import ConversationStartService


@pytest.mark.asyncio
async def test_conversation_start_returns_three_messages_and_initial_trace() -> None:
    service = ConversationStartService(
        gateway=FakeModelGateway(),
        retriever=LexicalEvidenceRetriever(
            RetrievalSettings(
                enabled=True,
                top_k=5,
                min_score=0,
                reuse_penalty=0.1,
                max_fact_candidates=20,
            )
        ),
    )
    session, trace = await service.start_conversation(
        configuration=sample_configuration(),
        candidate=sample_candidate(),
        target_role="Senior Backend Engineer",
    )
    assert len(session.outreach_plan.messages) == 3
    assert session.outreach_plan.messages[0].stage.value == "initial_outreach"
    assert session.messages[0].role.value == "agent"
    assert trace.retrieved_company_facts
    assert session.outreach_plan.messages[0].supported_claims
    assert "I noticed your experience building backend systems" in session.messages[0].content
    assert "your experience with Casey has" not in session.messages[0].content
    assert "backend systems" in trace.rationale_summary
    assert "Confidence is" in trace.rationale_summary
    assert trace.confidence != 0.85
    assert "fact_" not in trace.rationale_summary


@pytest.mark.asyncio
async def test_conversation_start_falls_back_when_outreach_plan_output_is_invalid() -> None:
    service = ConversationStartService(
        gateway=FakeModelGateway(
            scenarios={"generate_outreach_plan": ModelInvalidOutputError()}
        ),
        retriever=LexicalEvidenceRetriever(
            RetrievalSettings(
                enabled=True,
                top_k=5,
                min_score=0,
                reuse_penalty=0.1,
                max_fact_candidates=20,
            )
        ),
    )
    session, trace = await service.start_conversation(
        configuration=sample_configuration(),
        candidate=sample_candidate(),
        target_role="Senior Backend Engineer",
    )

    assert len(session.outreach_plan.messages) == 3
    assert session.outreach_plan.messages[0].company_fact_ids_used
    assert session.messages[0].content
    assert trace.company_facts_used
    assert session.outreach_plan.messages[0].objective == (
        "Open a relevant conversation with a specific reason for reaching out."
    )
    assert session.outreach_plan.messages[0].trigger == "candidate background appears relevant"
    assert session.outreach_plan.messages[0].supported_claims


@pytest.mark.asyncio
async def test_conversation_start_falls_back_when_unattributed_factual_outreach() -> None:
    service = ConversationStartService(
        gateway=FakeModelGateway(
            scenarios={
                "generate_outreach_plan": lambda **_: {
                    "overall_intent": "introduce the company",
                    "messages": [
                        {
                            "stage": "initial_outreach",
                            "objective": "mention the company and role",
                            "trigger": "initial contact",
                            "message": "Hi Casey, the company is hiring builders for this role.",
                            "company_fact_ids_used": ["fact_001"],
                        },
                        {
                            "stage": "follow_up",
                            "objective": "follow up",
                            "trigger": "no response",
                            "message": (
                                "Following up because the company is still hiring for "
                                "this role."
                            ),
                            "company_fact_ids_used": ["fact_001"],
                        },
                        {
                            "stage": "final_closeout",
                            "objective": "close out",
                            "trigger": "no response after follow-up",
                            "message": "I will close the loop for now.",
                            "company_fact_ids_used": [],
                        },
                    ],
                }
            }
        ),
        retriever=LexicalEvidenceRetriever(
            RetrievalSettings(
                enabled=True,
                top_k=5,
                min_score=0,
                reuse_penalty=0.1,
                max_fact_candidates=20,
            )
        ),
    )
    session, _ = await service.start_conversation(
        configuration=sample_configuration(),
        candidate=sample_candidate(),
        target_role="Senior Backend Engineer",
    )

    assert session.outreach_plan.messages[0].supported_claims


@pytest.mark.asyncio
async def test_conversation_start_repairs_semantically_wrong_claim_citations() -> None:
    configuration = await AgentConfigurationService(
        gateway=FakeModelGateway(
            scenarios={"configure_company_agent": ModelInvalidOutputError()}
        )
    ).configure_agent(context=sample_company_context())
    candidate = sample_candidate()
    service = ConversationStartService(
        gateway=FakeModelGateway(
            scenarios={
                "generate_outreach_plan": lambda **_: {
                    "overall_intent": "introduce the company",
                    "messages": [
                        {
                            "stage": "initial_outreach",
                            "objective": "open with product and hiring relevance",
                            "trigger": "candidate background appears relevant",
                            "message": (
                                "Hi Casey, Acme builds AI workflow software for teams that need "
                                "reliable automation and thoughtful product delivery. We hire "
                                "builders who can ship product, communicate well, and work "
                                "across functions."
                            ),
                            "supported_claims": [
                                {
                                    "claim": (
                                        "We hire builders who can ship product, communicate "
                                        "well, and work across functions."
                                    ),
                                    "evidence_fact_ids": ["fact_003"],
                                },
                                {
                                    "claim": (
                                        "Acme builds AI workflow software for teams that need "
                                        "reliable automation and thoughtful product delivery."
                                    ),
                                    "evidence_fact_ids": ["fact_003"],
                                },
                            ],
                        },
                        {
                            "stage": "follow_up",
                            "objective": "follow up",
                            "trigger": "no response",
                            "message": (
                                "Following up in case the role is still relevant. "
                                "We hire builders who can ship product, communicate well, "
                                "and work across functions."
                            ),
                            "supported_claims": [
                                {
                                    "claim": (
                                        "We hire builders who can ship product, communicate "
                                        "well, and work across functions."
                                    ),
                                    "evidence_fact_ids": ["fact_003"],
                                }
                            ],
                        },
                        {
                            "stage": "final_closeout",
                            "objective": "close out",
                            "trigger": "no response after follow-up",
                            "message": "I will close the loop for now.",
                            "supported_claims": [],
                        },
                    ],
                }
            }
        ),
        retriever=LexicalEvidenceRetriever(
            RetrievalSettings(
                enabled=True,
                top_k=5,
                min_score=0,
                reuse_penalty=0.1,
                max_fact_candidates=20,
            )
        ),
    )

    session, trace = await service.start_conversation(
        configuration=configuration,
        candidate=candidate,
        target_role="Senior Engineer",
    )

    first_message = session.outreach_plan.messages[0]
    claim_map = {
        claim.claim: claim.evidence_fact_ids
        for claim in first_message.supported_claims
    }
    assert claim_map[
        "Acme builds AI workflow software for teams that need reliable automation "
        "and thoughtful product delivery."
    ] == ["fact_001"]
    assert claim_map[
        "We hire builders who can ship product, communicate well, and work across functions."
    ] == ["fact_003"]
    assert first_message.company_fact_ids_used == ["fact_001", "fact_003"]
    assert [fact.id for fact in trace.company_facts_used] == ["fact_001", "fact_003"]


def test_candidate_experience_summary_normalizes_sentence_and_fragment() -> None:
    service = ConversationStartService(
        gateway=FakeModelGateway(),
        retriever=LexicalEvidenceRetriever(
            RetrievalSettings(
                enabled=True,
                top_k=5,
                min_score=0,
                reuse_penalty=0.1,
                max_fact_candidates=20,
            )
        ),
    )

    sentence_summary = service._candidate_experience_summary(sample_candidate())
    fragment_summary = service._candidate_experience_summary(
        sample_candidate().model_copy(
            update={"background_summary": "Built backend systems and AI tooling."}
        )
    )
    first_person_summary = service._candidate_experience_summary(
        sample_candidate().model_copy(
            update={
                "background_summary": (
                    "I build machine learning systems that work in production, "
                    "not just in notebooks. Machine Learning Engineer."
                )
            }
        )
    )

    assert sentence_summary == (
        "building backend systems, product integrations, and internal AI tooling"
    )
    assert fragment_summary == "building backend systems and AI tooling"
    assert first_person_summary == (
        "building machine learning systems that work in production, not just in notebooks"
    )


def test_conversation_start_rejects_broken_candidate_phrasing() -> None:
    service = ConversationStartService(
        gateway=FakeModelGateway(),
        retriever=LexicalEvidenceRetriever(
            RetrievalSettings(
                enabled=True,
                top_k=5,
                min_score=0,
                reuse_penalty=0.1,
                max_fact_candidates=20,
            )
        ),
    )

    assert service._has_broken_candidate_phrasing(
        "Hi Casey, your experience with Casey has built backend systems. stood out for the role."
    )
    assert not service._has_broken_candidate_phrasing(
        "Hi Casey, your experience building backend systems stood out for the role."
    )


@pytest.mark.asyncio
async def test_conversation_start_falls_back_from_raw_jd_and_duplicate_cta() -> None:
    raw_role_description = (
        "A Founding Engineer at a seed-stage startup is the first technical hire. "
        "You are tasked with translating product vision into code, establishing "
        "engineering foundations, and shipping customer-facing systems."
    )
    service = ConversationStartService(
        gateway=FakeModelGateway(
            scenarios={
                "generate_outreach_plan": lambda **_: {
                    "overall_intent": "introduce the role",
                    "messages": [
                        {
                            "stage": "initial_outreach",
                            "objective": "introduce role",
                            "trigger": "candidate relevance",
                            "message": (
                                "Hi Casey, your experience with Casey has built backend "
                                "systems stood out. The role itself centers on "
                                f"{raw_role_description}. "
                                "Would you be open to a brief conversation? Would you be open "
                                "to a brief conversation?"
                            ),
                            "supported_claims": [
                                {
                                    "claim": "Acme AI is hiring product-minded engineers.",
                                    "evidence_fact_ids": ["fact_001"],
                                }
                            ],
                        },
                        {
                            "stage": "follow_up",
                            "objective": "follow up",
                            "trigger": "no response",
                            "message": "Following up because Acme AI is hiring engineers.",
                            "supported_claims": [
                                {
                                    "claim": "Acme AI is hiring product-minded engineers.",
                                    "evidence_fact_ids": ["fact_001"],
                                }
                            ],
                        },
                        {
                            "stage": "final_closeout",
                            "objective": "close out",
                            "trigger": "no response",
                            "message": "I will close the loop for now.",
                            "supported_claims": [],
                        },
                    ],
                }
            }
        ),
        retriever=LexicalEvidenceRetriever(
            RetrievalSettings(
                enabled=True,
                top_k=5,
                min_score=0,
                reuse_penalty=0.1,
                max_fact_candidates=20,
            )
        ),
    )

    session, _ = await service.start_conversation(
        configuration=sample_configuration(),
        candidate=sample_candidate(),
        target_role="Founding Engineer",
        target_role_description=raw_role_description,
    )

    first_message = session.outreach_plan.messages[0].message
    assert first_message.count("?") == 1
    assert "role itself centers on" not in first_message
    assert "translating product vision into code" not in first_message
    assert "your experience with Casey has" not in first_message
    assert "I noticed your experience building backend systems" in first_message
    assert "that background seems especially relevant to" in first_message
    assert len(first_message.split(". ")) <= 4


@pytest.mark.asyncio
async def test_fallback_avoids_security_facts_when_role_is_not_security_related() -> None:
    configuration = sample_configuration()
    segments = configuration.evidence_corpus.source_segments
    configuration = configuration.model_copy(
        update={
            "evidence_corpus": EvidenceCorpus(
                source_segments=segments,
                evidence_facts=[
                    EvidenceFact(
                        id="fact_001",
                        fact=(
                            "Security and product snippets mention GDPR alignment, "
                            "data storage in France on AWS Paris, and AES-256 encryption."
                        ),
                        kind=EvidenceKind.PRODUCT,
                        source_segment_ids=[segments[0].id],
                        retrieval_tags=["security", "gdpr", "encryption"],
                    ),
                    EvidenceFact(
                        id="fact_002",
                        fact=(
                            "Strong candidate signals likely include automation thinking, "
                            "clear communication, analytical judgment, and improving talent search."
                        ),
                        kind=EvidenceKind.HIRING_PROFILE,
                        source_segment_ids=[segments[3].id],
                        retrieval_tags=["hiring", "automation", "communication"],
                    ),
                    EvidenceFact(
                        id="fact_003",
                        fact="The company values ownership and clear communication.",
                        kind=EvidenceKind.CULTURE,
                        source_segment_ids=[segments[2].id],
                        retrieval_tags=["culture", "communication"],
                    ),
                ],
            )
        }
    )
    service = ConversationStartService(
        gateway=FakeModelGateway(
            scenarios={"generate_outreach_plan": ModelInvalidOutputError()}
        ),
        retriever=LexicalEvidenceRetriever(
            RetrievalSettings(
                enabled=True,
                top_k=5,
                min_score=0,
                reuse_penalty=0.1,
                max_fact_candidates=20,
            )
        ),
    )

    session, _ = await service.start_conversation(
        configuration=configuration,
        candidate=sample_candidate(),
        target_role="Founding Engineer",
        target_role_description="Build product systems and early engineering foundations.",
    )

    first_message = session.outreach_plan.messages[0]
    assert "GDPR" not in first_message.message
    assert "AES-256" not in first_message.message
    assert first_message.company_fact_ids_used == ["fact_002"]
