"""Tests for graph routing helpers."""

from typing import cast

from psview_agent.agent.graph_state import ConversationGraphState
from psview_agent.agent.routing import (
    route_after_deterministic_checks,
    route_after_evaluation,
)
from psview_agent.domain.evaluation import DeterministicResponseCheck, ResponseEvaluation


def test_route_after_deterministic_checks() -> None:
    assert (
        route_after_deterministic_checks(
            cast(
                ConversationGraphState,
                {
                    "deterministic_check": DeterministicResponseCheck(
                        passed=True,
                        violations=[],
                        question_count=0,
                        character_count=10,
                    ),
                    "revision_count": 0,
                },
            ),
            max_revisions=1,
        )
        == "pass"
    )


def test_route_after_evaluation() -> None:
    route = route_after_evaluation(
        cast(
            ConversationGraphState,
            {
                "evaluation": ResponseEvaluation(
                    personality_consistency=0.4,
                    company_grounding=0.4,
                    candidate_relevance=0.4,
                    action_alignment=0.4,
                    conversational_naturalness=0.4,
                    repetition_risk=0.7,
                    unsupported_claims=[],
                    personality_violations=[],
                    policy_violations=[],
                    passed=False,
                    revision_instructions=["revise"],
                ),
                "revision_count": 0,
            },
        ),
        max_revisions=1,
    )
    assert route == "revise"
