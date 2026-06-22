"""Integration tests for the FastAPI app."""

from __future__ import annotations

import json

import pytest
from httpx import AsyncClient
from tests.fixtures.domain import sample_candidate, sample_company_context


@pytest.mark.asyncio
async def test_health_and_ready_endpoints(client: AsyncClient) -> None:
    health_response = await client.get("/health")
    ready_response = await client.get("/ready")

    assert health_response.status_code == 200
    assert health_response.json() == {
        "status": "ok",
        "service": "PSVIEW Recruiting Agent API",
        "version": "0.1.0",
        "environment": "test",
    }
    assert ready_response.status_code == 200
    assert ready_response.json() == {"status": "ready"}


@pytest.mark.asyncio
async def test_ready_endpoint_reports_starting_before_lifespan() -> None:
    from fastapi import FastAPI
    from httpx import ASGITransport

    from psview_agent.api.router import api_router

    app = FastAPI()
    app.state.ready = False
    app.state.settings = type(
        "SettingsProxy",
        (),
        {"app": type("AppSettingsProxy", (), {"name": "X", "version": "0", "env": "test"})()},
    )()
    app.include_router(api_router)

    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://testserver") as bare_client:
        response = await bare_client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "starting"}


@pytest.mark.asyncio
async def test_cors_preflight_returns_expected_headers(client: AsyncClient) -> None:
    response = await client.options(
        "/health",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert response.status_code == 204
    assert response.headers["Access-Control-Allow-Origin"] == "http://localhost:5173"
    assert response.headers["Access-Control-Allow-Methods"] == "*"
    assert response.headers["Access-Control-Allow-Headers"] == "*"
    assert response.headers["X-Request-ID"]


@pytest.mark.asyncio
async def test_configure_start_and_turn_round_trip(client: AsyncClient) -> None:
    configure_response = await client.post(
        "/api/v1/agents/configure",
        json={"company_context": sample_company_context().model_dump(mode="json")},
        headers={"X-Request-ID": "roundtrip-req-001"},
    )
    assert configure_response.status_code == 200
    assert configure_response.headers["X-Request-ID"] == "roundtrip-req-001"

    configuration = configure_response.json()["configuration"]
    start_response = await client.post(
        "/api/v1/conversations/start",
        json={
            "configuration": configuration,
            "candidate": sample_candidate().model_dump(mode="json"),
            "target_role": "Senior Engineer",
            "target_role_description": (
                "Own backend services, integrations, and platform foundations for an AI product."
            ),
        },
    )
    assert start_response.status_code == 200
    start_payload = start_response.json()
    assert len(start_payload["session"]["outreach_plan"]["messages"]) == 3
    assert start_payload["initial_decision_trace"]["selected_action"] == "introduce_opportunity"
    assert start_payload["session"]["target_role_description"]

    turn_response = await client.post(
        "/api/v1/conversations/turn",
        json={
            "session": start_payload["session"],
            "candidate_reply": "Can you share more about the role?",
        },
    )
    assert turn_response.status_code == 200
    turn_payload = turn_response.json()
    assert turn_payload["decision_trace"]["selected_action"] == "answer_candidate_question"
    assert turn_payload["evaluation"]["passed"] is True
    assert turn_payload["updated_state"]["turn_count"] == 1
    assert turn_payload["candidate_message"]["role"] == "candidate"
    assert turn_payload["agent_message"]["role"] == "agent"


@pytest.mark.asyncio
async def test_start_conversation_can_build_configuration_from_company_context(
    client: AsyncClient,
) -> None:
    start_response = await client.post(
        "/api/v1/conversations/start",
        json={
            "company_context": sample_company_context().model_dump(mode="json"),
            "candidate": sample_candidate().model_dump(mode="json"),
            "target_role": "Senior Engineer",
            "target_role_description": (
                "Own backend services, integrations, and platform foundations for an AI product."
            ),
        },
    )

    assert start_response.status_code == 200
    start_payload = start_response.json()
    assert start_payload["session"]["configuration"]["company_context"]["company_name"] == "Acme AI"
    assert len(start_payload["session"]["outreach_plan"]["messages"]) == 3
    assert start_payload["initial_decision_trace"]["selected_action"] == "introduce_opportunity"


@pytest.mark.asyncio
async def test_closed_conversation_returns_conflict(client: AsyncClient) -> None:
    configure_response = await client.post(
        "/api/v1/agents/configure",
        json={"company_context": sample_company_context().model_dump(mode="json")},
    )
    configuration = configure_response.json()["configuration"]
    start_response = await client.post(
        "/api/v1/conversations/start",
        json={
            "configuration": configuration,
            "candidate": sample_candidate().model_dump(mode="json"),
            "target_role": "Senior Engineer",
            "target_role_description": (
                "Own backend services, integrations, and platform foundations for an AI product."
            ),
        },
    )
    session = start_response.json()["session"]
    session["state"]["stage"] = "closed"
    session["state"]["is_closed"] = True
    session["state"]["close_reason"] = "candidate_opt_out"

    response = await client.post(
        "/api/v1/conversations/turn",
        json={"session": session, "candidate_reply": "Hello?"},
    )

    assert response.status_code == 409
    payload = response.json()
    assert payload["code"] == "conversation_closed"
    assert payload["request_id"]


@pytest.mark.asyncio
async def test_validation_errors_use_stable_error_schema(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/agents/configure",
        json={"company_context": {"company_name": ""}},
    )

    assert response.status_code == 422
    payload = response.json()
    assert payload["code"] == "validation_error"
    assert payload["message"] == "request validation failed"
    assert payload["request_id"]
    assert payload["details"]


@pytest.mark.asyncio
async def test_start_requires_exactly_one_configuration_source(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/conversations/start",
        json={
            "candidate": sample_candidate().model_dump(mode="json"),
            "target_role": "Senior Engineer",
        },
    )

    assert response.status_code == 422
    payload = response.json()
    assert payload["code"] == "validation_error"
    assert any(
        detail["message"] == "Value error, provide exactly one of configuration or company_context"
        for detail in payload["details"]
    )


@pytest.mark.asyncio
async def test_oversized_request_returns_413_with_request_id(client: AsyncClient) -> None:
    oversized_context = sample_company_context().model_dump(mode="json")
    oversized_context["additional_context"] = "A" * 120000
    body = json.dumps({"company_context": oversized_context})

    response = await client.post(
        "/api/v1/agents/configure",
        content=body,
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 413
    assert response.headers["X-Request-ID"]
    assert response.json()["code"] == "request_too_large"


@pytest.mark.asyncio
async def test_parse_resume_endpoint_txt(client: AsyncClient) -> None:
    files = {"file": ("resume.txt", b"Jane Doe\nAI Developer\nBuilding advanced LLM agents.", "text/plain")}
    response = await client.post(
        "/api/v1/agents/parse-resume",
        files=files,
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert "Jane Doe" in payload["text"]
    assert payload["extracted_profile"]["name"] == "Jane Doe"
    assert payload["extracted_profile"]["current_role"] == "AI Developer"


@pytest.mark.asyncio
async def test_parse_resume_endpoint_pdf_error(client: AsyncClient) -> None:
    files = {"file": ("resume.pdf", b"corrupt pdf data", "application/pdf")}
    response = await client.post(
        "/api/v1/agents/parse-resume",
        files=files,
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "error"
    assert "Failed to parse PDF resume" in payload["message"]
