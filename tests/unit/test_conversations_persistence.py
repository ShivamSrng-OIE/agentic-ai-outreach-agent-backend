import pytest
import json
from unittest.mock import AsyncMock, MagicMock
from fastapi import Request
from httpx import AsyncClient

from psview_agent.api.routes.conversations import (
    get_client_ip,
    is_private_ip,
    get_ip_location,
)

def test_is_private_ip() -> None:
    assert is_private_ip("127.0.0.1") is True
    assert is_private_ip("localhost") is True
    assert is_private_ip("::1") is True
    assert is_private_ip("10.0.0.1") is True
    assert is_private_ip("192.168.1.1") is True
    assert is_private_ip("172.16.0.1") is True
    assert is_private_ip("172.31.255.255") is True
    assert is_private_ip("172.32.0.1") is False
    assert is_private_ip("8.8.8.8") is False


def test_get_client_ip() -> None:
    req1 = MagicMock(spec=Request)
    req1.headers = {"x-forwarded-for": "1.2.3.4, 5.6.7.8"}
    assert get_client_ip(req1) == "1.2.3.4"

    req2 = MagicMock(spec=Request)
    req2.headers = {"x-real-ip": "5.6.7.8"}
    assert get_client_ip(req2) == "5.6.7.8"

    req3 = MagicMock(spec=Request)
    req3.headers = {}
    req3.client = MagicMock()
    req3.client.host = "9.9.9.9"
    assert get_client_ip(req3) == "9.9.9.9"


@pytest.mark.anyio
async def test_get_ip_location_none_api_key() -> None:
    res = await get_ip_location("1.2.3.4", None)
    assert res is None


@pytest.mark.anyio
async def test_get_ip_location_mocked(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "ip": "1.2.3.4",
        "city": "Dallas",
        "region": "Texas",
        "country_name": "United States",
        "latitude": 32.7767,
        "longitude": -96.7970,
    }

    class MockAsyncClient:
        def __init__(self, *args, **kwargs):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass
        async def get(self, url: str):
            return mock_response

    monkeypatch.setattr("httpx.AsyncClient", MockAsyncClient)

    loc = await get_ip_location("1.2.3.4", "some_key")
    assert loc is not None
    assert loc["city"] == "Dallas"
    assert loc["country"] == "United States"


@pytest.mark.anyio
async def test_api_routes_persistence_logging(client: AsyncClient, app) -> None:
    from tests.fixtures.domain import sample_configuration
    # Mock MongoDB state in app
    mock_db = MagicMock()
    mock_db.interactions = MagicMock()
    mock_db.interactions.insert_one = AsyncMock()
    app.state.mongodb = mock_db

    # Test /start persistence intercept
    config_obj = sample_configuration()
    start_payload = {
        "configuration": config_obj.model_dump(mode="json"),
        "candidate": {
            "name": "Alex",
            "current_role": "Developer",
            "background_summary": "Alex has built APIs and python services.",
        },
        "target_role": "Backend Engineer",
    }
    
    headers = {
        "X-User-ID": "usr-test-123",
        "X-User-Location": json.dumps({
            "ip": "8.8.8.8",
            "city": "Mountain View",
            "region": "California",
            "country": "United States",
        })
    }

    response = await client.post("/api/v1/conversations/start", json=start_payload, headers=headers)
    assert response.status_code == 200
    assert mock_db.interactions.insert_one.called
    
    # Check that it recorded the start action and user ID
    call_args = mock_db.interactions.insert_one.call_args[0][0]
    assert call_args["user_id"] == "usr-test-123"
    assert call_args["action"] == "start"
    assert call_args["location"]["city"] == "Mountain View"

    # Reset mock and test /turn persistence intercept
    mock_db.interactions.insert_one.reset_mock()
    session_data = response.json()["session"]
    
    turn_payload = {
        "session": session_data,
        "candidate_reply": "I am interested in hearing more about the tech stack.",
    }

    response = await client.post("/api/v1/conversations/turn", json=turn_payload, headers=headers)
    assert response.status_code == 200
    assert mock_db.interactions.insert_one.called

    call_args = mock_db.interactions.insert_one.call_args[0][0]
    assert call_args["user_id"] == "usr-test-123"
    assert call_args["action"] == "turn"
    assert call_args["candidate_reply"] == "I am interested in hearing more about the tech stack."
    assert call_args["candidate"]["name"] == "Alex"
    assert call_args["target_role"] == "Backend Engineer"


@pytest.mark.anyio
async def test_agent_configure_persistence_logging(client: AsyncClient, app) -> None:
    # Mock MongoDB state in app
    mock_db = MagicMock()
    mock_db.interactions = MagicMock()
    mock_db.interactions.insert_one = AsyncMock()
    app.state.mongodb = mock_db

    payload = {
        "company_context": {
            "company_name": "Test Acme Tech",
            "company_description": "We build AI workflow software for teams that need reliable automation and thoughtful product delivery.",
            "culture_and_values": "The team values ownership, clarity, curiosity, and respectful collaboration.",
            "hiring_profiles": "We hire builders who can ship product, communicate well, and work across functions.",
            "communication_tone": "Clear, warm, direct, and specific.",
            "recruiting_intent": "We want to engage strong product-minded engineers who may be a fit for our growth plans.",
            "additional_context": "The team is focused on real customer problems and thoughtful long-term execution."
        }
    }

    headers = {
        "X-User-ID": "usr-config-123",
        "X-User-Location": json.dumps({
            "ip": "8.8.8.8",
            "city": "Mountain View",
            "region": "California",
            "country": "United States",
        })
    }

    response = await client.post("/api/v1/agents/configure", json=payload, headers=headers)
    assert response.status_code == 200
    assert mock_db.interactions.insert_one.called

    call_args = mock_db.interactions.insert_one.call_args[0][0]
    assert call_args["user_id"] == "usr-config-123"
    assert call_args["action"] == "configure"
    assert call_args["company_context"]["company_name"] == "Test Acme Tech"


@pytest.mark.anyio
async def test_api_routes_error_persistence_logging(client: AsyncClient, app, monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.fixtures.domain import sample_configuration
    # Mock MongoDB state in app
    mock_db = MagicMock()
    mock_db.interactions = MagicMock()
    mock_db.interactions.insert_one = AsyncMock()
    app.state.mongodb = mock_db

    # First, get a valid session by starting a conversation
    config_obj = sample_configuration()
    start_payload = {
        "configuration": config_obj.model_dump(mode="json"),
        "candidate": {
            "name": "Alex",
            "current_role": "Developer",
            "background_summary": "Alex has built APIs and python services.",
        },
        "target_role": "Backend Engineer",
    }
    
    headers = {
        "X-User-ID": "usr-error-123",
        "X-User-Location": json.dumps({
            "ip": "8.8.8.8",
            "city": "Mountain View",
            "region": "California",
            "country": "United States",
        })
    }

    response = await client.post("/api/v1/conversations/start", json=start_payload, headers=headers)
    assert response.status_code == 200
    session_data = response.json()["session"]

    # Force ConversationTurnService to raise TurnLimitReachedError
    from psview_agent.core.errors import TurnLimitReachedError
    from psview_agent.services.conversation_turn import ConversationTurnService
    
    async def mock_process_turn(*args, **kwargs):
        raise TurnLimitReachedError("conversation turn limit reached")

    monkeypatch.setattr(ConversationTurnService, "process_turn", mock_process_turn)

    # Now make the turn call with the valid session which will trigger the error
    turn_payload = {
        "session": session_data,
        "candidate_reply": "Yes, I am interested.",
    }

    mock_db.interactions.insert_one.reset_mock()
    response = await client.post("/api/v1/conversations/turn", json=turn_payload, headers=headers)
    assert response.status_code == 409  # TurnLimitReachedError maps to 409 Conflict
    assert mock_db.interactions.insert_one.called

    call_args = mock_db.interactions.insert_one.call_args[0][0]
    assert call_args["user_id"] == "usr-error-123"
    assert call_args["action"] == "error"
    assert call_args["error_action"] == "turn"
    assert call_args["error_code"] == "turn_limit_reached"
    assert "limit" in call_args["message"].lower()


@pytest.mark.anyio
async def test_model_override_middleware(client: AsyncClient, app) -> None:
    from psview_agent.core.config import model_override_var
    
    # Add a temporary test endpoint to verify the contextvar
    @app.get("/api/test-context-override")
    async def test_context_override():
        override = model_override_var.get()
        if override:
            return {
                "provider": override.provider.value if override.provider else None,
                "model_name": override.model_name,
                "api_key": override.api_key,
                "resume_parsing_model_name": override.resume_parsing_model_name,
            }
        return {"override": None}
        
    # Send request with headers
    headers = {
        "X-Model-Provider": "gemini",
        "X-Model-Name": "gemini-2.5-flash",
        "X-Model-Api-Key": "my-gemini-key",
        "X-Model-Resume-Parsing-Name": "openai/gpt-4o-mini",
    }
    
    resp = await client.get("/api/test-context-override", headers=headers)
    assert resp.status_code == 200
    assert resp.json() == {
        "provider": "gemini",
        "model_name": "gemini-2.5-flash",
        "api_key": "my-gemini-key",
        "resume_parsing_model_name": "openai/gpt-4o-mini",
    }
    
    # Send request with invalid provider to trigger ValidationError handler
    invalid_headers = {
        "X-Model-Provider": "invalid-provider-name",
    }
    resp_invalid = await client.get("/api/test-context-override", headers=invalid_headers)
    assert resp_invalid.status_code == 200
    assert resp_invalid.json() == {"override": None}

    # Send request without headers
    resp_no_headers = await client.get("/api/test-context-override")
    assert resp_no_headers.status_code == 200
    assert resp_no_headers.json() == {"override": None}



@pytest.mark.anyio
async def test_model_override_middleware_and_test_connection(client: AsyncClient, monkeypatch: pytest.MonkeyPatch) -> None:
    # Mock AsyncOpenAI call inside test_connection
    mock_create = AsyncMock()
    mock_client_instance = MagicMock()
    mock_client_instance.chat = MagicMock()
    mock_client_instance.chat.completions = MagicMock()
    mock_client_instance.chat.completions.create = mock_create
    mock_client_instance.close = AsyncMock()
    
    mock_init = MagicMock(return_value=mock_client_instance)
    monkeypatch.setattr("psview_agent.api.routes.agents.AsyncOpenAI", mock_init)
    
    # Test connection request payload
    test_payload = {
        "provider": "openai",
        "model_name": "gpt-4o-test",
        "api_key": "my-custom-key"
    }
    
    response = await client.post("/api/v1/agents/test-connection", json=test_payload)
    assert response.status_code == 200
    assert response.json() == {"status": "success", "message": "Connection verified successfully!"}
    
    # Ensure it initialized with the correct base url and api key
    mock_init.assert_called_once_with(
        api_key="my-custom-key",
        base_url="https://api.openai.com/v1",
        timeout=10.0,
        max_retries=0,
        default_headers={},
    )


