"""Conversation routes."""

import os
import json
import logging
import datetime
from typing import Annotated

import httpx
from fastapi import APIRouter, Body, Depends, Header, Request

from psview_agent.api.dependencies import (
    get_agent_configuration_service,
    get_conversation_start_service,
    get_conversation_turn_service,
)
from psview_agent.domain.api import (
    ConversationTurnRequest,
    ConversationTurnResponse,
    StartConversationRequest,
    StartConversationResponse,
)
from psview_agent.services.agent_configuration import AgentConfigurationService
from psview_agent.services.conversation_start import ConversationStartService
from psview_agent.services.conversation_turn import ConversationTurnService

LOGGER = logging.getLogger("psview_agent.api.routes.conversations")

def get_client_ip(request: Request) -> str:
    x_forwarded_for = request.headers.get("x-forwarded-for")
    if x_forwarded_for:
        return x_forwarded_for.split(",")[0].strip()
    x_real_ip = request.headers.get("x-real-ip")
    if x_real_ip:
        return x_real_ip.strip()
    return request.client.host if request.client else "127.0.0.1"


def is_private_ip(ip: str) -> bool:
    if ip in ("127.0.0.1", "localhost", "::1", "0.0.0.0"):
        return True
    if ip.startswith("10.") or ip.startswith("192.168."):
        return True
    if ip.startswith("172."):
        try:
            parts = ip.split(".")
            if len(parts) >= 2:
                second_octet = int(parts[1])
                return 16 <= second_octet <= 31
        except ValueError:
            pass
    return False


async def get_ip_location(ip: str, api_key: str | None) -> dict[str, object] | None:
    if not api_key:
        return None
    url = f"https://ipapi.co/{ip}/json/"
    if is_private_ip(ip):
        url = "https://ipapi.co/json/"
    url += f"?key={api_key}"
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                data = resp.json()
                if "error" not in data:
                    return {
                        "ip": data.get("ip"),
                        "city": data.get("city"),
                        "region": data.get("region"),
                        "country": data.get("country_name"),
                        "lat": data.get("latitude"),
                        "lon": data.get("longitude")
                    }
    except Exception as err:
        LOGGER.warning(f"Failed to geolocate IP {ip}: {err}")
    return None


ERROR_FRIENDLY_MESSAGES = {
    "config_file_not_found": "Configuration file not found. Please contact support.",
    "config_yaml_parse_error": "Failed to parse configuration settings. Please contact support.",
    "unresolved_environment_placeholder": "Configuration environment placeholder unresolved. Please check backend environment variables.",
    "invalid_configuration": "Agent configuration is invalid. Please verify company context and guidelines.",
    "request_too_large": "The request is too large. Please limit the size of your input.",
    "conversation_closed": "This conversation has already been closed. No further messages can be processed.",
    "turn_limit_reached": "The maximum turn limit has been reached for this conversation.",
    "invalid_conversation_state": "The conversation has an invalid state. Please refresh the page and try again.",
    "invalid_company_evidence": "Validation failed for company evidence references. Please try again.",
    "retrieval_error": "Failed to retrieve relevant company evidence. Please check the search index.",
    "model_authentication_failed": "Authentication with the AI model provider failed. Please check backend API keys.",
    "model_rate_limited": "The AI model rate limit has been exceeded. Please wait a moment and try again.",
    "model_timeout": "The request to the AI model timed out. Please try again.",
    "model_connection_failed": "Could not connect to the AI model provider. Please check backend network settings.",
    "model_unavailable": "The AI model provider is currently unavailable. Please try again later.",
    "model_refusal": "The AI model refused to process the request due to content policy.",
    "model_incomplete_response": "The AI model returned an incomplete response. Please try again.",
    "model_invalid_output": "The AI model output could not be validated. Please try again.",
    "model_unsupported_feature": "Requested feature is not supported by the AI model.",
    "validation_error": "Request validation failed. Please check the input fields.",
    "internal_error": "An unexpected internal server error occurred. Please try again."
}


async def log_error_interaction(
    *,
    fastapi_request: Request,
    x_user_id: str | None,
    x_user_location: str | None,
    action: str,
    session_id: str | None,
    candidate: dict[str, object] | None,
    target_role: str | None,
    exc: Exception,
) -> None:
    db = fastapi_request.app.state.mongodb
    if db is None:
        return

    from psview_agent.core.errors import AppError
    from fastapi.exceptions import RequestValidationError
    from pydantic import ValidationError

    if isinstance(exc, AppError):
        error_code = exc.code
        raw_message = exc.message
    elif isinstance(exc, (RequestValidationError, ValidationError)):
        error_code = "validation_error"
        raw_message = "request validation failed"
    else:
        error_code = "internal_error"
        raw_message = "internal server error"

    friendly_msg = ERROR_FRIENDLY_MESSAGES.get(error_code, raw_message)

    try:
        location = None
        if x_user_location:
            try:
                location = json.loads(x_user_location)
            except Exception:
                location = {"raw": x_user_location}

        if not location or not location.get("city"):
            ip = get_client_ip(fastapi_request)
            api_key = os.getenv("IPAPI_KEY")
            resolved_location = await get_ip_location(ip, api_key)
            if resolved_location:
                location = resolved_location

        await db.interactions.insert_one({
            "user_id": x_user_id,
            "location": location,
            "timestamp": datetime.datetime.now(datetime.timezone.utc),
            "action": "error",
            "session_id": session_id,
            "candidate": candidate,
            "target_role": target_role,
            "error_action": action,
            "error_code": error_code,
            "message": friendly_msg,
            "error_type": exc.__class__.__name__,
            "details": [str(d) for d in getattr(exc, "details", [])]
        })
    except Exception as err:
        LOGGER.warning(f"Failed to log error interaction: {err}")


router = APIRouter(prefix="/api/v1/conversations", tags=["conversations"])

START_CONVERSATION_EXAMPLES = {
    "start_from_company_context": {
        "summary": "Easy start request",
        "description": (
            "Let the backend configure the agent and start the conversation in one call."
        ),
        "value": {
            "company_context": {
                "company_name": "Acme AI",
                "company_description": (
                    "Acme builds AI workflow software for teams that need reliable "
                    "automation and thoughtful product delivery."
                ),
                "culture_and_values": (
                    "The team values ownership, clarity, curiosity, and respectful collaboration."
                ),
                "hiring_profiles": (
                    "We hire builders who can ship product, communicate well, and work "
                    "across functions."
                ),
                "communication_tone": "Clear, warm, direct, and specific.",
                "recruiting_intent": (
                    "We want to engage strong product-minded engineers who may be a fit "
                    "for our growth plans."
                ),
                "additional_context": (
                    "The team is focused on real customer problems and thoughtful "
                    "long-term execution."
                ),
            },
            "candidate": {
                "name": "Casey",
                "current_role": "Senior Software Engineer",
                "background_summary": (
                    "Casey has built backend systems, product integrations, and "
                    "internal AI tooling."
                ),
            },
            "target_role": "Senior Engineer",
            "target_role_description": (
                "Own backend architecture, build product integrations, collaborate directly "
                "with product and design, and help shape the engineering foundation for a "
                "growing AI software company."
            ),
        },
    }
}


@router.post(
    "/start",
    response_model=StartConversationResponse,
    summary="Start a conversation",
    description=(
        "Provide either a full configuration object or raw company_context. "
        "Using company_context is the easiest option in the API docs because the "
        "backend will configure the recruiting agent for you before starting the "
        "conversation."
    ),
)
async def start_conversation(
    request: Annotated[
        StartConversationRequest,
        Body(openapi_examples=START_CONVERSATION_EXAMPLES),
    ],
    service: Annotated[
        ConversationStartService,
        Depends(get_conversation_start_service),
    ],
    configuration_service: Annotated[
        AgentConfigurationService,
        Depends(get_agent_configuration_service),
    ],
    fastapi_request: Request,
    x_user_id: Annotated[str | None, Header(alias="X-User-ID")] = None,
    x_user_location: Annotated[str | None, Header(alias="X-User-Location")] = None,
) -> StartConversationResponse:
    try:
        configuration = request.configuration
        if configuration is None:
            assert request.company_context is not None
            configuration = await configuration_service.configure_agent(context=request.company_context)
        session, trace = await service.start_conversation(
            configuration=configuration,
            candidate=request.candidate,
            target_role=request.target_role,
            target_role_description=request.target_role_description,
        )
    except Exception as exc:
        await log_error_interaction(
            fastapi_request=fastapi_request,
            x_user_id=x_user_id,
            x_user_location=x_user_location,
            action="start",
            session_id=None,
            candidate=request.candidate.model_dump(mode="json") if request.candidate else None,
            target_role=request.target_role,
            exc=exc,
        )
        raise exc

    db = fastapi_request.app.state.mongodb
    if db is not None:
        try:
            location = None
            if x_user_location:
                try:
                    location = json.loads(x_user_location)
                except Exception:
                    location = {"raw": x_user_location}

            if not location or not location.get("city"):
                ip = get_client_ip(fastapi_request)
                api_key = os.getenv("IPAPI_KEY")
                resolved_location = await get_ip_location(ip, api_key)
                if resolved_location:
                    location = resolved_location

            await db.interactions.insert_one({
                "user_id": x_user_id,
                "location": location,
                "timestamp": datetime.datetime.now(datetime.timezone.utc),
                "action": "start",
                "session_id": str(session.conversation_id),
                "target_role": request.target_role,
                "target_role_description": request.target_role_description,
                "candidate": request.candidate.model_dump(mode="json"),
                "initial_response": session.messages[-1].content if session.messages else None,
                "initial_decision_trace": trace.model_dump(mode="json") if trace else None
            })
        except Exception as err:
            LOGGER.warning(f"Failed to log start interaction: {err}")

    return StartConversationResponse(session=session, initial_decision_trace=trace)


@router.post("/turn", response_model=ConversationTurnResponse)
async def conversation_turn(
    request: ConversationTurnRequest,
    service: Annotated[
        ConversationTurnService,
        Depends(get_conversation_turn_service),
    ],
    fastapi_request: Request,
    x_user_id: Annotated[str | None, Header(alias="X-User-ID")] = None,
    x_user_location: Annotated[str | None, Header(alias="X-User-Location")] = None,
) -> ConversationTurnResponse:
    try:
        response = await service.process_turn(
            session=request.session,
            candidate_reply=request.candidate_reply,
        )
    except Exception as exc:
        await log_error_interaction(
            fastapi_request=fastapi_request,
            x_user_id=x_user_id,
            x_user_location=x_user_location,
            action="turn",
            session_id=str(request.session.conversation_id) if request.session else None,
            candidate=request.session.candidate.model_dump(mode="json") if request.session and request.session.candidate else None,
            target_role=request.session.target_role if request.session else None,
            exc=exc,
        )
        raise exc

    db = fastapi_request.app.state.mongodb
    if db is not None:
        try:
            location = None
            if x_user_location:
                try:
                    location = json.loads(x_user_location)
                except Exception:
                    location = {"raw": x_user_location}

            if not location or not location.get("city"):
                ip = get_client_ip(fastapi_request)
                api_key = os.getenv("IPAPI_KEY")
                resolved_location = await get_ip_location(ip, api_key)
                if resolved_location:
                    location = resolved_location

            await db.interactions.insert_one({
                "user_id": x_user_id,
                "location": location,
                "timestamp": datetime.datetime.now(datetime.timezone.utc),
                "action": "turn",
                "session_id": str(request.session.conversation_id),
                "candidate": request.session.candidate.model_dump(mode="json"),
                "target_role": request.session.target_role,
                "candidate_reply": request.candidate_reply,
                "agent_response": response.agent_message.content if response.agent_message else None,
                "decision_trace": response.decision_trace.model_dump(mode="json") if response.decision_trace else None
            })
        except Exception as err:
            LOGGER.warning(f"Failed to log turn interaction: {err}")

    return response
