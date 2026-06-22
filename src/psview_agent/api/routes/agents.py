"""Agent configuration routes."""

import os
import json
import logging
import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Header, Request, Body, UploadFile, File
from openai import AsyncOpenAI

from psview_agent.api.dependencies import get_agent_configuration_service, get_model_gateway
from psview_agent.domain.api import ConfigureAgentRequest, ConfigureAgentResponse
from psview_agent.services.agent_configuration import AgentConfigurationService
from psview_agent.api.routes.conversations import get_client_ip, get_ip_location
from psview_agent.integrations.models.protocol import ModelGateway

LOGGER = logging.getLogger("psview_agent.api.routes.agents")

router = APIRouter(prefix="/api/v1/agents", tags=["agents"])


@router.post("/test-connection")
async def test_connection(
    request: Request,
    provider: str = Body(..., embed=True),
    model_name: str = Body(..., embed=True),
    api_key: str = Body(..., embed=True),
) -> dict[str, object]:
    """Test connection to an AI provider with the given api key and model."""
    provider_str = provider.lower()
    base_url = ""
    if provider_str == "openai":
        base_url = "https://api.openai.com/v1"
    elif provider_str == "gemini":
        base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
    elif provider_str == "openrouter":
        base_url = "https://openrouter.ai/api/v1"
    elif provider_str == "nvidia":
        base_url = "https://integrate.api.nvidia.com/v1"
    else:
        return {"status": "error", "message": f"Unsupported provider: {provider}"}

    headers = {}
    if provider_str == "openrouter":
        settings = request.app.state.settings
        if settings.openrouter.site_url is not None:
            headers["HTTP-Referer"] = str(settings.openrouter.site_url)
        if settings.openrouter.app_name:
            headers["X-OpenRouter-Title"] = settings.openrouter.app_name

    try:
        client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=10.0,
            max_retries=0,
            default_headers=headers,
        )
        # Make a tiny chat completions call to verify
        await client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "Ping"}],
            max_tokens=2,
        )
        await client.close()
        return {"status": "success", "message": "Connection verified successfully!"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.post("/configure", response_model=ConfigureAgentResponse)
async def configure_agent(
    request: ConfigureAgentRequest,
    service: Annotated[
        AgentConfigurationService,
        Depends(get_agent_configuration_service),
    ],
    fastapi_request: Request,
    x_user_id: Annotated[str | None, Header(alias="X-User-ID")] = None,
    x_user_location: Annotated[str | None, Header(alias="X-User-Location")] = None,
) -> ConfigureAgentResponse:
    configuration = await service.configure_agent(context=request.company_context)

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

            model_provider = None
            model_name = None
            try:
                from psview_agent.core.config import model_override_var
                override = model_override_var.get()
                if override:
                    model_provider = override.provider.value
                    model_name = override.model_name
                else:
                    settings = fastapi_request.app.state.settings
                    if settings:
                        model_provider = settings.model.provider.value
                        model_name = settings.model.model_name
            except Exception:
                pass

            await db.interactions.insert_one({
                "user_id": x_user_id,
                "location": location,
                "timestamp": datetime.datetime.now(datetime.timezone.utc),
                "action": "configure",
                "company_context": request.company_context.model_dump(mode="json"),
                "configuration": configuration.model_dump(mode="json"),
                "model_provider": model_provider,
                "model_name": model_name,
            })
        except Exception as err:
            LOGGER.warning(f"Failed to log configure interaction: {err}")

    return ConfigureAgentResponse(configuration=configuration)


@router.post("/parse-resume")
async def parse_resume(
    file: UploadFile = File(...),
    gateway: ModelGateway = Depends(get_model_gateway),
) -> dict[str, object]:
    """Parse a candidate resume (TXT/PDF) and extract details using the AI model."""
    import io
    from pypdf import PdfReader
    
    content = await file.read()
    filename = file.filename or ""
    
    text = ""
    if filename.lower().endswith(".pdf"):
        try:
            pdf_file = io.BytesIO(content)
            reader = PdfReader(pdf_file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        except Exception as exc:
            LOGGER.error(f"Failed to parse PDF resume: {exc}")
            return {"status": "error", "message": f"Failed to parse PDF resume: {str(exc)}"}
    else:
        try:
            text = content.decode("utf-8", errors="ignore")
        except Exception as exc:
            return {"status": "error", "message": f"Failed to decode text resume: {str(exc)}"}
            
    if not text.strip():
        return {"status": "error", "message": "Extracted resume text is empty"}
        
    try:
        profile = await gateway.extract_profile_from_resume(resume_text=text)
        return {
            "status": "success",
            "text": text,
            "extracted_profile": profile.model_dump(),
        }
    except Exception as exc:
        LOGGER.error(f"AI parsing of resume failed: {exc}")
        return {"status": "error", "message": f"AI model failed to parse resume details: {str(exc)}"}
