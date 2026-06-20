"""Shared pytest fixtures."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient


def write_test_config(path: Path) -> None:
    path.write_text(
        (
            "app:\n"
            "  name: PSVIEW Recruiting Agent API\n"
            "  env: test\n"
            "  version: 0.1.0\n"
            "  api_v1_prefix: /api/v1\n"
            "  log_level: INFO\n"
            "model:\n"
            "  provider: openrouter\n"
            "  api_key: ${MODEL_API_KEY}\n"
            "  base_url: https://openrouter.ai/api/v1\n"
            "  model_name: ${MODEL_NAME}\n"
            "  structured_output_mode: auto\n"
            "  timeout_seconds: 5\n"
            "  max_retries: 1\n"
            "  max_output_tokens: 600\n"
            "  temperature: 0.2\n"
            "  repair_attempts: 1\n"
            "  concurrency_limit: 2\n"
            "  extra_body: {}\n"
            "openrouter:\n"
            "  site_url: null\n"
            "  app_name: PSVIEW Recruiting Agent\n"
            "runtime:\n"
            "  allowed_origins:\n"
            "    - http://localhost:5173\n"
            "  max_request_body_bytes: 100000\n"
            "  max_history_messages: 20\n"
            "  max_conversation_turns: 16\n"
            "  max_response_characters: 1000\n"
            "  max_revision_attempts: 1\n"
            "  langgraph_recursion_limit: 20\n"
            "retrieval:\n"
            "  enabled: true\n"
            "  top_k: 5\n"
            "  min_score: 0.05\n"
            "  reuse_penalty: 0.15\n"
            "  max_fact_candidates: 20\n"
            "database:\n"
            "  mongodb_uri: ${MONGODB_URI}\n"
        ),
        encoding="utf-8",
    )


@pytest.fixture
def config_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    config = tmp_path / "config.yaml"
    monkeypatch.setenv("MONGODB_URI", "mongodb://localhost:27017/hirewire_test")
    write_test_config(config)
    monkeypatch.setenv("CONFIG_FILE", str(config))
    monkeypatch.setenv("MODEL_API_KEY", "test-key")
    monkeypatch.setenv("MODEL_NAME", "test-model")
    monkeypatch.chdir(tmp_path)
    return config


@pytest_asyncio.fixture
async def app(config_path: Path) -> AsyncIterator[FastAPI]:
    from psview_agent.agent.graph import build_conversation_graph
    from psview_agent.main import create_app
    from psview_agent.services.agent_configuration import AgentConfigurationService
    from psview_agent.services.conversation_start import ConversationStartService
    from psview_agent.services.conversation_turn import ConversationTurnService
    from tests.fakes.fake_model_gateway import FakeModelGateway

    application = create_app()
    async with application.router.lifespan_context(application):
        fake_gateway = FakeModelGateway()
        settings = application.state.settings
        retriever = application.state.retriever
        graph = build_conversation_graph(
            settings=settings,
            gateway=fake_gateway,
            retriever=retriever,
        )
        application.state.model_gateway = fake_gateway
        application.state.graph = graph
        application.state.agent_configuration_service = AgentConfigurationService(
            gateway=fake_gateway
        )
        application.state.conversation_start_service = ConversationStartService(
            gateway=fake_gateway,
            retriever=retriever,
            retrieval_limit=settings.retrieval.top_k,
        )
        application.state.conversation_turn_service = ConversationTurnService(
            gateway=fake_gateway,
            retriever=retriever,
            graph=graph,
            settings=settings,
        )
        yield application


@pytest_asyncio.fixture
async def client(app: FastAPI) -> AsyncIterator[AsyncClient]:
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://testserver") as http_client:
        yield http_client
