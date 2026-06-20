"""Application lifespan wiring."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager

from fastapi import FastAPI

from psview_agent.core.config import clear_settings_cache
from psview_agent.core.config_loader import LoadedSettings, load_settings
from psview_agent.core.logging import configure_logging


def build_lifespan() -> Callable[[FastAPI], AbstractAsyncContextManager[None]]:
    """Build the application lifespan manager."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        clear_settings_cache()
        loaded: LoadedSettings = load_settings()
        settings = loaded.settings
        configure_logging(settings.app.log_level)

        from psview_agent.agent.graph import build_conversation_graph
        from psview_agent.integrations.models.client import create_model_client
        from psview_agent.integrations.models.gateway import OpenAICompatibleModelGateway
        from psview_agent.retrieval.lexical_retriever import LexicalEvidenceRetriever
        from psview_agent.services.agent_configuration import AgentConfigurationService
        from psview_agent.services.conversation_start import ConversationStartService
        from psview_agent.services.conversation_turn import ConversationTurnService

        import logging
        logger = logging.getLogger("psview_agent.core.lifespan")

        from motor.motor_asyncio import AsyncIOMotorClient
        from psview_agent.core.config import AppEnvironment

        mongodb_client = None
        mongodb = None
        if settings.app.env is not AppEnvironment.TEST:
            try:
                mongodb_client = AsyncIOMotorClient(settings.database.mongodb_uri)
                try:
                    mongodb = mongodb_client.get_default_database()
                except Exception:
                    mongodb = mongodb_client["hirewire"]
                logger.info("MongoDB client connected successfully")
            except Exception as err:
                logger.warning(f"Failed to connect to MongoDB: {err}")

        client = create_model_client(settings)
        gateway = OpenAICompatibleModelGateway(client=client, settings=settings)
        retriever = LexicalEvidenceRetriever(settings=settings.retrieval)
        graph = build_conversation_graph(settings=settings, gateway=gateway, retriever=retriever)
        agent_service = AgentConfigurationService(gateway=gateway)
        start_service = ConversationStartService(gateway=gateway, retriever=retriever)
        turn_service = ConversationTurnService(
            gateway=gateway,
            retriever=retriever,
            graph=graph,
            settings=settings,
        )

        app.state.settings = settings
        app.state.config_diagnostics = loaded.diagnostics
        app.state.model_client = client
        app.state.model_gateway = gateway
        app.state.retriever = retriever
        app.state.agent_configuration_service = agent_service
        app.state.conversation_start_service = start_service
        app.state.conversation_turn_service = turn_service
        app.state.graph = graph
        app.state.mongodb_client = mongodb_client
        app.state.mongodb = mongodb
        app.state.ready = True

        yield

        app.state.ready = False
        await client.close()
        if mongodb_client is not None:
            mongodb_client.close()

    return lifespan
