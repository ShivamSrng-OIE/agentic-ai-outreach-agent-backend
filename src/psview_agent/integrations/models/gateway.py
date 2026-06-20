"""OpenAI-compatible provider-neutral model gateway."""

import asyncio
import json
import logging
from collections.abc import Sequence
from enum import StrEnum
from typing import Protocol, TypeVar, cast

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from pydantic import BaseModel, ValidationError

from psview_agent.core.config import Settings, StructuredOutputMode
from psview_agent.core.errors import ModelIncompleteResponseError, ModelInvalidOutputError
from psview_agent.domain.agent import (
    AgentConfiguration,
    CompanyAgentConfigurationDraft,
    OutreachPlanDraft,
)
from psview_agent.domain.candidate import CandidateProfile
from psview_agent.domain.company import CompanyContextInput, SourceSegment
from psview_agent.domain.conversation import ConversationMessage, ConversationState
from psview_agent.domain.decisions import AgentDecision, AgentDecisionDraft, CandidateAnalysis
from psview_agent.domain.enums import OutreachStage
from psview_agent.domain.evaluation import (
    GeneratedResponseDraft,
    ResponseEvaluation,
)
from psview_agent.domain.retrieval import RetrievedEvidence
from psview_agent.integrations.models.errors import map_openai_error
from psview_agent.integrations.models.protocol import ModelGateway
from psview_agent.integrations.models.structured_output import (
    build_response_format,
    is_unsupported_format_error,
    mode_sequence,
    prompt_json_instructions,
)
from psview_agent.prompts.action_planning import build_action_planning_prompts
from psview_agent.prompts.candidate_analysis import build_candidate_analysis_prompts
from psview_agent.prompts.company_configuration import build_company_configuration_prompts
from psview_agent.prompts.outreach_planning import build_outreach_planning_prompts
from psview_agent.prompts.response_evaluation import build_response_evaluation_prompts
from psview_agent.prompts.response_generation import build_response_generation_prompts
from psview_agent.prompts.response_revision import build_response_revision_prompts
from psview_agent.utils.text import sanitize_model_strings

LOGGER = logging.getLogger(__name__)
TModel = TypeVar("TModel", bound=BaseModel)


class ModelWorkload(StrEnum):
    GENERAL_CHAT = "general_chat"
    STRUCTURED_JSON = "structured_json"
    CODING_BACKEND = "coding_backend"


class _ChatCreateCallable(Protocol):
    async def __call__(self, **kwargs: object) -> object: ...


class OpenAICompatibleModelGateway(ModelGateway):
    """Gateway around OpenAI-compatible chat completions."""

    def __init__(self, *, client: AsyncOpenAI, settings: Settings) -> None:
        self._client = client
        self._settings = settings
        self._semaphore = asyncio.Semaphore(settings.model.concurrency_limit)
        self._cached_modes: dict[str, StructuredOutputMode] = {}

    async def configure_company_agent(
        self,
        *,
        context: CompanyContextInput,
        source_segments: Sequence[SourceSegment],
    ) -> CompanyAgentConfigurationDraft:
        system_prompt, user_prompt = build_company_configuration_prompts(
            context=context,
            source_segments=list(source_segments),
        )
        return await self._request_structured(
            output_model=CompanyAgentConfigurationDraft,
            schema_name="company_agent_configuration",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            workload=ModelWorkload.STRUCTURED_JSON,
        )

    async def generate_outreach_plan(
        self,
        *,
        configuration: AgentConfiguration,
        candidate: CandidateProfile,
        target_role: str,
        target_role_description: str | None,
        retrieved_evidence: Sequence[RetrievedEvidence],
    ) -> OutreachPlanDraft:
        system_prompt, user_prompt = build_outreach_planning_prompts(
            configuration=configuration,
            candidate=candidate,
            target_role=target_role,
            target_role_description=target_role_description,
            retrieved_evidence=list(retrieved_evidence),
        )
        draft = await self._request_structured(
            output_model=OutreachPlanDraft,
            schema_name="outreach_plan",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            workload=ModelWorkload.STRUCTURED_JSON,
        )
        sanitized_draft = sanitize_model_strings(draft)
        return self._normalize_outreach_plan(
            draft=sanitized_draft,
            retrieved_evidence=retrieved_evidence,
        )

    async def analyze_candidate_reply(
        self,
        *,
        configuration: AgentConfiguration,
        candidate: CandidateProfile,
        target_role: str,
        target_role_description: str | None,
        state: ConversationState,
        history: Sequence[ConversationMessage],
        candidate_reply: str,
    ) -> CandidateAnalysis:
        system_prompt, user_prompt = build_candidate_analysis_prompts(
            configuration=configuration,
            candidate=candidate,
            target_role=target_role,
            target_role_description=target_role_description,
            state=state,
            history=history,
            candidate_reply=candidate_reply,
        )
        return await self._request_structured(
            output_model=CandidateAnalysis,
            schema_name="candidate_analysis",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            workload=ModelWorkload.STRUCTURED_JSON,
        )

    async def plan_next_action(
        self,
        *,
        configuration: AgentConfiguration,
        candidate: CandidateProfile,
        target_role: str,
        target_role_description: str | None,
        state: ConversationState,
        analysis: CandidateAnalysis,
        history: Sequence[ConversationMessage],
        retrieved_evidence: Sequence[RetrievedEvidence],
    ) -> AgentDecisionDraft:
        system_prompt, user_prompt = build_action_planning_prompts(
            configuration=configuration,
            candidate=candidate,
            target_role=target_role,
            target_role_description=target_role_description,
            state=state,
            analysis=analysis,
            history=history,
            retrieved_evidence=retrieved_evidence,
        )
        return await self._request_structured(
            output_model=AgentDecisionDraft,
            schema_name="agent_decision",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            workload=ModelWorkload.STRUCTURED_JSON,
        )

    async def generate_candidate_response(
        self,
        *,
        configuration: AgentConfiguration,
        candidate: CandidateProfile,
        target_role: str,
        target_role_description: str | None,
        state: ConversationState,
        decision: AgentDecision,
        history: Sequence[ConversationMessage],
        candidate_reply: str,
    ) -> GeneratedResponseDraft:
        system_prompt, user_prompt = build_response_generation_prompts(
            configuration=configuration,
            candidate=candidate,
            target_role=target_role,
            target_role_description=target_role_description,
            state=state,
            decision=decision,
            history=history,
            candidate_reply=candidate_reply,
        )
        draft = await self._request_structured(
            output_model=GeneratedResponseDraft,
            schema_name="generated_response",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            workload=ModelWorkload.GENERAL_CHAT,
        )
        sanitized_draft = sanitize_model_strings(draft)
        return self._normalize_response_fact_ids(draft=sanitized_draft, decision=decision)

    async def evaluate_candidate_response(
        self,
        *,
        configuration: AgentConfiguration,
        candidate: CandidateProfile,
        target_role: str,
        target_role_description: str | None,
        state: ConversationState,
        decision: AgentDecision,
        history: Sequence[ConversationMessage],
        candidate_reply: str,
        response: GeneratedResponseDraft,
    ) -> ResponseEvaluation:
        system_prompt, user_prompt = build_response_evaluation_prompts(
            configuration=configuration,
            candidate=candidate,
            target_role=target_role,
            target_role_description=target_role_description,
            state=state,
            decision=decision,
            history=history,
            candidate_reply=candidate_reply,
            response=response,
        )
        return await self._request_structured(
            output_model=ResponseEvaluation,
            schema_name="response_evaluation",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            workload=ModelWorkload.STRUCTURED_JSON,
        )

    async def revise_candidate_response(
        self,
        *,
        configuration: AgentConfiguration,
        candidate: CandidateProfile,
        target_role: str,
        target_role_description: str | None,
        state: ConversationState,
        decision: AgentDecision,
        history: Sequence[ConversationMessage],
        candidate_reply: str,
        response: GeneratedResponseDraft,
        evaluation: ResponseEvaluation | None,
        deterministic_violations: Sequence[str],
    ) -> GeneratedResponseDraft:
        system_prompt, user_prompt = build_response_revision_prompts(
            configuration=configuration,
            candidate=candidate,
            target_role=target_role,
            target_role_description=target_role_description,
            state=state,
            decision=decision,
            history=history,
            candidate_reply=candidate_reply,
            response=response,
            evaluation=evaluation,
            deterministic_violations=deterministic_violations,
        )
        draft = await self._request_structured(
            output_model=GeneratedResponseDraft,
            schema_name="revised_response",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            workload=ModelWorkload.GENERAL_CHAT,
        )
        sanitized_draft = sanitize_model_strings(draft)
        return self._normalize_response_fact_ids(draft=sanitized_draft, decision=decision)

    async def _request_structured(
        self,
        *,
        output_model: type[TModel],
        schema_name: str,
        system_prompt: str,
        user_prompt: str,
        workload: ModelWorkload,
    ) -> TModel:
        exceptions: list[Exception] = []
        model_name = self._resolve_model_name(workload)
        for mode in self._mode_attempts(model_name):
            try:
                return await self._request_with_mode(
                    output_model=output_model,
                    schema_name=schema_name,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    mode=mode,
                    model_name=model_name,
                )
            except Exception as exc:
                mapped = map_openai_error(exc)
                if mapped is not exc:
                    if isinstance(mapped, Exception) and is_unsupported_format_error(
                        str(exc),
                        mode=mode,
                    ):
                        exceptions.append(mapped)
                        continue
                    raise mapped from exc
                if (
                    is_unsupported_format_error(str(exc), mode=mode)
                    and self._settings.model.structured_output_mode is StructuredOutputMode.AUTO
                ):
                    exceptions.append(exc if isinstance(exc, Exception) else Exception(str(exc)))
                    continue
                raise
        raise ModelInvalidOutputError(
            f"no structured output mode succeeded for {schema_name}: "
            + "; ".join(str(exc) for exc in exceptions)
        )

    def _resolve_model_name(self, workload: ModelWorkload) -> str:
        if workload is ModelWorkload.GENERAL_CHAT:
            return self._settings.model.general_chat_model_name or self._settings.model.model_name
        if workload is ModelWorkload.CODING_BACKEND:
            return self._settings.model.coding_backend_model_name or self._settings.model.model_name
        return self._settings.model.structured_json_model_name or self._settings.model.model_name

    def _mode_attempts(self, model_name: str) -> list[StructuredOutputMode]:
        cached_mode = self._cached_modes.get(model_name)
        if cached_mode is not None:
            return [cached_mode]
        return mode_sequence(
            self._settings.model.provider,
            self._settings.model.structured_output_mode,
        )

    async def _request_with_mode(
        self,
        *,
        output_model: type[TModel],
        schema_name: str,
        system_prompt: str,
        user_prompt: str,
        mode: StructuredOutputMode,
        model_name: str,
    ) -> TModel:
        prompt_suffix = ""
        response_format = build_response_format(mode, schema_name, output_model)
        if mode in {StructuredOutputMode.JSON_OBJECT, StructuredOutputMode.PROMPT_JSON}:
            prompt_suffix = "\n" + prompt_json_instructions(output_model)
        content = await self._call_chat_completion(
            model_name=model_name,
            system_prompt=system_prompt + prompt_suffix,
            user_prompt=user_prompt,
            response_format=response_format,
        )
        parsed = self._parse_content_as_model(content=content, output_model=output_model)
        self._cached_modes[model_name] = mode
        return sanitize_model_strings(parsed)

    async def _call_chat_completion(
        self,
        *,
        model_name: str,
        system_prompt: str,
        user_prompt: str,
        response_format: dict[str, object] | None,
    ) -> str:
        async with self._semaphore:
            create = cast(_ChatCreateCallable, self._client.chat.completions.create)
            payload: dict[str, object] = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": self._settings.model.temperature,
                "max_tokens": self._settings.model.max_output_tokens,
                "extra_body": self._settings.model.extra_body,
            }
            if response_format is not None:
                payload["response_format"] = response_format
            response_object = await create(**payload)
        response = cast(ChatCompletion, response_object)
        request_id_obj = getattr(response, "_request_id", None)
        request_id = request_id_obj if isinstance(request_id_obj, str) else None
        LOGGER.info(
            "model completion succeeded",
            extra={
                "provider": self._settings.model.provider.value,
                "model_name": model_name,
                "structured_output_mode": (
                    self._cached_modes[model_name].value
                    if model_name in self._cached_modes
                    else self._settings.model.structured_output_mode.value
                ),
                "provider_request_id": request_id,
            },
        )
        if not response.choices:
            raise ModelIncompleteResponseError("provider returned no choices")
        content = response.choices[0].message.content
        if content is None or not content.strip():
            raise ModelIncompleteResponseError("provider returned empty content")
        return content

    def _clean_json_text(self, content: str) -> str:
        content = content.strip()
        first_brace = content.find('{')
        last_brace = content.rfind('}')
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            return content[first_brace:last_brace + 1].strip()
        return content

    def _parse_content_as_model(self, *, content: str, output_model: type[TModel]) -> TModel:
        cleaned_content = self._clean_json_text(content)
        try:
            return output_model.model_validate_json(cleaned_content)
        except ValidationError as exc:
            repaired = self._attempt_repair(
                content=cleaned_content,
                errors=str(exc),
                output_model=output_model,
            )
            if repaired is None:
                raise ModelInvalidOutputError("structured output validation failed") from exc
            return repaired

    def _attempt_repair(
        self,
        *,
        content: str,
        errors: str,
        output_model: type[TModel],
    ) -> TModel | None:
        if self._settings.model.repair_attempts <= 0:
            return None
        try:
            raw = json.loads(content)
        except json.JSONDecodeError:
            return None
        if not isinstance(raw, dict):
            return None

        import re
        from enum import Enum

        # 1. Map camelCase and kebab-case keys to snake_case
        def to_snake(name: str) -> str:
            s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
            s2 = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
            return s2.replace('-', '_')

        normalized_raw = {}
        for k, v in raw.items():
            normalized_raw[to_snake(k)] = v
        raw = normalized_raw

        # 2. Drop extra keys to avoid validation issues with extra="forbid"
        for key in list(raw.keys()):
            if key not in output_model.model_fields:
                raw.pop(key)

        # 3. Heal missing or incorrect value types
        for field_name, field_info in output_model.model_fields.items():
            if field_name not in raw:
                # Fill default values or factories
                if field_info.default is not None:
                    raw[field_name] = field_info.default
                elif field_info.default_factory is not None:
                    raw[field_name] = field_info.default_factory()
                else:
                    # Provide type-based fallbacks for missing fields
                    annotation = field_info.annotation
                    if annotation is str:
                        raw[field_name] = "N/A"
                    elif annotation is float or annotation is int:
                        raw[field_name] = 0
                    elif annotation is bool:
                        raw[field_name] = False
                    elif hasattr(annotation, "__origin__") and annotation.__origin__ is list:
                        raw[field_name] = []
                    elif isinstance(annotation, type) and issubclass(annotation, Enum):
                        raw[field_name] = list(annotation)[0]
            else:
                val = raw[field_name]
                annotation = field_info.annotation

                # Wrap non-list elements if list is expected
                if hasattr(annotation, "__origin__") and annotation.__origin__ is list:
                    if not isinstance(val, list):
                        if val is None:
                            raw[field_name] = []
                        else:
                            raw[field_name] = [str(val)]
                
                # Coerce numeric values
                elif annotation is float:
                    try:
                        raw[field_name] = float(val)
                    except (ValueError, TypeError):
                        raw[field_name] = 0.0
                elif annotation is int:
                    try:
                        raw[field_name] = int(val)
                    except (ValueError, TypeError):
                        raw[field_name] = 0

                # Coerce boolean values
                elif annotation is bool and not isinstance(val, bool):
                    if isinstance(val, str):
                        raw[field_name] = val.lower() in ("true", "1", "yes")
                    else:
                        raw[field_name] = bool(val)

                # Coerce enum values (match string to member value or name case-insensitively)
                elif isinstance(annotation, type) and issubclass(annotation, Enum):
                    if isinstance(val, str):
                        matched = None
                        for member in annotation:
                            if member.value.lower() == val.lower() or member.name.lower() == val.lower().replace('-', '_'):
                                matched = member
                                break
                        if matched is not None:
                            raw[field_name] = matched
                        else:
                            raw[field_name] = list(annotation)[0]

        try:
            return output_model.model_validate(raw)
        except ValidationError:
            LOGGER.debug("repair failed", extra={"error_category": "model_invalid_output"})
            return None

    def _normalize_outreach_plan(
        self,
        *,
        draft: OutreachPlanDraft,
        retrieved_evidence: Sequence[RetrievedEvidence],
    ) -> OutreachPlanDraft:
        _ = retrieved_evidence
        stage_defaults = {
            OutreachStage.INITIAL_OUTREACH: {
                "objective": (
                    "Open a relevant conversation with a specific reason for reaching out."
                ),
                "trigger": "candidate background appears relevant",
            },
            OutreachStage.FOLLOW_UP: {
                "objective": "Follow up briefly and politely after no response.",
                "trigger": "no reply to the initial note",
            },
            OutreachStage.FINAL_CLOSEOUT: {
                "objective": "Close the loop gracefully while leaving the door open.",
                "trigger": "no reply after the follow-up",
            },
        }
        normalized_messages = []
        changed = False
        for message in draft.messages:
            defaults = stage_defaults[message.stage]
            objective = defaults["objective"]
            trigger = defaults["trigger"]
            if objective != message.objective or trigger != message.trigger:
                changed = True
            normalized_messages.append(
                {
                    **message.model_dump(),
                    "objective": objective,
                    "trigger": trigger,
                }
            )
        if not changed:
            return draft
        return draft.model_validate(
            {
                "overall_intent": draft.overall_intent,
                "messages": normalized_messages,
            }
        )

    def _normalize_response_fact_ids(
        self,
        *,
        draft: GeneratedResponseDraft,
        decision: AgentDecision,
    ) -> GeneratedResponseDraft:
        _ = decision
        return draft
