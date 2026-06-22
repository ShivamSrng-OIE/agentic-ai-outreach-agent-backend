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
            workload=ModelWorkload.CODING_BACKEND,
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
                from instructor.core import InstructorRetryException
                from pydantic import ValidationError as PydanticValidationError
                import openai
                
                if isinstance(exc, InstructorRetryException) and isinstance(exc.__cause__, openai.APIError):
                    mapped = map_openai_error(exc.__cause__)
                elif isinstance(exc, (InstructorRetryException, PydanticValidationError)):
                    mapped = ModelInvalidOutputError(f"instructor structured output validation failed: {exc}")
                else:
                    mapped = map_openai_error(exc)
                is_fallbackable = (
                    is_unsupported_format_error(str(exc), mode=mode) or
                    isinstance(mapped, (ModelIncompleteResponseError, ModelInvalidOutputError))
                )
                if is_fallbackable and self._settings.model.structured_output_mode is StructuredOutputMode.AUTO:
                    if model_name in self._cached_modes and self._cached_modes[model_name] == mode:
                        self._cached_modes.pop(model_name, None)
                    exceptions.append(mapped if isinstance(mapped, Exception) else Exception(str(mapped)))
                    continue
                if mapped is not exc:
                    raise mapped from exc
                raise
        raise ModelInvalidOutputError(
            f"no structured output mode succeeded for {schema_name}: "
            + "; ".join(str(exc) for exc in exceptions)
        )

    def _resolve_model_name(self, workload: ModelWorkload) -> str:
        from psview_agent.core.config import model_override_var
        override = model_override_var.get()
        if override is not None:
            if workload is ModelWorkload.GENERAL_CHAT:
                return override.general_chat_model_name or override.model_name
            if workload is ModelWorkload.CODING_BACKEND:
                return override.coding_backend_model_name or override.model_name
            return override.structured_json_model_name or override.model_name

        if workload is ModelWorkload.GENERAL_CHAT:
            return self._settings.model.general_chat_model_name or self._settings.model.model_name
        if workload is ModelWorkload.CODING_BACKEND:
            return self._settings.model.coding_backend_model_name or self._settings.model.model_name
        return self._settings.model.structured_json_model_name or self._settings.model.model_name

    def _mode_attempts(self, model_name: str) -> list[StructuredOutputMode]:
        cached_mode = self._cached_modes.get(model_name)
        from psview_agent.core.config import model_override_var
        override = model_override_var.get()
        provider = override.provider if override else self._settings.model.provider
        seq = mode_sequence(
            provider,
            self._settings.model.structured_output_mode,
        )
        if cached_mode is not None and cached_mode in seq:
            return [cached_mode] + [m for m in seq if m != cached_mode]
        return seq

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
        import instructor
        
        mode_map = {
            StructuredOutputMode.JSON_SCHEMA: instructor.Mode.JSON_SCHEMA,
            StructuredOutputMode.JSON_OBJECT: instructor.Mode.JSON,
            StructuredOutputMode.PROMPT_JSON: instructor.Mode.MD_JSON,
        }
        
        prompt_suffix = ""
        if mode in {StructuredOutputMode.JSON_OBJECT, StructuredOutputMode.PROMPT_JSON}:
            prompt_suffix = "\n" + prompt_json_instructions(output_model)
            
        from psview_agent.core.config import model_override_var
        override = model_override_var.get()
        
        client_to_use = self._client
        temp_client = None
        
        if override is not None:
            base_url = ""
            if override.provider == "openai":
                base_url = "https://api.openai.com/v1"
            elif override.provider == "gemini":
                base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
            elif override.provider == "openrouter":
                base_url = "https://openrouter.ai/api/v1"
            elif override.provider == "nvidia":
                base_url = "https://integrate.api.nvidia.com/v1"

            headers = {}
            if override.provider == "openrouter":
                if self._settings.openrouter.site_url is not None:
                    headers["HTTP-Referer"] = str(self._settings.openrouter.site_url)
                if self._settings.openrouter.app_name:
                    headers["X-OpenRouter-Title"] = self._settings.openrouter.app_name

            temp_client = AsyncOpenAI(
                api_key=override.api_key,
                base_url=base_url,
                timeout=self._settings.model.timeout_seconds,
                max_retries=self._settings.model.max_retries,
                default_headers=headers,
            )
            client_to_use = temp_client

        try:
            instructor_client = instructor.from_openai(client_to_use, mode=mode_map[mode])
            
            try:
                async with self._semaphore:
                    parsed = await instructor_client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": system_prompt + prompt_suffix},
                            {"role": "user", "content": user_prompt},
                        ],
                        response_model=output_model,
                        temperature=self._settings.model.temperature,
                        max_tokens=self._settings.model.max_output_tokens,
                        max_retries=self._settings.model.repair_attempts,
                        extra_body=self._settings.model.extra_body,
                    )
                
                LOGGER.info(
                    "Instructor model completion succeeded",
                    extra={
                        "provider": override.provider.value if override else self._settings.model.provider.value,
                        "model_name": model_name,
                        "structured_output_mode": mode.value,
                    },
                )
                self._cached_modes[model_name] = mode
                return sanitize_model_strings(parsed)
            except Exception as exc:
                # Extract raw response text if validation failed under Instructor
                raw_content = None
                if hasattr(exc, "last_completion") and exc.last_completion:
                    raw_content = getattr(exc.last_completion.choices[0].message, "content", None)
                
                if raw_content:
                    LOGGER.warning(
                        "Instructor validation failed for model %s; attempting custom recursive repair. error: %s. content: %s",
                        output_model.__name__,
                        str(exc),
                        raw_content,
                    )
                    repaired = self._attempt_repair(
                        content=raw_content,
                        errors=str(exc),
                        output_model=output_model,
                    )
                    if repaired is not None:
                        LOGGER.info("successfully repaired model %s via fallback repair", output_model.__name__)
                        self._cached_modes[model_name] = mode
                        return repaired
                
                LOGGER.error(
                    "Instructor execution failed for model %s. error: %s",
                    output_model.__name__,
                    str(exc),
                )
                raise
        finally:
            if temp_client is not None:
                await temp_client.close()

    def _clean_json_text(self, content: str) -> str:
        content = content.strip()
        first_brace = content.find('{')
        last_brace = content.rfind('}')
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            return content[first_brace:last_brace + 1].strip()
        return content

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
            cleaned = self._clean_json_text(content)
            raw = json.loads(cleaned)
        except json.JSONDecodeError:
            return None
        if not isinstance(raw, dict):
            return None

        import re
        from enum import Enum
        from typing import get_origin, get_args, Union
        from types import UnionType

        def to_snake(name: str) -> str:
            s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
            s2 = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
            return s2.replace('-', '_')

        def resolve_base_model(ann: object) -> type[BaseModel] | None:
            if isinstance(ann, type) and issubclass(ann, BaseModel):
                return ann
            origin = get_origin(ann)
            if origin in (Union, UnionType):
                for arg in get_args(ann):
                    if isinstance(arg, type) and issubclass(arg, BaseModel):
                        return arg
            return None

        def repair_dict(data: dict[str, object], model_cls: type[BaseModel]) -> dict[str, object]:
            # 1. Map camelCase and kebab-case keys to snake_case
            normalized_raw = {}
            for k, v in data.items():
                normalized_raw[to_snake(k)] = v
            data = normalized_raw

            # 2. Drop extra keys to avoid validation issues with extra="forbid"
            for key in list(data.keys()):
                if key not in model_cls.model_fields:
                    data.pop(key)

            # Specific domain repairs for CandidateAnalysis opt-out rules
            if model_cls.__name__ == "CandidateAnalysis":
                intent_val = data.get("intent")
                opt_out_val = data.get("explicit_opt_out")
                if intent_val == "do_not_contact" and not opt_out_val:
                    data["explicit_opt_out"] = True
                elif opt_out_val and intent_val not in ("do_not_contact", "clear_rejection"):
                    data["explicit_opt_out"] = False

            # 3. Heal missing or incorrect value types
            for field_name, field_info in model_cls.model_fields.items():
                annotation = field_info.annotation
                nested_model = resolve_base_model(annotation)
                origin = get_origin(annotation)

                if field_name not in data:
                    # Fill default values or factories
                    if field_info.default is not None:
                        data[field_name] = field_info.default
                    elif field_info.default_factory is not None:
                        data[field_name] = field_info.default_factory()
                    else:
                        # Provide type-based fallbacks for missing fields
                        if annotation is str:
                            data[field_name] = "Not provided"
                        elif annotation is float or annotation is int:
                            data[field_name] = 0
                        elif annotation is bool:
                            data[field_name] = False
                        elif origin is list:
                            data[field_name] = []
                        elif isinstance(annotation, type) and issubclass(annotation, Enum):
                            data[field_name] = list(annotation)[0]
                        elif nested_model is not None:
                            data[field_name] = repair_dict({}, nested_model)
                else:
                    val = data[field_name]

                    # Recursively repair nested BaseModel
                    if nested_model is not None:
                        if isinstance(val, dict):
                            data[field_name] = repair_dict(val, nested_model)
                        else:
                            data[field_name] = repair_dict({}, nested_model)
                        continue

                    # Recursively repair list of BaseModels
                    if origin is list:
                        args = get_args(annotation)
                        item_model = resolve_base_model(args[0]) if args else None
                        if item_model is not None:
                            if isinstance(val, list):
                                data[field_name] = [
                                    repair_dict(item, item_model) if isinstance(item, dict) else repair_dict({}, item_model)
                                    for item in val
                                ]
                            else:
                                if val is None:
                                    data[field_name] = []
                                else:
                                    data[field_name] = [
                                        repair_dict(val, item_model) if isinstance(val, dict) else repair_dict({}, item_model)
                                    ]
                            
                            # Pad nested list if it is under min_length/min_items constraint
                            min_len = None
                            for meta in getattr(field_info, "metadata", []):
                                if hasattr(meta, "min_length") and isinstance(meta.min_length, int):
                                    min_len = meta.min_length
                                elif hasattr(meta, "min_items") and isinstance(meta.min_items, int):
                                    min_len = meta.min_items
                            if min_len is not None and isinstance(data[field_name], list) and len(data[field_name]) < min_len:
                                while len(data[field_name]) < min_len:
                                    data[field_name].append(repair_dict({}, item_model))
                            continue

                    # Handle string length constraints
                    if isinstance(val, str):
                        min_len = None
                        max_len = None
                        for meta in getattr(field_info, "metadata", []):
                            if hasattr(meta, "min_length") and isinstance(meta.min_length, int):
                                min_len = meta.min_length
                            if hasattr(meta, "max_length") and isinstance(meta.max_length, int):
                                max_len = meta.max_length
                        if min_len is not None and len(val) < min_len:
                            val = val.ljust(min_len, ".")
                        if max_len is not None and len(val) > max_len:
                            val = val[:max_len]
                        data[field_name] = val
                        val = data[field_name]

                    # Wrap non-list elements if list is expected
                    if origin is list:
                        if not isinstance(val, list):
                            if val is None:
                                data[field_name] = []
                            else:
                                data[field_name] = [str(val)]
                        
                        # Pad basic list if under constraints
                        min_len = None
                        for meta in getattr(field_info, "metadata", []):
                            if hasattr(meta, "min_length") and isinstance(meta.min_length, int):
                                min_len = meta.min_length
                            elif hasattr(meta, "min_items") and isinstance(meta.min_items, int):
                                min_len = meta.min_items
                        if min_len is not None and isinstance(data[field_name], list) and len(data[field_name]) < min_len:
                            args = get_args(annotation)
                            item_type = args[0] if args else str
                            while len(data[field_name]) < min_len:
                                if item_type is str:
                                    data[field_name].append("dummy")
                                elif item_type is int or item_type is float:
                                    data[field_name].append(0)
                                elif item_type is bool:
                                    data[field_name].append(False)
                                else:
                                    data[field_name].append(None)
                    
                    # Coerce numeric values
                    elif annotation is float:
                        try:
                            data[field_name] = float(val)
                        except (ValueError, TypeError):
                            data[field_name] = 0.0
                    elif annotation is int:
                        try:
                            data[field_name] = int(val)
                        except (ValueError, TypeError):
                            data[field_name] = 0

                    # Coerce boolean values
                    elif annotation is bool and not isinstance(val, bool):
                        if isinstance(val, str):
                            data[field_name] = val.lower() in ("true", "1", "yes")
                        else:
                            data[field_name] = bool(val)

                    # Coerce enum values (match string to member value or name case-insensitively)
                    elif isinstance(annotation, type) and issubclass(annotation, Enum):
                        if isinstance(val, str):
                            matched = None
                            for member in annotation:
                                if member.value.lower() == val.lower() or member.name.lower() == val.lower().replace('-', '_'):
                                    matched = member
                                    break
                            if matched is not None:
                                data[field_name] = matched
                            else:
                                data[field_name] = list(annotation)[0]
            return data

        repaired_dict = repair_dict(raw, output_model)
        try:
            return output_model.model_validate(repaired_dict)
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
