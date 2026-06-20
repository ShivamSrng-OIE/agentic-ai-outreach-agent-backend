# Phase 03 - Provider Integration

## Objective
Implement the provider-neutral model gateway around OpenAI-compatible chat completions, with robust structured-output handling and test doubles.

## Inputs / Dependencies
- Phase 01 foundation completed
- Phase 02 domain models completed

## Modules / Files To Add
- `src/psview_agent/integrations/models/protocol.py`
- `src/psview_agent/integrations/models/client.py`
- `src/psview_agent/integrations/models/gateway.py`
- `src/psview_agent/integrations/models/structured_output.py`
- `src/psview_agent/integrations/models/errors.py`
- `tests/fakes/fake_model_gateway.py`

## Implementation Tasks
- Build a client factory using `AsyncOpenAI`.
- Implement `OpenAICompatibleModelGateway`.
- Resolve workload-specific model names from settings so different backend tasks can use different models without route-level branching.
- Support structured output modes: auto, JSON schema, JSON object, and prompt JSON fallback.
- Add capability downgrade behavior only for unsupported structured-output features.
- Map provider failures into typed application errors.
- Capture safe provider metadata such as request IDs.
- Ensure provider credentials come only from resolved settings, never from routes or request payloads.
- Apply provider default base URLs after provider selection.
- Validate OpenRouter-specific metadata only when OpenRouter is selected.

## Interfaces Affected
- model gateway protocol
- structured output request/response contract inside service layer
- fake gateway test contract

## Tests To Add
- JSON schema path
- JSON object path
- prompt JSON path
- auto downgrade behavior
- no downgrade on auth failure
- invalid output repair
- refusal, timeout, rate-limit, and connection mapping
- provider request-ID extraction
- workload-aware model selection uses the configured general-chat and structured-JSON model names
- fallback to `MODEL_NAME` works when workload-specific names are unset

## Exit Criteria
- normal tests perform zero network calls
- all structured-output paths are covered
- provider switching is configuration-only

## Risks To Avoid
- leaking OpenAI naming into domain logic
- swallowing provider errors into generic failures
- coupling gateway behavior to route handlers
