# Phase 02 - Domain Models

## Objective
Define the full typed domain model surface for company context, persona, outreach, conversation state, decisions, evaluations, and API payloads.

## Inputs / Dependencies
- Phase 01 foundation completed

## Modules / Files To Add
- `src/psview_agent/domain/base.py`
- `src/psview_agent/domain/enums.py`
- `src/psview_agent/domain/company.py`
- `src/psview_agent/domain/agent.py`
- `src/psview_agent/domain/candidate.py`
- `src/psview_agent/domain/conversation.py`
- `src/psview_agent/domain/decisions.py`
- `src/psview_agent/domain/evaluation.py`
- `src/psview_agent/domain/api.py`
- supporting normalization utilities in `src/psview_agent/utils/`

## Implementation Tasks
- Create the shared strict base model.
- Create enums with `StrEnum`.
- Implement company-context request models.
- Implement evidence, persona, outreach, conversation-state, and decision-trace models.
- Add normalization helpers for whitespace, deduplication, IDs, truncation, and timestamps.
- Enforce cross-field rules, including valid outreach order and closed-state rules.

## Interfaces Affected
- configure request/response models
- conversation start request/response models
- conversation turn request/response models
- stable error response model

## Tests To Add
- unknown field rejection
- whitespace-only rejection
- length constraints
- duplicate normalization
- invalid evidence references
- invalid closed-state combinations
- outreach stage order validation

## Exit Criteria
- all domain validation tests pass
- unknown fields are rejected consistently
- normalization and cross-field rules are deterministic

## Risks To Avoid
- untyped dict-shaped responses
- putting provider-specific types into domain models
- leaving timestamps or IDs to model-generated output

