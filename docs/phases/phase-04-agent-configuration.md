# Phase 04 - Agent Configuration

## Objective
Turn raw company context into source segments, a structured company profile, a grounded evidence corpus, and a recruiting persona with mandatory boundaries.

## Inputs / Dependencies
- Phases 01 through 03 completed

## Modules / Files To Add
- `src/psview_agent/prompts/company_configuration.py`
- `src/psview_agent/services/agent_configuration.py`
- `src/psview_agent/retrieval/corpus_builder.py`
- `src/psview_agent/api/routes/agents.py`

## Implementation Tasks
- segment company context deterministically
- prompt the model for company profile, evidence drafts, and persona
- validate evidence references against real source segments
- assign evidence IDs in Python
- merge mandatory persona boundaries into the final persona
- return a typed `AgentConfiguration`

## Interfaces Affected
- `POST /api/v1/agents/configure`

## Tests To Add
- source segment IDs are stable
- evidence IDs are assigned in Python
- unsupported source references fail
- materially different companies produce materially different personas
- mandatory boundaries are always present

## Exit Criteria
- configuration output is grounded, typed, and traceable
- model output does not control IDs, timestamps, or hidden policy rules

## Risks To Avoid
- letting model output invent unsupported company facts
- trusting model-supplied IDs
- hardcoding one fixed persona regardless of company context
