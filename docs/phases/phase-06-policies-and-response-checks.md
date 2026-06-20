# Phase 06 - Policies and Response Checks

## Objective
Implement deterministic rules that constrain the autonomous agent and validate its response before it can be returned.

## Inputs / Dependencies
- Phases 01 through 05 completed

## Modules / Files To Add
- `src/psview_agent/agent/policies.py`
- `src/psview_agent/agent/response_checks.py`
- `src/psview_agent/agent/fallbacks.py`

## Implementation Tasks
- Implement deterministic policies for opt-out, rejection, busy replies, hostility, AI identity questions, and missing-info questions.
- Implement evidence reference validation.
- Enforce response length, question-count, HTML rejection, code-fence rejection, repetition checks, and sensitive-topic claim checks.
- Require AI disclosure when asked directly.
- Define safe fallback responses when validation or policy handling fails.

## Interfaces Affected
- policy decision surface used by the graph
- deterministic validation output passed into evaluation/revision routing

## Tests To Add
- do-not-contact closes conversation
- rejection prevents persuasion
- busy response pauses respectfully
- hostile response exits calmly
- AI question forces disclosure
- unknown compensation, visa, and work-mode responses do not invent facts
- invalid fact IDs fail
- length, repetition, and multiple-question failures are caught

## Exit Criteria
- policy unit tests pass
- fallback behavior is deterministic and safe

## Risks To Avoid
- letting model output bypass policy decisions
- allowing unsupported salary or sponsorship claims

