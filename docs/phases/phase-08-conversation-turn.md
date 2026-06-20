# Phase 08 - Conversation Turn

## Objective
Implement the multi-turn API flow that accepts client-supplied session state and returns the next grounded agent response plus updated state.

## Inputs / Dependencies
- Phases 01 through 07 completed

## Modules / Files To Add
- `src/psview_agent/services/conversation_turn.py`
- expand `src/psview_agent/api/routes/conversations.py`

## Implementation Tasks
- validate conversation state before graph execution
- reject closed sessions with HTTP 409
- invoke the compiled graph with the candidate reply and current session
- build the next candidate and agent messages
- return updated state, evaluation summary, and public decision trace

## Interfaces Affected
- `POST /api/v1/conversations/turn`

## Tests To Add
- turn happy path
- closed-state rejection
- turn-limit rejection
- direct-question prioritization
- opt-out closure

## Exit Criteria
- the turn endpoint is stateless from the server perspective
- returned state is sufficient for the next call

## Risks To Avoid
- trusting client input for next action fields
- mutating any server-side conversation store
