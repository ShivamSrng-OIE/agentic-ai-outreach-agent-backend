# Phase 05 - Conversation Start

## Objective
Use the grounded evidence corpus to retrieve relevant facts, generate the outreach preview, and create the initial stateless conversation session.

## Inputs / Dependencies
- Phases 01 through 04 completed

## Modules / Files To Add
- `src/psview_agent/retrieval/query_builder.py`
- `src/psview_agent/retrieval/scoring.py`
- `src/psview_agent/retrieval/lexical_retriever.py`
- `src/psview_agent/prompts/outreach_planning.py`
- `src/psview_agent/services/conversation_start.py`
- `src/psview_agent/api/routes/conversations.py`

## Implementation Tasks
- build the initial outreach retrieval query
- retrieve top company facts from the in-memory corpus
- generate exactly three ordered preview messages
- validate evidence IDs used in the preview
- create initial conversation state, first agent message, and initial decision trace

## Interfaces Affected
- `POST /api/v1/conversations/start`

## Tests To Add
- preview contains exactly three messages
- preview stages are ordered correctly
- preview messages are candidate-specific
- preview facts all exist in the evidence corpus
- initial state is valid and serializable

## Exit Criteria
- the returned session is fully stateless and ready for later turns
- initial outreach is grounded in retrieved evidence rather than generic messaging

## Risks To Avoid
- treating the preview like a rigid script after the candidate replies
- generating preview messages with unsupported claims
