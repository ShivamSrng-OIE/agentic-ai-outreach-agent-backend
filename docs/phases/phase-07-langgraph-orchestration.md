# Phase 07 - LangGraph Orchestration

## Objective
Assemble the autonomous turn workflow with explicit state, retrieval, deterministic routing, one-shot revision, and fallback behavior.

## Inputs / Dependencies
- Phases 01 through 06 completed

## Modules / Files To Add
- `src/psview_agent/agent/graph_state.py`
- `src/psview_agent/agent/nodes.py`
- `src/psview_agent/agent/routing.py`
- `src/psview_agent/agent/graph.py`

## Implementation Tasks
- define typed graph state
- add nodes for analysis, state update, retrieval query, retrieval, action planning, policy enforcement, response generation, checks, evaluation, revision, fallback, and finalization
- wire conditional routing after deterministic checks and evaluation
- enforce revision and recursion limits

## Interfaces Affected
- internal graph invocation contract used by the turn service

## Tests To Add
- happy path
- deterministic-failure revision path
- semantic-failure revision path
- fallback path
- graph termination guarantees

## Exit Criteria
- the graph can be invoked asynchronously
- every route terminates
- retrieval is always part of the turn path before planning and generation

## Risks To Avoid
- hidden mutable server-side state
- planning from non-retrieved company facts
- non-terminating revision loops
