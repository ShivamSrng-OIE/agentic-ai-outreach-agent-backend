# Phase 09 - Reliability And Security

## Objective
Harden the backend against oversized requests, unsafe logging, prompt injection, and misuse of provider or evidence contracts.

## Inputs / Dependencies
- Phases 01 through 08 completed

## Modules / Files To Add
- expand middleware, logging, redaction, and validation modules as needed

## Implementation Tasks
- enforce request body-size limits
- enforce history and turn limits
- enforce provider concurrency limits
- redact secrets from diagnostics and exceptions
- treat company context, history, and candidate replies as untrusted prompt input
- prevent non-retrieved evidence IDs from being used in responses

## Interfaces Affected
- error responses for oversized requests and invalid conversation state
- logging and startup diagnostics

## Tests To Add
- oversized body rejection
- request ID on error responses
- secret redaction in diagnostics
- prompt-injection safety expectations
- CORS behavior

## Exit Criteria
- logs do not leak secrets or full payloads
- unsupported evidence usage is rejected deterministically

## Risks To Avoid
- logging full company context or candidate replies
- exposing provider credentials, raw provider bodies, or resolved secrets
