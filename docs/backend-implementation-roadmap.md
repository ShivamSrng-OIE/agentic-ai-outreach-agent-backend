# Backend Implementation Roadmap

## Summary
This repository root, `e:\psview-ai-agent\backend`, is the backend project root. Docs live in `docs/`, source lives in `src/`, and there is no nested `backend/backend` layout.

The backend is a stateless recruiting-agent simulator with four pillars:

1. strict YAML + `.env` configuration with required secret placeholder resolution
2. company grounding that turns raw context into source segments, evidence facts, and an in-memory evidence corpus
3. autonomous conversation orchestration with LangGraph and deterministic policies
4. grounded response generation that retrieves company evidence on every turn, checks it, evaluates it, and revises once if needed

## Product Boundaries
### In scope
- FastAPI API
- YAML + `.env` configuration loading
- provider-neutral model gateway built on the OpenAI-compatible client
- deterministic lexical retrieval over submitted company context only
- LangGraph orchestration
- deterministic policy enforcement and response checks
- typed decision trace without hidden chain-of-thought
- tests, Docker, CI, and backend docs

### Out of scope
- frontend implementation
- auth, accounts, sessions, or persistence
- database, Redis, queues, jobs, or background workers
- vector databases, embeddings, semantic search, or external RAG
- file ingestion, scraping, crawling, or third-party data sources
- external recruiter messaging integrations

## Canonical Repo Shape
- `.github/workflows/backend-ci.yml`
- `.gitignore`
- `.dockerignore`
- `Dockerfile`
- `Makefile`
- `README.md`
- `.env.example`
- `config.yaml.example`
- `docs/`
- `src/psview_agent/`
- `tests/`
- `pyproject.toml`
- `uv.lock`

## Public API Contract
- `GET /health`
- `GET /ready`
- `POST /api/v1/agents/configure`
- `POST /api/v1/conversations/start`
- `POST /api/v1/conversations/turn`

Stable response and state rules:
- errors always return `code`, `message`, `request_id`, and `details`
- the server is stateless for conversations
- the client sends the full conversation session back on every turn
- decision traces expose intent, sentiment, engagement, stage, objective, action, retrieved facts, missing info, confidence, continue or stop, rationale summary, and policy overrides
- decision traces never expose hidden chain-of-thought

## Architecture
### Request lifecycle
1. validate typed request payload
2. analyze candidate reply
3. update explicit conversation state
4. build retrieval query
5. retrieve top company facts from the in-memory corpus
6. plan the next recruiting action
7. enforce deterministic policies
8. generate a grounded response
9. run deterministic checks
10. evaluate semantic quality
11. revise once if needed
12. fallback if needed
13. return updated state plus public decision trace

### Grounding model
- source material comes only from submitted company context
- company context is segmented deterministically
- evidence facts retain source provenance
- retrieval is deterministic, lexical, and pure Python
- only retrieved evidence IDs may be used in planning and response generation

See [grounding-and-retrieval.md](/e:/psview-ai-agent/backend/docs/grounding-and-retrieval.md).

## Configuration Architecture
- `config.yaml.example` is committed
- `config.yaml` is local runtime config and ignored
- `.env.example` is committed
- `.env` is ignored and auto-loaded locally
- required secret references use exact `${ENV_VAR}` syntax
- optional values use `null`, omission, or literals
- workload-specific model routing is supported through general-chat, structured-JSON, and coding/backend model names
- precedence is fixed:
  - shell or deployment environment > `.env`
  - explicit environment overrides > resolved YAML > hardcoded defaults

See [configuration-model.md](/e:/psview-ai-agent/backend/docs/configuration-model.md).

## Runtime Stack
- Python 3.12
- `uv`
- `fastapi`
- `uvicorn[standard]`
- `pydantic`
- `pydantic-settings`
- `pyyaml`
- `python-dotenv`
- `openai`
- `langgraph`

Install command:

```bash
uv add fastapi "uvicorn[standard]" pydantic pydantic-settings pyyaml python-dotenv openai langgraph
```

## Phase Order
1. [Phase 01 - Foundation](/e:/psview-ai-agent/backend/docs/phases/phase-01-foundation.md)
2. [Phase 02 - Domain Models](/e:/psview-ai-agent/backend/docs/phases/phase-02-domain-models.md)
3. [Phase 03 - Provider Integration](/e:/psview-ai-agent/backend/docs/phases/phase-03-provider-integration.md)
4. [Phase 04 - Agent Configuration](/e:/psview-ai-agent/backend/docs/phases/phase-04-agent-configuration.md)
5. [Phase 05 - Conversation Start](/e:/psview-ai-agent/backend/docs/phases/phase-05-conversation-start.md)
6. [Phase 06 - Policies and Response Checks](/e:/psview-ai-agent/backend/docs/phases/phase-06-policies-and-response-checks.md)
7. [Phase 07 - LangGraph Orchestration](/e:/psview-ai-agent/backend/docs/phases/phase-07-langgraph-orchestration.md)
8. [Phase 08 - Conversation Turn](/e:/psview-ai-agent/backend/docs/phases/phase-08-conversation-turn.md)
9. [Phase 09 - Reliability and Security](/e:/psview-ai-agent/backend/docs/phases/phase-09-reliability-and-security.md)
10. [Phase 10 - Quality Gates](/e:/psview-ai-agent/backend/docs/phases/phase-10-quality-gates.md)
11. [Phase 11 - Docker and CI](/e:/psview-ai-agent/backend/docs/phases/phase-11-docker-and-ci.md)
12. [Phase 12 - Documentation and Final Verification](/e:/psview-ai-agent/backend/docs/phases/phase-12-documentation-and-final-verification.md)

## Definition Of Done
- configuration loads from defaults, YAML, and environment with clear startup failures
- company context produces a distinct profile, persona, source segments, and evidence corpus
- outreach preview returns exactly three ordered messages
- retrieval uses only in-memory company evidence and is deterministic
- the agent answers direct questions before steering elsewhere
- unsupported compensation, visa, work-mode, benefits, revenue, funding, customer, and team-size facts are not invented
- AI identity is disclosed when asked
- opt-out, rejection, busy, and hostile replies follow deterministic policy rules
- every turn runs retrieval, checks, evaluation, and at most one revision
- no conversation state is stored server-side
- tests require no network by default
- coverage stays at or above 85%
- `uv lock --check`, frozen sync, Ruff, mypy, pytest, and Docker build all pass
