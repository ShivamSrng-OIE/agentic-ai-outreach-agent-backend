# Phase 01 - Foundation

## Objective
Create the backend project skeleton and core runtime infrastructure, including the full configuration subsystem.

## Inputs / Dependencies
- [backend-implementation-roadmap.md](/e:/psview-ai-agent/backend/docs/backend-implementation-roadmap.md)
- [configuration-model.md](/e:/psview-ai-agent/backend/docs/configuration-model.md)

## Modules / Files To Add
- `pyproject.toml`
- `.python-version`
- `uv.lock`
- `.gitignore`
- `src/psview_agent/main.py`
- `src/psview_agent/core/config.py`
- `src/psview_agent/core/config_loader.py`
- `src/psview_agent/core/config_merge.py`
- `src/psview_agent/core/env_placeholders.py`
- `src/psview_agent/core/config_redaction.py`
- `src/psview_agent/core/errors.py`
- `src/psview_agent/core/exception_handlers.py`
- `src/psview_agent/core/logging.py`
- `src/psview_agent/core/middleware.py`
- `src/psview_agent/core/lifespan.py`
- `src/psview_agent/api/router.py`
- `src/psview_agent/api/routes/health.py`
- `.env.example`
- `config.yaml.example`

## Implementation Tasks
- Initialize the `uv` project with Python 3.12 and a `src` layout.
- Add runtime and development dependencies, including `pyyaml` and `python-dotenv`.
- Implement typed settings models and enums.
- Implement YAML loading, dotenv loading, placeholder resolution, merge logic, and settings validation.
- Add production-only validation for required secrets and CORS rules.
- Add request ID middleware backed by a `ContextVar`.
- Add JSON logging with redaction-safe metadata.
- Add typed application errors and global exception handlers.
- Add the FastAPI app factory, lifespan wiring, CORS, and `GET /health` plus `GET /ready`.

## Interfaces Affected
- `GET /health`
- `GET /ready`
- `CONFIG_FILE` file-selection behavior
- startup error behavior for invalid config

## Tests To Add
- config loading precedence tests
- placeholder parsing tests
- empty optional override tests
- required placeholder failure tests
- request ID middleware tests
- health and readiness API tests
- production wildcard CORS validation tests

## Exit Criteria
- `uv sync` succeeds
- `uv run ruff check .` succeeds
- `uv run mypy src` succeeds
- `uv run pytest` succeeds
- app starts with local `config.yaml` + `.env`

## Risks To Avoid
- letting `.env` overwrite deployment-provided environment variables
- letting secrets leak in logs or exceptions
- mixing route logic with configuration or startup logic
- treating missing required placeholder values as empty strings

