# Phase 11 - Docker and CI

## Objective
Package the backend for reproducible local and CI execution.

## Inputs / Dependencies
- Phases 01 through 10 completed

## Modules / Files To Add
- `Dockerfile`
- `.dockerignore`
- `Makefile`
- `.github/workflows/backend-ci.yml`

## Implementation Tasks
- build a Python 3.12 image using frozen `uv.lock` dependencies
- exclude development dependencies from runtime image
- run as non-root
- expose port 8000 and add a health check
- implement Make targets for install, sync, lock, dev, format, lint, typecheck, tests, coverage, check, and Docker workflows
- configure GitHub Actions to run frozen sync, Ruff, mypy, pytest, and Docker build

## Interfaces Affected
- local developer command surface
- CI command contract
- Docker runtime command

## Tests To Add
- CI should exercise lint, typing, tests, and Docker build
- live provider tests must remain excluded by default

## Exit Criteria
- `docker build -t psview-agent-api .` succeeds
- CI workflow is deterministic and does not require external model credentials

## Risks To Avoid
- copying `.env` into the image
- baking secrets into Docker layers
- running live provider tests in CI

