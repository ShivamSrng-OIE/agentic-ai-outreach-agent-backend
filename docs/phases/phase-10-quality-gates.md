# Phase 10 - Quality Gates

## Objective
Bring the codebase to the required lint, type-check, test, and coverage standards without weakening quality bars.

## Inputs / Dependencies
- Phases 01 through 09 completed

## Modules / Files To Add
- test coverage expansions as required
- `pyproject.toml` tool configuration finalized

## Implementation Tasks
- run `ruff format`
- run `ruff check`
- run `mypy` in strict mode on `src` and `tests`
- run `pytest` with coverage threshold
- fix root causes instead of suppressing failures

## Interfaces Affected
- none intentionally, except bug fixes revealed by quality gates

## Tests To Add
- any missing tests needed to reach stable coverage and catch regressions

## Exit Criteria
- `uv run ruff format --check .`
- `uv run ruff check .`
- `uv run mypy src tests`
- `uv run pytest`
- coverage is at least 85%

## Risks To Avoid
- adding blanket ignores
- lowering lint, type, or coverage standards to force a pass

