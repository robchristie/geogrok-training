`uv` manages the Python environment in this repo. Use `uv` for dependency installation and for running repo tooling.

## CI Workflow

Current CI steps:

- `uv sync --extra dev`
- `uv run ruff check .`
- `uv run ty check`
- `uv run --extra dev pytest -q`

## Lint And Type Checking

- `ruff` is the fast lint/import-hygiene pass.
- `ty` is enabled as a pragmatic static check against the active `uv` environment.
- `thirdparty/` is to be excluded from repo tooling by config.
