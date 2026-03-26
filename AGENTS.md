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

## Frontend Tooling

- The SvelteKit UI under `web/` uses:
  - `svelte-check` for type and template validation
  - `Biome` for linting, formatting, and import organization
- Frontend checks should run from `web/`:
  - `npm run check`
  - `npm run lint`
  - `npm run format:check`
- Use the existing Biome scope and avoid broadening it to generated assets such
  as `web/static/kakadujs/` unless there is a deliberate reason.

## Runtime Expectations

- Raster- or manifest-backed commands should assume the repo-local GDAL +
  Kakadu runtime.
- When working with real imagery, source `.local/gdal-kakadu/env.sh` first
  unless the code path activates GDAL explicitly through the Python helper.
- GPU-backed training, pretrained encoder benchmarks, and torch-based retrieval
  work should use the training extra:
  - `uv sync --extra dev --extra train`

## Dataset Layout

- `datasets` in the repo root is expected to be a symlink to
  `/mnt/media/datasets` on this node.
- `datasets/spacenet.ai/` is expected to mirror the SpaceNet bucket key layout.
- New manifest, smoke-test, and local-path logic should preserve that bucket
  mirror assumption unless there is an explicit repo-wide migration.

## Benchmark Discipline

- Substantive benchmark or evaluation-protocol changes should update:
  - `README.md`
  - `docs/implementation-plan.md`
  - `docs/NOTEBOOK.md`
- When a benchmark result is reported, prefer preserving comparability with the
  prior protocol. If comparability is broken, call that out explicitly in the
  docs or notebook entry.
- Avoid silently overwriting meaningful experiment outputs when comparing
  methods. Prefer distinct run roots for materially different experiments unless
  the overwrite is an intentional smoke run.

## Performance-First Workflow

- For meaningful data-path, training-path, or evaluation-path changes, benchmark
  before and after when practical.
- Treat throughput and latency as first-class outcomes, not secondary details.
- If a change materially shifts performance or benchmark results, record that in
  `docs/NOTEBOOK.md`.

## Notebook Maintenance

- `docs/NOTEBOOK.md` is the permanent engineering notebook for the repo.
- After substantive implementation work, benchmark changes, or measured results,
  update `docs/NOTEBOOK.md` in the same turn unless the user explicitly asks not
  to.
- Notebook entries should record:
  - what was implemented
  - why it was done
  - key implementation detail
  - observed results or measured outcomes
- Keep the notebook additive and durable: prefer new dated or sequential entries
  over rewriting history unless correcting a factual mistake.
