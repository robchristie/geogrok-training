# Observability Plan

## Goal

Build an analyst-facing observability and review surface for this repo that
helps us:

- inspect chips and pairs directly
- understand benchmark successes and failures visually
- surface outliers and confusable examples
- accumulate human review and annotations over time

This system should accelerate model development in this repo. It is not the
product application itself.

## Current Status

Phase 1 is now implemented enough to support real review loops, with the first
review-artifact layer in place.

Working pieces:

- Python API scaffold in `src/geogrok/obs/api.py`
- run indexing from `artifacts/runs/`
- chip and pair parquet loading via `src/geogrok/obs/data.py`
- first real chip quicklook rendering in `src/geogrok/obs/quicklook.py`
- live read-only endpoints:
  - `GET /api/runs`
  - `GET /api/runs/{run_id}`
  - `GET /api/runs/{run_id}/failures`
  - `GET /api/chips`
  - `GET /api/chip-facets`
  - `GET /api/chips/{chip_id}`
  - `GET /api/chips/{chip_id}/image`
  - `GET /api/pairs`
  - `GET /api/pairs/{pair_key}`
- first review queue materialization from saved benchmark embeddings
- teacher-student disagreement queues for pan-adapt runs
- pair composite imagery and lightweight pair-review persistence
- annotation-aware filtering and a dedicated review queue surface
- queue bookmarking and grouped review worklists
- on-demand chip review artifact generation in `src/geogrok/obs/review_artifacts.py`
- live review-artifact endpoints:
  - `GET /api/review-artifacts/runtime`
  - `GET /api/chips/{chip_id}/review-artifact`
  - `GET /api/chips/{chip_id}/review-artifact/content`
  - `GET /api/pairs/{pair_key}/review-artifact`
- Svelte review components that prefer artifact-backed chip and pair display
- SvelteKit routes for `/chips`, `/pairs`, `/runs`, and `/runs/[runId]` wired to
  live backend data

Still pending for the next observability phase:

- copying or building `kakadujs` WASM assets under `web/static/kakadujs/` so
  browser-side HTJ2K decode can become active on this node
- richer review controls such as queue-level export and annotation facets

Current node status:

- `pykdu` is installed and `.j2c` artifact generation is active
- `kakadujs` browser assets are installed under `web/static/kakadujs/`
- the runtime now reports both:
  - `pykdu_available=true`
  - `kakadujs_assets_available=true`

## Hard Constraints

- Training and evaluation must continue to use raw NITF / GeoTIFF reads through
  the repo-local GDAL + Kakadu path.
- Review imagery may use compressed derived artifacts for human consumption, but
  those artifacts must never be used as training inputs.
- `reference/geogrok/` is reference material only. We can adapt ideas and
  selected components, but we are not merging the full product stack.

## Reuse Policy

Use `reference/geogrok/` as a component donor and pattern reference.

Do reuse or adapt:

- `reference/geogrok/libs/py/pykdu`
  - for review-only HTJ2K / J2C artifact generation and decode
  - consume it as an external package; do not copy it into this repo
- `reference/geogrok/services/ui/src/lib/Jp2k16.ts`
- `reference/geogrok/services/ui/src/lib/jp2k16.worker.ts`
- `reference/geogrok/services/ui/src/lib/Viewer.svelte`
  - as the basis for a high-bit-depth browser viewer
- `reference/geogrok/services/ui/src/lib/components/OverlayCanvas.svelte`
  - as a reference for future annotation overlays
- `reference/geogrok/libs/py/eopm/src/eopm/api/main.py`
- `reference/geogrok/libs/py/eopm/src/eopm/api/chip.py`
  - as API design references, not as drop-in code

Do not pull in:

- Postgres / pgvector
- MinIO / S3 product storage assumptions
- Celery
- EOPM ingest workflows
- product-specific schema or service deployment logic
- a vendored copy of `pykdu`

## Architecture

The observability stack should have three layers.

### 1. Scientific Truth Plane

This is the existing repo data layer:

- raw mirrored assets under `datasets/spacenet.ai/`
- manifests under `datasets/manifests/...`
- pairs under `datasets/pairs/...`
- benchmark outputs under `artifacts/runs/...`

This plane is the source of truth for training, evaluation, and reproducibility.

### 2. Review Artifact Plane

This is a derived human-consumption layer:

- review-only HTJ2K / J2C chips or windows
- quick metadata sidecars
- failure queues and review tables

Suggested layout:

```text
artifacts/observability/
  review_artifacts/
    chips/
      <chip_id>.j2c
      <chip_id>.png
      <chip_id>.json
    pairs/
      <pair_key>.json
  review_tables/
    run_<id>_false_negatives.parquet
    run_<id>_false_positives.parquet
    run_<id>_disagreement.parquet
  annotations/
    review.sqlite
```

### 3. Review Application Plane

This is the human-facing system:

- Python observability API in this repo
- SvelteKit UI in this repo
- browser-side Kakadu WASM decode for review artifacts
- fallback rendering from cached PNG review artifacts when HTJ2K decode is not
  yet active

## Storage Model

Use three storage types.

### Parquet

For immutable analytical artifacts:

- manifests
- pairs
- run summaries
- review candidate tables
- failure tables

### SQLite

For mutable review state:

- annotations
- queue state
- notes
- tags
- reviewer actions

### HTJ2K / J2C review artifacts

For review-only image payloads:

- browser-efficient
- bit-depth aware
- shader-friendly
- cacheable client-side

## Review Artifact Policy

Review artifacts are derived products and must be treated separately from
scientific inputs.

Rules:

- review artifacts must record provenance back to source `chip_id` and asset
- review artifacts may be visually lossless or lightly lossy
- review artifacts must not be consumed by training code
- review artifact storage belongs under `artifacts/observability/`, not under
  `datasets/`

Suggested codec profiles:

- `review_lossless`
- `review_visually_lossless`
- `review_fast_preview`

## Backend Plan

Add a lightweight Python observability API under `src/geogrok/obs/`.

Initial modules:

- `src/geogrok/obs/run_index.py`
  - index benchmark and training runs from `artifacts/runs/`
- `src/geogrok/obs/quicklook.py`
  - review-image rendering contract
- `src/geogrok/obs/review_tables.py`
  - materialize false positives, false negatives, disagreements
- `src/geogrok/obs/annotations.py`
  - SQLite persistence for human review
- `src/geogrok/obs/api.py`
  - HTTP surface for the SvelteKit app

Initial API endpoints:

- `GET /health`
- `GET /api/runs`
- `GET /api/runs/{run_id}`
- `GET /api/chips`
- `GET /api/chips/{chip_id}`
- `GET /api/chips/{chip_id}/image`
- `GET /api/pairs`
- `GET /api/pairs/{pair_key}`
- `GET /api/pairs/{pair_key}/images`
- `GET /api/runs/{run_id}/failures`

## Frontend Plan

Add a SvelteKit app under `web/`.

Initial routes:

- `/`
  - overview and navigation
- `/chips`
  - browse chips by city / split / sensor / modality
- `/pairs`
  - browse and inspect labeled pairs
- `/runs`
  - browse benchmark and training runs
- later:
  - `/runs/[runId]`
  - `/review`

UI principles:

- image-led layout
- low-chrome analyst workspace
- chips and pairs are the primary visual anchor
- metadata should be dense but quiet
- controls should be simple and useful, not decorative

## Initial Views

### Chip Browser

Purpose:

- inspect chips from train / val / test
- filter by city, split, modality, sensor
- inspect edge cases visually

### Pair Inspector

Purpose:

- inspect `positive_exact`, `positive_weak`, and `negative_hard`
- display overlap, time delta, center distance, city, split
- provide linked black/white/gamma controls over decoded review artifacts
- eventually annotate label quality and issues

### Run Explorer

Purpose:

- browse benchmark runs and training runs
- inspect retrieval failures and teacher-student disagreements
- view top-k examples around a query chip

## Review Queues

Materialize the following queues from benchmark outputs.

### False negatives

- labeled positives that rank badly
- useful for finding model misses or label issues

### False positives

- labeled negatives that rank too highly
- especially useful for confusable same-city non-overlap pairs

### Teacher-student disagreements

- teacher high, student low
- student high, teacher low

### Data quality queue

- cloud / shadow
- striping
- low dynamic range
- obvious geometry problems

## Reference Reuse Map

The intended reuse path is:

### `reference/geogrok/libs/py/pykdu`

Use for:

- review artifact encoding
- fast decode / ROI-oriented helpers
- future review-only HTJ2K pipelines

Do not use it to redefine the primary training input path.

### `reference/geogrok/services/ui`

Use for:

- viewer structure
- worker-based Kakadu WASM decode
- dynamic range adjustment and shader ideas
- annotation overlay references later

Do not copy product-specific routing or service assumptions blindly.

### `reference/geogrok/libs/py/eopm`

Use for:

- API shape ideas
- benchmark/report organization ideas
- review/export workflow patterns

Do not import its product DB / task / object-store assumptions into this repo.

## Phase Plan

### Phase 1: Read-only observability

Deliverables:

- this plan document
- `src/geogrok/obs/` scaffold
- `web/` SvelteKit scaffold
- run indexing
- chip / pair / run browsing shell

### Phase 2: Review artifacts

Deliverables:

- HTJ2K / J2C review-only artifact generation
- browser-side Kakadu WASM decode
- review cache path
- chip/pair viewer integrated with real artifacts

### Phase 3: Failure and outlier queues

Deliverables:

- false positive queue
- false negative queue
- teacher-student disagreement queue
- adversarial hard-negative review queue

### Phase 4: Human review

Deliverables:

- SQLite-backed annotations
- issue tags and free-text notes
- queue state
- export of reviewed examples

## Immediate Concrete Repo Work

The first concrete implementation in this repo should be:

1. `docs/observability-plan.md`
2. `src/geogrok/obs/` scaffold with run indexing and API skeleton
3. `web/` SvelteKit shell
4. README links so observability becomes a visible repo concern

After that, the next implementation step should be the read-only run/chip/pair
browser before we add annotations or review-only codestream generation.
