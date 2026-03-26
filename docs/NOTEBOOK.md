# Project Notebook

Last updated: 2026-03-26

## Purpose

This file is the permanent engineering notebook for the repo. It records the
main implementation steps that have been taken, why they were taken, how they
were implemented, and what was observed.

The intent is different from `README.md` and different from
`implementation-plan.md`:

- `README.md` explains what exists now and how to run it
- `implementation-plan.md` explains the intended staged roadmap
- `NOTEBOOK.md` records the actual sequence of engineering decisions and what
  they produced

## Entry Format

Each entry uses the same structure:

- `Implemented`: what changed
- `Reason`: why the change was made
- `Detail`: the implementation shape or protocol change
- `Results`: what the repo or benchmark showed afterwards

## Entry 01: Repo-Local GDAL + Kakadu Runtime

- `Implemented`: repo-local GDAL + Kakadu build and validation workflow
- `Reason`: the project needs reliable NITF / JP2K access for WV2/WV3 PAN
  products, and stock Python GDAL wheels are not sufficient for the Kakadu path
- `Detail`:
  - added `scripts/build_gdal_kakadu.sh`
  - added `scripts/check_gdal_kakadu.sh`
  - added `scripts/smoke_gdal_kakadu_rasters.sh`
  - added Python-side runtime activation in `src/geogrok/io/gdal_env.py`
  - build script handles SWIG discovery, Kakadu auxiliary library generation,
    Python bindings, `RPATH` patching, and environment activation via
    `.local/gdal-kakadu/env.sh`
- `Results`:
  - `gdalinfo` sees both `JP2KAK` and `NITF`
  - `from osgeo import gdal` works against the repo-local runtime
  - real dataset NITF opens correctly, for example Jacksonville WV3 PAN
    products under the mirrored `datasets/spacenet.ai/` tree

## Entry 02: SpaceNet Manifest Pipeline

- `Implemented`: first manifest generation pipeline
- `Reason`: model work needed a stable, reproducible dataset view rather than
  ad hoc bucket exploration
- `Detail`:
  - added `src/geogrok/data/manifests.py`
  - added CLI `geogrok-make-manifests`
  - outputs:
    - `assets.parquet`
    - `scenes.parquet`
    - `chips.parquet`
    - `summary.json`
  - bootstrap split policy:
    - `train`: default
    - `val`: `Omaha`
    - `test`: `UCSD`
- `Results`:
  - the repo can now normalize the local SpaceNet metadata slice into stable
    tabular artifacts
  - manifests resolve correctly against the real bucket-mirror root
    `datasets/spacenet.ai/`, not the earlier `raw/` assumption

## Entry 03: Real Bucket-Mirror Layout Fixes

- `Implemented`: dataset root handling and smoke-path fixes for the actual local
  layout
- `Reason`: the local dataset is a mirror of the `spacenet.ai` bucket, not a
  hand-curated extracted directory
- `Detail`:
  - updated raster smoke scripts to search `datasets/spacenet.ai/`
  - updated manifest logic to use `datasets/spacenet.ai` as the default
    download root
  - verified uppercase `.NTF` / `.TIF` handling
- `Results`:
  - smoke reads now use real local mirrored assets by default
  - `local_exists` and `local_path` in manifests resolve correctly against the
    mirror

## Entry 04: GDAL-Backed Raster I/O and Optional Chip Extraction

- `Implemented`: raster reader and initial chip extraction path
- `Reason`: manifests alone are not enough; the repo needed a reliable bridge
  from a manifest row to an actual image window read
- `Detail`:
  - added `src/geogrok/io/raster.py`
  - added `src/geogrok/data/chips.py`
  - added `scripts/smoke_chip_extraction.sh`
  - the raster layer exposes inspection, window reads, and optional materialized
    chip export
- `Results`:
  - real mirrored PAN chips can be cut successfully from source rasters
  - materialized chip export works, but this immediately raised the question of
    whether one-file-per-chip is the right training format

## Entry 05: Shift To Manifest-Only / On-Demand Training

- `Implemented`: manifest-only, on-demand chip dataset as the primary training
  path
- `Reason`: materializing millions of chips as files was likely to create a
  storage and operational burden before the right format was even known
- `Detail`:
  - added `src/geogrok/data/runtime.py`
  - on-demand path reads chip windows directly from source rasters through GDAL
  - retained chip extraction only as an optional fallback
- `Results`:
  - the repo can now read real PAN chips directly from mirrored WV3 NITF files
  - early smoke reads returned `uint16` `(1, 1024, 1024)` arrays with values
    consistent with 11-bit PAN data

## Entry 06: Performance Measurement As A First-Class Requirement

- `Implemented`: explicit throughput and latency benchmarking for the data path
- `Reason`: the project needed continuous visibility into performance so tuning
  decisions would be driven by measurements rather than guesswork
- `Detail`:
  - added `src/geogrok/data/benchmark.py`
  - added CLI `geogrok-benchmark-chips`
  - reports:
    - `samples_per_second`
    - `megapixels_per_second`
    - `mebibytes_per_second`
    - mean / p50 / p95 / max latency
  - benchmark output is written to structured JSON artifacts
- `Results`:
  - on this node, early real-data smoke reads were around:
    - `171.16 samples/s`
    - `179.47 MPix/s`
    - `342.32 MiB/s`
    - mean latency around `5.83 ms`

## Entry 07: Trainer-Facing Dataset And Training Loop

- `Implemented`: trainer dataset wrapper and deterministic dry-run training loop
- `Reason`: once data loading existed, the next requirement was a model-facing
  interface that preserved performance visibility instead of hiding it
- `Detail`:
  - added `src/geogrok/data/training.py`
  - added `src/geogrok/training/loop.py`
  - added `src/geogrok/training/baseline.py`
  - the training path exposes:
    - `read_ms`
    - `transform_ms`
    - `total_ms`
  - epoch summaries are written to `metrics.jsonl` and `summary.json`
- `Results`:
  - trainer-path benchmarking showed that normalization and resize add visible
    cost on top of source reads
  - the dry-run loop surfaced warm/cold performance differences clearly

## Entry 08: Initial Retrieval Baselines

- `Implemented`: successive retrieval baselines from deterministic to learned
- `Reason`: the retrieval stack needed a working benchmark harness before model
  selection could become meaningful
- `Detail`:
  - deterministic baseline in `src/geogrok/retrieval/baseline.py`
  - shallow learned projection in `src/geogrok/retrieval/learned.py`
  - tiny NumPy CNN in `src/geogrok/retrieval/cnn.py`
  - PyTorch PAN encoder in `src/geogrok/retrieval/torch_encoder.py`
- `Results`:
  - the early `scene_id`-based smoke benchmark was useful for plumbing but too
    weak semantically
  - the small GPU-backed PyTorch baseline was the first learned model to beat
    the deterministic control on early-rank retrieval under that earlier setup

## Entry 09: Explicit Pair Protocol Replaced Same-Raster Relevance

- `Implemented`: explicit chip pair mining and pair-based retrieval evaluation
- `Reason`: “same scene” was too weak for large heterogeneous satellite images;
  relevance needed to be grounded in world geometry, overlap, and hard
  non-overlap
- `Detail`:
  - added `src/geogrok/data/pairs.py`
  - pair labels:
    - `positive_exact`
    - `positive_weak`
    - `negative_hard`
  - added ROI extraction in ground units and asset-pair pruning
  - added pair evaluator `src/geogrok/retrieval/pair_eval.py`
- `Results`:
  - pair-based retrieval became the main honest benchmark
  - real smoke mining on Jacksonville / Omaha / UCSD produced nontrivial pair
    sets rather than synthetic placeholders

## Entry 10: Held-Out Pair Benchmark

- `Implemented`: train on `train`, evaluate on held-out `val/test` pairs
- `Reason`: the earlier train/train and same-raster evaluations were too
  optimistic
- `Detail`:
  - added held-out smoke path
  - pair labels now include split-aware metadata
  - the evaluation contract became explicit and reusable
- `Results`:
  - the held-out benchmark exposed the real generalization gap immediately
  - the early learned encoder that looked fine on the easy setup became weak on
    held-out overlap retrieval

## Entry 11: Frozen Pretrained Benchmark

- `Implemented`: frozen encoder benchmark over the held-out pair protocol
- `Reason`: before spending more time on custom PAN students, the repo needed a
  benchmark against strong off-the-shelf encoders
- `Detail`:
  - added `src/geogrok/retrieval/pretrained_benchmark.py`
  - benchmarked generic and remote-sensing models under the same pair protocol
  - model set included:
    - `resnet50`
    - `resnet101`
    - `resnet152`
    - `remoteclip_rn50`
    - `georsclip_vit_b32_ret2`
    - `dinov2_vitb14`
    - `dinov3_vitb16`
- `Results`:
  - current smoke summary in `artifacts/runs/pretrained-benchmark-smoke/summary.json`
  - strong generic controls were surprisingly robust on PAN
  - current top frozen references are:
    - `resnet152` for top-k exact retrieval
    - `dinov3_vitb16` for early-rank quality
  - the remote-sensing CLIP controls did not beat the best generic frozen
    controls on this PAN setup

## Entry 12: First Teacher-Student PAN Adaptation Benchmark

- `Implemented`: initial PAN adaptation benchmark
- `Reason`: once the frozen-model controls were known, the next step was to
  measure whether a PAN-only student could learn from a frozen teacher embedding
  space
- `Detail`:
  - added `src/geogrok/retrieval/pan_adapt_benchmark.py`
  - teacher: frozen pretrained encoder
  - student: PAN-only CNN
  - initial objective:
    - pairwise contrastive loss
    - cosine alignment to teacher embeddings
- `Results`:
  - this benchmark worked end to end, but the first student results were far
    below teacher quality
  - conclusion at that stage: the bottleneck was adaptation quality, not just
    having a teacher

## Entry 13: Stronger Residual Student And Richer Distillation Loss

- `Implemented`: stronger `residual_cnn` student and richer adaptation objective
- `Reason`: the first student was too weak, and the loss used only per-chip
  alignment
- `Detail`:
  - added `baseline_cnn` and `residual_cnn` student options
  - added:
    - teacher similarity-structure matching
    - multi-view PAN augmentation
    - view-consistency loss
  - smoke runner defaults moved to `dinov3_vitb16 + residual_cnn`
- `Results`:
  - this materially improved exact-overlap retrieval relative to the plain
    alignment baseline
  - the student was still far from teacher quality, but the signal moved in the
    right direction

## Entry 14: Weighted Positive Pairs And Hard-Negative Terms

- `Implemented`: weighted `positive_exact` vs `positive_weak` supervision and a
  hard-negative loss
- `Reason`: not all positives are equally strong, and same-city non-overlap
  confusions needed an explicit training penalty
- `Detail`:
  - exact positives are sampled and weighted more heavily than weak positives
  - hard negatives are penalized with a similarity ceiling
  - later extended to a dynamic teacher-derived ceiling
- `Results`:
  - on some runs this helped the `dinov3` teacher path
  - but the hard-negative term often remained inactive because the sampled
    training negatives were not hard enough or had been trimmed away upstream

## Entry 15: Dynamic Hard-Negative Target

- `Implemented`: dynamic hard-negative margin based on teacher similarities
- `Reason`: a fixed global similarity ceiling was too blunt, and in practice it
  often produced zero hard-negative loss
- `Detail`:
  - negative target computed from teacher negative similarity and the current
    positive teacher similarity scale
  - added:
    - `--hard-negative-gap-scale`
    - `--hard-negative-min-similarity`
- `Results`:
  - retrieval mix shifted slightly, but `hardneg_mean` still often stayed zero
  - conclusion: the issue was not only the target formula, it was the training
    negative pool itself

## Entry 16: Adversarial Hard-Negative Mining

- `Implemented`: adversarial hard-negative mining from teacher-similar
  `negative_hard` pairs
- `Reason`: the hard-negative path needed to act on genuinely confusable
  non-overlap pairs instead of arbitrary negatives
- `Detail`:
  - widened the PAN adaptation training dataset so that chips from
    `negative_hard` pairs are retained
  - added mined-negative selection by teacher similarity
  - new controls:
    - `--adversarial-negative-top-fraction`
    - `--adversarial-negative-max-pairs`
    - `--adversarial-negative-min-teacher-similarity`
  - training now reports:
    - mined hard-negative pair count
    - mean teacher similarity of mined negatives
    - p95 teacher similarity of mined negatives
- `Results`:
  - this is the first stage where the hard-negative term became genuinely active
  - latest smoke result in `artifacts/runs/pan-adapt-smoke/`:
    - teacher: `dinov3_vitb16`
    - student: `residual_cnn`
    - retrieval:
      - `exact_R@10 = 0.087`
      - `any_R@10 = 0.122`
      - `any_MRR = 0.056`
    - mined hard negatives:
      - `30` pairs
      - `teacher_sim_mean = 0.9113`
      - `teacher_sim_p95 = 0.9486`
    - training:
      - `hardneg_mean = 0.2234`
  - important interpretation:
    - this did not improve the aggregate retrieval score yet
    - but it did finally align the training signal with the intended failure
      mode: highly confusable same-city non-overlap pairs

## Entry 17: Observability Architecture And Scaffolds

- `Implemented`: concrete observability plan plus backend/frontend scaffolds
- `Reason`: before continuing deeper into training and evaluation, the repo
  needed a durable path for visual inspection of chips, pairs, and benchmark
  failures
- `Detail`:
  - added [observability-plan.md](/nvme/development/geogrok-training/docs/observability-plan.md)
  - added Python observability scaffold under
    [src/geogrok/obs/](/nvme/development/geogrok-training/src/geogrok/obs/)
    - `run_index.py`
    - `api.py`
  - added SvelteKit observability UI scaffold under
    [web/](/nvme/development/geogrok-training/web/)
  - added optional `obs` extra in
    [pyproject.toml](/nvme/development/geogrok-training/pyproject.toml)
  - explicitly treated `reference/geogrok/` as reference-only, not as a merged
    product stack
- `Results`:
  - the repo now has a documented architecture for:
    - raw scientific truth plane
    - derived review artifact plane
    - SvelteKit + Python review application plane
  - the initial backend scaffold can already index run summaries from
    `artifacts/runs/`
  - the initial UI scaffold establishes routes for:
    - `/`
    - `/chips`
    - `/pairs`
    - `/runs`
  - the intended reuse of `pykdu`, `kakadujs`, and selected viewer/API patterns
    from `reference/geogrok/` is now explicit and bounded

## Entry 18: First Live Observability Endpoints And UI Wiring

- `Implemented`: first real chip-image and data endpoints, plus live `/runs`
  and `/pairs` UI wiring
- `Reason`: the observability scaffolds were useful structurally, but the repo
  still needed real end-to-end inspection of chips, pairs, and benchmark runs
  before the review workflow could become practical
- `Detail`:
  - added `src/geogrok/obs/data.py`
    - resolves manifest and pair parquet paths from the local repo / mirrored
      dataset layout
    - normalizes chip and pair records for API responses
  - added `src/geogrok/obs/quicklook.py`
    - reads real chip windows from source rasters
    - applies percentile-based grayscale display normalization
    - renders PNG quicklooks for browser display
  - expanded `src/geogrok/obs/api.py`
    - `GET /api/chips`
    - `GET /api/chips/{chip_id}`
    - `GET /api/chips/{chip_id}/image`
    - `GET /api/pairs`
    - `GET /api/pairs/{pair_key}`
    - kept `GET /api/runs` backed by `run_index.py`
  - wired Svelte routes to the backend:
    - `web/src/routes/runs/+page.ts`
    - `web/src/routes/pairs/+page.ts`
    - `web/src/routes/runs/+page.svelte`
    - `web/src/routes/pairs/+page.svelte`
  - added Vite dev proxying in `web/vite.config.ts` so the Svelte app can call
    the local Python API directly during development
- `Results`:
  - the repo now has the first real read-only observability loop:
    - browse run summaries from `artifacts/runs/`
    - browse labeled pairs from `pairs.parquet`
    - render real query / candidate chip images from source rasters
  - the implementation stayed within the intended architecture:
    - no product-stack merge from `reference/geogrok/`
    - raw scientific inputs remain separate from future review artifacts
  - repo validation still passes:
    - `uv run ruff check .`
    - `uv run ty check`
    - `uv run --extra dev pytest -q`

## Entry 19: Live Chip Browser And First Failure Queues

- `Implemented`: live `/chips` browser plus run-specific false-negative and
  false-positive review queues
- `Reason`: observability needed to move from “can the UI fetch data?” to “can
  an engineer inspect the actual chips and the actual benchmark mistakes?”
- `Detail`:
  - added `src/geogrok/obs/review_tables.py`
    - loads saved embedding/index artifacts from benchmark runs
    - ranks labeled pairs against the saved embedding space
    - derives:
      - false negatives = labeled positives ranked outside top-k
      - false positives = `negative_hard` pairs ranked inside top-k
  - expanded `src/geogrok/obs/api.py`
    - `GET /api/runs/{run_id}`
    - `GET /api/runs/{run_id}/failures`
    - `GET /api/chip-facets`
  - updated `src/geogrok/obs/run_index.py`
    - pretrained benchmark runs now expose best-model summary metrics
  - wired the Svelte UI to these endpoints:
    - `web/src/routes/chips/+page.ts`
    - `web/src/routes/chips/+page.svelte`
    - `web/src/routes/runs/[runId]/+page.ts`
    - `web/src/routes/runs/[runId]/+page.svelte`
- `Results`:
  - `/chips` is now a real manifest-backed image browser instead of a placeholder
  - run detail pages can now inspect the first meaningful review queues derived
    from saved benchmark artifacts
  - observability is now anchored to the same retrieval contract used by the
    benchmark code, not a separate UI heuristic
  - repo validation still passes:
    - `uv run ruff check .`
    - `uv run ty check`
    - `uv run --extra dev pytest -q`

## Entry 20: Teacher-Student Disagreement Queues

- `Implemented`: pan-adapt disagreement queues for teacher-ahead positives and
  student-confused hard negatives
- `Reason`: failure queues show what a single model gets wrong, but the
  teacher-student adaptation loop also needs visibility into where the student
  diverges most sharply from the teacher
- `Detail`:
  - expanded `src/geogrok/obs/review_tables.py`
    - joins teacher and student ranked pair tables on explicit pair labels
    - computes rank and similarity advantages in both directions
    - materializes disagreement queues for:
      - teacher ahead on positives
      - student ahead on positives
      - student more confused on `negative_hard`
      - teacher more confused on `negative_hard`
  - added `GET /api/runs/{run_id}/disagreements` in
    `src/geogrok/obs/api.py`
  - updated `web/src/routes/runs/[runId]/+page.ts` and
    `web/src/routes/runs/[runId]/+page.svelte` to surface the highest-signal
    disagreement slices directly in the run detail view
- `Results`:
  - real smoke on `artifacts/runs/pan-adapt-smoke/` produced non-empty queues:
    - `teacher_ahead_positives = 357`
    - `student_ahead_positives = 128`
    - `student_confused_negatives = 660`
    - `teacher_confused_negatives = 1321`
  - a representative teacher-ahead positive had:
    - teacher rank `1`
    - student rank `503`
    - teacher similarity `0.913`
    - student similarity `-0.178`
  - a representative student-confused hard negative had:
    - teacher rank `426`
    - student rank `44`
    - teacher similarity `0.442`
    - student similarity `0.739`
  - this gives the observability UI a concrete adaptation-debug surface rather
    than only aggregate retrieval metrics

## Entry 21: Pair Composite Images And Lightweight Review Actions

- `Implemented`: pair composite imagery plus SQLite-backed pair annotations
- `Reason`: review needed to become actionable. Engineers should be able to
  inspect a pair as one visual unit and record judgment without leaving the
  observability surface
- `Detail`:
  - added `src/geogrok/obs/annotations.py`
    - SQLite-backed pair review store under
      `artifacts/observability/annotations/review.sqlite`
    - supports pair-level status, note, and tags
  - expanded `src/geogrok/obs/quicklook.py`
    - added pair composite PNG rendering from source chips
  - expanded `src/geogrok/obs/api.py`
    - `GET /api/pairs/{pair_key}/image`
    - `GET /api/annotations/pairs/{pair_key}`
    - `POST /api/annotations/pairs/{pair_key}`
    - pair annotations are now attached to pair, failure, and disagreement
      payloads
  - added reusable Svelte review control:
    - `web/src/lib/components/PairAnnotation.svelte`
  - updated review surfaces:
    - `web/src/routes/pairs/+page.svelte`
    - `web/src/routes/runs/[runId]/+page.svelte`
- `Results`:
  - the pair browser now renders a single composite strip per pair rather than
    only two independent chip images
  - the run-detail review queues now support inline pair review actions
  - real smoke with the GDAL runtime active confirmed:
    - pair composite render succeeded for a real pair
    - PNG payload size was `26266` bytes at `size=128`
    - a real annotation write/read succeeded with status `interesting`
  - repo validation still passes:
    - `uv run ruff check .`
    - `uv run ty check`
    - `uv run --extra dev pytest -q`
    - `npm run check`

## Entry 22: Annotation-Aware Filtering And Review Queue Surface

- `Implemented`: annotation-aware filtering across observability endpoints plus a
  dedicated `/review` queue page
- `Reason`: once review actions existed, the next bottleneck was prioritization.
  The repo needed a first-class way to focus on unreviewed examples instead of
  only browsing everything
- `Detail`:
  - expanded `src/geogrok/obs/annotations.py`
    - added shared filtering by annotation status:
      - `unreviewed`
      - `reviewed`
      - `confirmed`
      - `incorrect_label`
      - `interesting`
      - `needs_followup`
  - expanded `src/geogrok/obs/api.py`
    - `annotation_status` filtering for:
      - `GET /api/pairs`
      - `GET /api/runs/{run_id}/failures`
      - `GET /api/runs/{run_id}/disagreements`
    - added `GET /api/pair-facets`
  - updated `web/src/routes/pairs/+page.ts` and
    `web/src/routes/pairs/+page.svelte`
    - pair browser now has filter controls for label, city, split, and review
      state
  - updated `web/src/routes/runs/[runId]/+page.ts` and
    `web/src/routes/runs/[runId]/+page.svelte`
    - run queues now have shareable URL-driven review-state filtering
  - added `web/src/routes/review/+page.ts` and
    `web/src/routes/review/+page.svelte`
    - new dedicated analyst worklist for:
      - unreviewed pair labels
      - unreviewed student failures
      - unreviewed teacher-student disagreements
- `Results`:
  - observability now supports a practical “show me the next unreviewed things”
    workflow rather than only free browsing
  - the new queue page is built on the same run artifacts and pair labels as the
    rest of the system, not a separate review-only data source
  - repo validation still passes:
    - `uv run ruff check .`
    - `uv run ty check`
    - `uv run --extra dev pytest -q`
    - `npm run check`

## Entry 23: Queue Bookmarking And Grouped Review Worklists

- `Implemented`: bookmark state inside pair annotations plus bookmark-aware
  filtering and grouping in review surfaces
- `Reason`: once annotation-aware filtering existed, the next useful behavior
  was a durable “come back to this” signal that survives page reloads and can be
  used to reshape the review queue
- `Detail`:
  - expanded `src/geogrok/obs/annotations.py`
    - pair annotations now carry `bookmarked: bool`
    - SQLite schema migrates forward to add `bookmarked` when missing
    - shared review filtering now supports `bookmarked_only`
    - records are sorted with bookmarked items first
  - expanded `src/geogrok/obs/api.py`
    - bookmark state is accepted on pair annotation writes
    - `bookmarked_only` filtering is supported on:
      - `GET /api/pairs`
      - `GET /api/runs/{run_id}/failures`
      - `GET /api/runs/{run_id}/disagreements`
  - updated `web/src/lib/components/PairAnnotation.svelte`
    - added an inline bookmark toggle
  - updated queue routes:
    - `web/src/routes/pairs/+page.ts`
    - `web/src/routes/pairs/+page.svelte`
    - `web/src/routes/runs/[runId]/+page.ts`
    - `web/src/routes/runs/[runId]/+page.svelte`
    - `web/src/routes/review/+page.ts`
    - `web/src/routes/review/+page.svelte`
    - `/review` now includes a dedicated bookmarked section above the active
      work queues
- `Results`:
  - real API smoke confirmed:
    - bookmark write succeeded for a real pair
    - `GET /api/pairs?bookmarked_only=true` returned the bookmarked pair with
      attached annotation state
    - failure queues correctly report zero visible items when no benchmark
      failures have been bookmarked yet
  - this gives the review system a durable triage signal separate from status
    labels such as `interesting` or `needs_followup`

## Entry 24: Review Artifact Layer And HTJ2K-Capable Viewer Path

- `Implemented`: the first review-artifact plane plus a frontend path that
  prefers artifact-backed display over ad hoc PNG composites
- `Reason`: the review UI needed a durable, cacheable image layer that is
  clearly separated from the scientific truth path and can evolve toward
  high-bit-depth browser rendering without changing the rest of the review
  application
- `Detail`:
  - added `src/geogrok/obs/review_artifacts.py`
    - on-demand review artifact generation under:
      - `artifacts/observability/review_artifacts/chips/`
    - metadata sidecars with source provenance, source window, codec profile,
      and artifact path
    - `.j2c` generation through `pykdu` when available
    - cached PNG fallback artifacts when `pykdu` is unavailable
    - new CLI:
      - `geogrok-build-review-artifacts`
  - expanded `src/geogrok/obs/api.py`
    - added:
      - `GET /api/review-artifacts/runtime`
      - `GET /api/chips/{chip_id}/review-artifact`
      - `GET /api/chips/{chip_id}/review-artifact/content`
      - `GET /api/pairs/{pair_key}/review-artifact`
  - added web-side artifact plumbing:
    - `web/src/lib/Jp2k16.ts`
    - `web/src/lib/jp2k16.worker.ts`
    - `web/src/lib/reviewArtifacts.ts`
    - `web/src/lib/components/ReviewChipImage.svelte`
    - `web/src/lib/components/ReviewPairImage.svelte`
    - the browser now:
      - prefers `.j2c` artifacts when both artifact and decoder assets exist
      - otherwise uses cached PNG review artifacts
      - otherwise falls back to the existing source-rendered PNG endpoints
  - updated review-facing routes to use artifact-backed imagery:
    - `/chips`
    - `/pairs`
    - `/review`
    - `/runs/[runId]`
- `Results`:
  - repo validation passed:
    - `uv run ruff check .`
    - `uv run ty check`
    - `uv run --extra dev pytest -q`
    - `npm run check`
  - live API smoke on this node confirmed:
    - `GET /api/review-artifacts/runtime` returned:
      - `pykdu_available=false`
      - `kakadujs_assets_available=false`
    - `GET /api/chips/chip_8af0982d913db245/review-artifact` generated and
      returned a cached PNG artifact with metadata
    - `GET /api/pairs/chip_00f1d70108af2975__chip_0042419062bfa0b1/review-artifact`
      returned paired artifact metadata for two real Jacksonville PAN chips
  - the observability system now has a real derived review plane even before
    `.j2c` emission is active in this repo environment

## Entry 25: Reference `pykdu` Option A Install Path

- `Implemented`: repo-local installer and checker scripts for using the
  reference submodule `pykdu` package as an editable dependency
- `Reason`: the review-artifact layer is now ready for `.j2c`, but the repo
  needed a clean way to enable `pykdu` without copying native binding code into
  this repo
- `Detail`:
  - added `scripts/install_reference_pykdu.sh`
    - uses `reference/geogrok/libs/py/pykdu` as the editable install source
    - reuses this repo’s Kakadu source tree under `third_party/kakadu`
    - stages the Kakadu shared-library layout `pykdu` expects under
      `.build/reference-pykdu-kakadu/lib`
    - performs an editable install with `uv pip install -e`
    - runs a real `uint16` encode/decode smoke round-trip after install
  - added `scripts/check_reference_pykdu.sh`
    - verifies `pykdu` import and a small round-trip check in the active repo
      environment
  - updated `README.md` and `docs/observability-plan.md`
    - the documented policy is now explicit:
      - use the reference submodule as the source of truth
      - do not copy `pykdu` into this repo
- `Results`:
  - the repo now has a concrete Option A path for enabling `.j2c` review
    artifacts without merging native binding code into the training repo
  - verified on this node:
    - `./scripts/install_reference_pykdu.sh` completed successfully
    - `./scripts/check_reference_pykdu.sh` passed
    - `GET /api/review-artifacts/runtime` reported:
      - `pykdu_available=true`
      - `kakadujs_assets_available=false`
    - `GET /api/chips/chip_8af0982d913db245/review-artifact` generated a real
      `.j2c` artifact:
      - `bits_per_sample=11`
      - `file_size_bytes=129519`
    - `GET /api/pairs/chip_00f1d70108af2975__chip_0042419062bfa0b1/review-artifact`
      returned two real `.j2c` artifacts for a Jacksonville pair

## Entry 26: Reference `kakadujs` WASM Sync And Browser Asset Activation

- `Implemented`: repo-local sync/build script for the reference `kakadujs`
  decoder assets, with outputs copied into this repo’s Svelte static assets
- `Reason`: once server-side `.j2c` generation was working, the remaining gap
  was browser-side decode. The review UI needed the WASM assets in this repo
  without merging the full reference UI stack
- `Detail`:
  - added `scripts/sync_reference_kakadujs.sh`
    - initializes `reference/geogrok/third_party/kakadujs` if missing
    - bridges the reference build’s Kakadu source expectation to this repo’s
      top-level `third_party/kakadu`
    - runs the reference `kakadujs` WASM build flow when artifacts are missing
    - copies:
      - `kakadujs.js`
      - `kakadujs.wasm`
      into `web/static/kakadujs/`
  - verified Vite serves the decoder assets from this repo’s UI:
    - `GET /kakadujs/kakadujs.js` returned `200`
  - runtime check now reports:
    - `pykdu_available=true`
    - `kakadujs_assets_available=true`
- `Results`:
  - this node now has the full observability codec path available:
    - server-side `.j2c` review artifact generation
    - browser-side decoder assets for HTJ2K/J2C review rendering
  - repo validation still passes:
    - `uv run ruff check .`
    - `uv run ty check`
    - `uv run --extra dev pytest -q`
    - `cd web && npm run check`

## Entry 27: Linked Tone Controls And Review Overlays

- `Implemented`: linked black/white/gamma controls for decoded review artifacts
  plus contextual overlays on chip and pair views
- `Reason`: once the HTJ2K/J2C review path was active, the next observability
  gap was display control. Analysts need to inspect the same pair under a
  consistent window without flattening everything to static PNG behavior, and
  the review UI needed on-image context instead of pushing all metadata below
  the frame
- `Detail`:
  - updated `web/src/lib/components/ReviewChipImage.svelte`
    - retains decoded sample buffers from `.j2c` review artifacts
    - redraws the canvas reactively when display controls change
    - applies black / white windowing and gamma correction on the decoded
      samples rather than on a pre-flattened image
    - adds lightweight overlays for:
      - chip or role badge
      - artifact stats such as bit depth and artifact kind
      - contextual metadata
  - replaced `web/src/lib/components/ReviewPairImage.svelte`
    - owns linked pair-level display controls
    - adds a compact control strip:
      - black
      - white
      - gamma
      - reset
    - propagates the same settings to both query and candidate views
    - adds pair-level metadata above the imagery
  - updated routes to use the richer component state:
    - `/chips` now shows city / split / sensor overlays on chip imagery
    - `/pairs` enables the linked control strip for deliberate pair analysis
    - `/review` and `/runs/[runId]` keep the overlays but avoid per-card
      control clutter
  - updated `README.md` and `docs/observability-plan.md` to reflect the new
    viewer semantics
- `Results`:
  - repo validation passed:
    - `uv run ruff check .`
    - `uv run ty check`
    - `uv run --extra dev pytest -q`
    - `cd web && npm run check`
  - the observability UI now has a real analysis surface instead of only an
    image browser:
    - the pair inspector can review decoded `.j2c` artifacts under linked tone
      settings
    - contextual overlays stay visible directly on top of the imagery
    - the same component structure leaves room for future geometry overlays and
      richer annotation layers without replacing the viewer again

## Entry 28: Biome For SvelteKit Linting And Formatting

- `Implemented`: Biome-based frontend linting, formatting, and import
  organization for `web/`, alongside the existing `svelte-check` pass
- `Reason`: the repo had a good Python discipline layer with `ruff` and `ty`,
  but the SvelteKit app only had `svelte-check`. The UI needed a first-class
  formatting and linting path without pulling in a heavier ESLint + Prettier
  stack
- `Detail`:
  - added `@biomejs/biome` to `web/package.json`
  - added frontend scripts in `web/package.json`:
    - `npm run lint`
    - `npm run lint:fix`
    - `npm run format`
    - `npm run format:check`
  - added `web/biome.jsonc`
    - keeps formatting and linting scoped to repo-owned UI source and top-level
      config files
    - enables import organization
    - uses single quotes and tab indentation to match the current code style
    - disables `noUnusedImports` and `noUnusedVariables` specifically for
      `.svelte` files, because those rules were too eager against template-used
      symbols in the current Biome/Svelte path
    - excludes generated review-decoder assets by scope, rather than linting
      `web/static/kakadujs/`
  - updated:
    - `README.md`
    - `web/README.md`
    - `AGENTS.md`
    so the frontend workflow is now explicit in repo docs
- `Results`:
  - frontend validation now passes cleanly with:
    - `cd web && npm run check`
    - `cd web && npm run lint`
    - `cd web && npm run format:check`
  - the current `web/` tree was reformatted under Biome, so the repo now has a
    stable frontend formatting baseline
  - this repo now has a clean frontend equivalent to the Python discipline
    split:
    - `svelte-check` for correctness
    - `Biome` for linting and formatting

## Entry 29: One-Command Observability Dev Launcher

- `Implemented`: a repo-local launcher for bringing up the observability API
  and SvelteKit UI together
- `Reason`: the UI stack had become real enough that the default startup path
  should be one command, not a two-terminal memory exercise. The repo needed a
  single dev entrypoint that:
  - sources the GDAL/Kakadu runtime
  - starts the Python API
  - starts the Vite dev server
  - tears both down cleanly
- `Detail`:
  - added `scripts/run_obs_dev.sh`
    - sources `.local/gdal-kakadu/env.sh` when present
    - verifies `uv`, `npm`, and `web/node_modules`
    - starts:
      - `uv run --extra obs geogrok-obs-api`
      - `npm run dev -- --host 127.0.0.1 --port 5173`
    - tails both logs with `[api]` and `[ui]` prefixes
    - cleans up both child processes on `Ctrl-C`
    - prints the useful local URLs for `/chips`, `/pairs`, `/review`, and
      `/runs`
  - updated:
    - `README.md`
    - `web/README.md`
    so the new launcher is the documented fast path for local UI testing
- `Results`:
  - verified on this node with:
    - `timeout 10 ./scripts/run_obs_dev.sh`
  - both services started successfully:
    - API on `http://127.0.0.1:8787`
    - UI on `http://127.0.0.1:5173`
  - validation still passed:
    - `uv run ruff check .`
    - `uv run ty check`
    - `uv run --extra dev pytest -q`
    - `cd web && npm run check`
    - `cd web && npm run lint`
    - `cd web && npm run format:check`

## Entry 30: Headless-LAN Defaults For Observability Dev

- `Implemented`: updated the observability dev launcher to default to a
  LAN-friendly UI bind address and non-conflicting port on this node
- `Reason`: this machine is headless and the UI is intended to be opened from
  another machine on the local network. The prior launcher defaulted to
  `127.0.0.1:5173`, which was wrong for that workflow and also clashed with an
  already-used port
- `Detail`:
  - updated `scripts/run_obs_dev.sh`
    - default UI bind host is now `0.0.0.0`
    - default UI port is now `5174`
    - the script attempts to detect the first LAN IP via `hostname -I`
    - when available, it prints a browser-friendly URL using that IP rather
      than `0.0.0.0`
    - the launcher still starts the Python API locally on
      `127.0.0.1:8787`; remote browsers continue to reach it through the Vite
      proxy
    - `UI_HOST`, `UI_PORT`, and `UI_OPEN_HOST` can still override the UI side
      if needed
  - updated:
    - `README.md`
    - `web/README.md`
    to document the headless-LAN default behavior
- `Results`:
  - the launcher is now aligned with the actual deployment shape for this node:
    - UI bound on `0.0.0.0:5174`
    - LAN-friendly URL printed for remote browsers
  - the repo checks still passed after the change:
    - `uv run ruff check .`
    - `uv run ty check`
    - `uv run --extra dev pytest -q`
    - `cd web && npm run check`
    - `cd web && npm run lint`
    - `cd web && npm run format:check`

## Current Read

The most important current conclusions are:

- the repo’s data and benchmark plumbing is real and usable
- pair-based retrieval is a much more meaningful benchmark than same-raster
  coherence
- generic frozen visual encoders remain strong PAN controls
- the PAN student is still substantially weaker than the best frozen teachers
- the biggest remaining problem is not “add another model”, but “improve the
  adaptation objective and supervision geometry”
- adversarial hard-negative mining is now active, which means future work can
  optimize against a real failure mode instead of a dormant loss term

## Suggested Next Notebook Entries

When future work lands, good follow-on entries would be:

- explicit adversarial-negative evaluation metrics
- broader train-city coverage for PAN adaptation
- teacher ensembles instead of a single frozen teacher
- first privileged RGB/MS teacher-student alignment experiment
- first dense-task baseline using the same manifest and split contracts
