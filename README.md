# geogrok-training

Training and evaluation scaffolding for a PAN-first geospatial image
intelligence stack.

The current repo focus is phase 0:

- build a reliable GDAL + Kakadu runtime for NITF / JP2K access
- normalize SpaceNet metadata into stable manifests
- prepare the data layer for retrieval and dense-task baselines

`uv` manages the Python environment in this repo.

## Current pieces

- [docs/implementation-plan.md](/nvme/development/geogrok-training/docs/implementation-plan.md):
  staged implementation plan
- [docs/observability-plan.md](/nvme/development/geogrok-training/docs/observability-plan.md):
  concrete observability architecture and reference-only reuse plan
- [docs/NOTEBOOK.md](/nvme/development/geogrok-training/docs/NOTEBOOK.md):
  permanent engineering notebook of implementation steps, reasons, and results
- [scripts/build_gdal_kakadu.sh](/nvme/development/geogrok-training/scripts/build_gdal_kakadu.sh):
  repo-local GDAL + Kakadu build
- [scripts/check_gdal_kakadu.sh](/nvme/development/geogrok-training/scripts/check_gdal_kakadu.sh):
  validation for an existing GDAL + Kakadu build
- [scripts/smoke_gdal_kakadu_rasters.sh](/nvme/development/geogrok-training/scripts/smoke_gdal_kakadu_rasters.sh):
  real raster open/read smoke test
- [scripts/smoke_on_demand_chips.sh](/nvme/development/geogrok-training/scripts/smoke_on_demand_chips.sh):
  manifest-only, on-demand chip read smoke test
- [scripts/smoke_chip_benchmark.sh](/nvme/development/geogrok-training/scripts/smoke_chip_benchmark.sh):
  on-demand chip throughput benchmark smoke test
- [scripts/smoke_training_dataset.sh](/nvme/development/geogrok-training/scripts/smoke_training_dataset.sh):
  trainer-facing dataset smoke test
- [scripts/smoke_training_loop.sh](/nvme/development/geogrok-training/scripts/smoke_training_loop.sh):
  deterministic train/val dry-run loop with per-epoch metrics
- [scripts/smoke_pairs.sh](/nvme/development/geogrok-training/scripts/smoke_pairs.sh):
  real-data chip-pair mining smoke test
- [scripts/smoke_embedding_baseline.sh](/nvme/development/geogrok-training/scripts/smoke_embedding_baseline.sh):
  deterministic embedding and retrieval smoke test
- [scripts/smoke_learned_embedding.sh](/nvme/development/geogrok-training/scripts/smoke_learned_embedding.sh):
  shallow learned embedding smoke test
- [scripts/smoke_cnn_embedding.sh](/nvme/development/geogrok-training/scripts/smoke_cnn_embedding.sh):
  tiny CNN embedding smoke test
- [scripts/smoke_torch_embedding.sh](/nvme/development/geogrok-training/scripts/smoke_torch_embedding.sh):
  PyTorch GPU embedding smoke test
- [scripts/smoke_torch_pair_eval.sh](/nvme/development/geogrok-training/scripts/smoke_torch_pair_eval.sh):
  PyTorch GPU pair-based retrieval smoke test
- [scripts/smoke_torch_pair_holdout.sh](/nvme/development/geogrok-training/scripts/smoke_torch_pair_holdout.sh):
  PyTorch GPU held-out pair-based retrieval smoke test
- [scripts/smoke_pretrained_benchmark.sh](/nvme/development/geogrok-training/scripts/smoke_pretrained_benchmark.sh):
  frozen pretrained encoder benchmark on the held-out pair protocol
- [scripts/smoke_pan_adapt_benchmark.sh](/nvme/development/geogrok-training/scripts/smoke_pan_adapt_benchmark.sh):
  teacher-student PAN adaptation benchmark smoke test
- [scripts/smoke_chip_extraction.sh](/nvme/development/geogrok-training/scripts/smoke_chip_extraction.sh):
  optional manifest-to-chip extraction smoke test
- [src/geogrok/io/raster.py](/nvme/development/geogrok-training/src/geogrok/io/raster.py):
  GDAL-backed raster inspection, array reads, and optional chip export
- [src/geogrok/data/manifests.py](/nvme/development/geogrok-training/src/geogrok/data/manifests.py):
  manifest generation CLI
- [src/geogrok/data/runtime.py](/nvme/development/geogrok-training/src/geogrok/data/runtime.py):
  manifest-backed on-demand chip dataset
- [src/geogrok/data/benchmark.py](/nvme/development/geogrok-training/src/geogrok/data/benchmark.py):
  throughput benchmark CLI and reporting
- [src/geogrok/data/training.py](/nvme/development/geogrok-training/src/geogrok/data/training.py):
  trainer-facing dataset, collation, and training-path benchmark
- [src/geogrok/data/pairs.py](/nvme/development/geogrok-training/src/geogrok/data/pairs.py):
  chip ROI extraction and explicit pair-label mining
- [src/geogrok/training/loop.py](/nvme/development/geogrok-training/src/geogrok/training/loop.py):
  deterministic batching and per-epoch throughput metrics
- [src/geogrok/training/baseline.py](/nvme/development/geogrok-training/src/geogrok/training/baseline.py):
  framework-light dry-run training runner
- [src/geogrok/retrieval/baseline.py](/nvme/development/geogrok-training/src/geogrok/retrieval/baseline.py):
  deterministic PAN embedding and retrieval baseline
- [src/geogrok/retrieval/learned.py](/nvme/development/geogrok-training/src/geogrok/retrieval/learned.py):
  shallow contrastive projection baseline over PAN features
- [src/geogrok/retrieval/cnn.py](/nvme/development/geogrok-training/src/geogrok/retrieval/cnn.py):
  tiny learned PAN image encoder baseline
- [src/geogrok/retrieval/torch_encoder.py](/nvme/development/geogrok-training/src/geogrok/retrieval/torch_encoder.py):
  GPU-capable PyTorch PAN retrieval encoder baseline
- [src/geogrok/retrieval/pretrained_benchmark.py](/nvme/development/geogrok-training/src/geogrok/retrieval/pretrained_benchmark.py):
  frozen pretrained encoder benchmark runner
- [src/geogrok/retrieval/pan_adapt_benchmark.py](/nvme/development/geogrok-training/src/geogrok/retrieval/pan_adapt_benchmark.py):
  teacher-student PAN adaptation benchmark runner
- [src/geogrok/obs/run_index.py](/nvme/development/geogrok-training/src/geogrok/obs/run_index.py):
  observability run indexing over `artifacts/runs/`
- [src/geogrok/obs/api.py](/nvme/development/geogrok-training/src/geogrok/obs/api.py):
  observability API with live chip, pair, run, and failure-review endpoints
- [src/geogrok/obs/review_tables.py](/nvme/development/geogrok-training/src/geogrok/obs/review_tables.py):
  failure and disagreement queue derivation from saved benchmark embeddings and
  pair labels
- [src/geogrok/obs/annotations.py](/nvme/development/geogrok-training/src/geogrok/obs/annotations.py):
  SQLite-backed pair review state for observability surfaces
- [src/geogrok/data/chips.py](/nvme/development/geogrok-training/src/geogrok/data/chips.py):
  optional chip extraction CLI
- [src/geogrok/io/gdal_env.py](/nvme/development/geogrok-training/src/geogrok/io/gdal_env.py):
  Python-side GDAL runtime activation helper
- [web/README.md](/nvme/development/geogrok-training/web/README.md):
  SvelteKit observability UI scaffold

## Python setup

Install the root environment and dev tools:

```bash
uv sync --extra dev
```

Repo checks:

```bash
uv run ruff check .
uv run ty check
uv run --extra dev pytest -q
```

Install the optional training extra for GPU-backed retrieval work:

```bash
uv sync --extra dev --extra train
```

That extra now includes `torch`, `torchvision`, `open-clip-torch`, and
`huggingface_hub`, which the repo uses for the frozen pretrained encoder
benchmark.

Install the optional observability API extra:

```bash
uv sync --extra dev --extra obs
```

That extra provides `fastapi` and `uvicorn` for the review API scaffold.

## Frontend setup

The SvelteKit observability UI lives under [web/](/nvme/development/geogrok-training/web/).

Fastest local dev path:

```bash
./scripts/run_obs_dev.sh
```

That starts both:

- the Python observability API on `http://127.0.0.1:8787`
- the SvelteKit UI on `http://0.0.0.0:5174`

and tails both logs in one terminal until `Ctrl-C`.

On this headless node, the launcher is intended for LAN access. It prints a
LAN-friendly browser URL when it can detect one.

Install dependencies and run the frontend checks:

```bash
cd web
npm install
npm run check
npm run lint
npm run format:check
```

Frontend tooling is split intentionally:

- `svelte-check` validates Svelte template and TypeScript correctness
- `Biome` handles linting, formatting, and import organization

Optional `pykdu` integration for HTJ2K/J2C review artifacts:

```bash
./scripts/install_reference_pykdu.sh
./scripts/check_reference_pykdu.sh
```

This uses the reference submodule at
`reference/geogrok/libs/py/pykdu` as the source of truth and installs it into
this repo environment as an editable dependency. It does not copy `pykdu` into
this repo.

Optional `kakadujs` browser decoder integration:

```bash
./scripts/sync_reference_kakadujs.sh
```

This uses the reference build flow under `reference/geogrok/`, then copies
`kakadujs.js` and `kakadujs.wasm` into `web/static/kakadujs/` for the Svelte
observability UI.

Run the observability API locally:

```bash
uv run --extra obs geogrok-obs-api
```

Run the full local observability stack:

```bash
./scripts/run_obs_dev.sh
```

Pre-build review artifacts without starting the API:

```bash
uv run --extra obs geogrok-build-review-artifacts --limit 16
```

Current live review surfaces:

- `/review`
  - annotation-aware analyst worklist for unreviewed pairs, failures, and
    disagreements, with a dedicated bookmarked section
- `/chips`
  - manifest-backed chip browser that prefers cached review artifacts and falls
    back to source-rendered quicklooks
- `/pairs`
  - labeled pair inspection with annotation-aware filtering, artifact-backed
    pair review imagery, linked black/white/gamma controls for decoded `.j2c`
    views, inline review actions, and bookmark filtering
- `/runs`
  - run summary list
- `/runs/[runId]`
  - run-specific false-negative, false-positive, and pan-adapt
    teacher-student disagreement queues derived from saved embeddings and
    explicit pair labels, with inline pair review actions, review-state
    filtering, and bookmark-only slices

Review artifact runtime notes:

- review artifacts are stored under `artifacts/observability/review_artifacts/`
- current artifact generation prefers `.j2c` via `pykdu` when available
- `./scripts/install_reference_pykdu.sh` is the supported local Option A path
  for enabling `pykdu` from the reference submodule
- on this node, `pykdu` is now installed in the repo environment and the API is
  generating `.j2c` review artifacts from real PAN chips
- browser-side HTJ2K decode is enabled when `web/static/kakadujs/kakadujs.js`
  and `web/static/kakadujs/kakadujs.wasm` are present
- on this node, the browser-side decoder assets are now present and the review
  runtime reports `kakadujs_assets_available=true`
- review artifacts are for human observability only and must not be used as
  training inputs

## GDAL + Kakadu

The repo includes vendored source trees under:

- `third_party/gdal/`
- `third_party/kakadu/`

The goal is to produce a repo-local GDAL build with:

- Kakadu-backed `JP2KAK`
- `NITF`
- Python bindings importable as `from osgeo import gdal`

### Build

Run:

```bash
./scripts/build_gdal_kakadu.sh
```

Validate an existing build without rebuilding:

```bash
./scripts/check_gdal_kakadu.sh
```

Run a real raster smoke test through the built Python bindings:

```bash
./scripts/smoke_gdal_kakadu_rasters.sh
```

Optional overrides:

- `GEOTIFF_PATH=/abs/path/to/file.tif`
- `NITF_PATH=/abs/path/to/file.ntf`
- `RASTER_SEARCH_ROOTS=/extra/search/root1:/extra/search/root2`

This installs into:

- prefix: `.local/gdal-kakadu/`
- build dir: `.build/gdal-kakadu/`
- Python build env: `.build/gdal-kakadu-python/`

The script handles:

- Kakadu auxiliary library generation when missing
- SWIG discovery from the local environment
- GDAL Python binding build
- runtime patching so the installed Python extensions load the local
  Kakadu-enabled `libgdal.so`
- generation of `.local/gdal-kakadu/env.sh`

### Activate

Source the generated environment before using the repo-local GDAL runtime:

```bash
source .local/gdal-kakadu/env.sh
```

This sets:

- `PATH`
- `GDAL_DATA`
- `PROJ_DATA`
- `PROJ_LIB`
- `LD_LIBRARY_PATH`
- `PYTHONPATH`

### Verified working state

The current validated commands are:

```bash
source .local/gdal-kakadu/env.sh
.local/gdal-kakadu/bin/gdalinfo --formats | rg 'JP2KAK|NITF'
.build/gdal-kakadu-python/bin/python - <<'PY'
from osgeo import gdal
print(gdal.VersionInfo("--version"))
print("JP2KAK", bool(gdal.GetDriverByName("JP2KAK")))
print("NITF", bool(gdal.GetDriverByName("NITF")))
PY
```

The same validation is wrapped in:

```bash
./scripts/check_gdal_kakadu.sh
```

The end-to-end raster open/read smoke test is:

```bash
./scripts/smoke_gdal_kakadu_rasters.sh
```

Expected output includes:

- `JP2KAK -raster,vector-`
- `NITF -raster-`
- `JP2KAK True`
- `NITF True`

### Notes

- The Python bindings are installed under
  `.local/gdal-kakadu/lib/python3.13/site-packages/`.
- The script currently falls back to the host Pixi `proj.db` when the local
  install prefix does not contain a populated PROJ database.
- The build uses `patchelf` so the installed `osgeo` extension modules resolve
  the local `libgdal.so` rather than any unrelated GDAL already present on the
  host.
- The raster smoke test prefers real files under `datasets/spacenet.ai/` when
  present. It searches the bucket-mirror tree directly, including uppercase
  `.NTF` and `.TIF` names. If the local dataset mirror is empty, it falls back to small
  GeoTIFF and NITF fixtures from `third_party/gdal/autotest/`. The default NITF
  fixture is JP2-compressed so the smoke read exercises the Kakadu-backed path.

### Common failure modes

`swig` is not on `PATH`

- Symptom: CMake fails with a Python binding error complaining that SWIG is
  missing.
- Fix: provide a usable SWIG install. The build script can discover a local
  SWIG binary and its library directory, but if that fails you can override it:

```bash
SWIG_EXECUTABLE=/path/to/swig SWIG_DIR=/path/to/share/swig/4.x.y \
  ./scripts/build_gdal_kakadu.sh
```

SWIG runs but cannot find `swig.swg` or `python.swg`

- Symptom: wrapper generation fails with messages like `Unable to find
  'swig.swg'`.
- Cause: the SWIG binary is present, but its library tree is not being found.
- Fix: ensure `SWIG_DIR` points at the directory containing `swig.swg`. The
  script exports `SWIG_LIB` for this reason.

Python imports `osgeo.gdal` but `JP2KAK` is missing

- Symptom: `gdalinfo --formats` shows `JP2KAK`, but Python reports
  `gdal.GetDriverByName("JP2KAK") is None`.
- Cause: the Python extension modules are loading a different `libgdal.so`
  already present on the host.
- Fix: source `.local/gdal-kakadu/env.sh` before using the bindings. The build
  script also rewrites the Python extension `RPATH` to prefer the local GDAL
  install.

`gdalinfo` or Python fails with missing shared libraries

- Symptom: runtime errors mentioning libraries like `libdeflate.so`,
  `libtiff.so`, or similar.
- Cause: the local GDAL build depends on host libraries that are not on the
  runtime loader path.
- Fix: source `.local/gdal-kakadu/env.sh` so `LD_LIBRARY_PATH` includes both
  the local GDAL install and the host Python environment library directory.

PROJ errors such as `proj_create: no database context specified`

- Symptom: `gdalinfo` emits PROJ errors about missing CRS parsing support.
- Cause: `PROJ_DATA` points at a directory without a valid `proj.db`.
- Fix: source `.local/gdal-kakadu/env.sh`. The script currently points
  `PROJ_DATA` and `PROJ_LIB` at the host Pixi `share/proj` when the local
  prefix does not contain a populated PROJ database.

Kakadu core exists but auxiliary library is missing

- Symptom: GDAL configure or link fails even though `libkdu_vs85R.so` is
  present.
- Cause: GDAL needs both Kakadu core and auxiliary libraries.
- Fix: rerun `./scripts/build_gdal_kakadu.sh`. It will build the Kakadu managed
  auxiliary library if `libkdu_a*.so` is missing.

## Manifest generation

Generate bootstrap manifests from the current SpaceNet metadata:

```bash
uv run geogrok-make-manifests
```

Build harder retrieval pairs from chip ground ROIs:

```bash
source .local/gdal-kakadu/env.sh
uv run geogrok-make-pairs \
  --chips-path datasets/manifests/spacenet/chips.parquet \
  --scenes-path datasets/manifests/spacenet/scenes.parquet \
  --output-root datasets/pairs/spacenet \
  --modality PAN
```

Smoke test on a small slice:

```bash
uv run geogrok-make-manifests --limit-assets 8
```

Default outputs go under:

- `datasets/manifests/spacenet/`

Current artifacts:

- `assets.parquet`
- `scenes.parquet`
- `chips.parquet`
- `summary.json`

## On-Demand Chips

Source the GDAL runtime first so the Kakadu-enabled Python bindings load with
the correct shared-library path:

```bash
source .local/gdal-kakadu/env.sh
```

Read a small local subset directly from the manifest without writing chip files:

```bash
./scripts/smoke_on_demand_chips.sh
```

This path uses [runtime.py](/nvme/development/geogrok-training/src/geogrok/data/runtime.py)
to:

- filter `chips.parquet` down to local rows
- read windows from the original source rasters on demand
- return `uint16` arrays in `(C, H, W)` layout for training code

The current smoke path reads real mirrored WV3 PAN NITF chips and reports
shapes and value ranges without materializing intermediate TIFFs.

## Throughput Benchmarking

Performance measurement is intended to be part of the normal workflow.

Benchmark on-demand chip reads:

```bash
source .local/gdal-kakadu/env.sh
uv run geogrok-benchmark-chips \
  --chips-path datasets/manifests/spacenet/chips.parquet \
  --split train \
  --modality PAN \
  --limit 32 \
  --repeat 2 \
  --warmup 2
```

Reported metrics include:

- `samples_per_second`
- `megapixels_per_second`
- `mebibytes_per_second`
- `latency_ms_mean`
- `latency_ms_p50`
- `latency_ms_p95`
- `latency_ms_max`
- `unique_source_files`

The benchmark also writes a JSON report. Default path:

- `artifacts/benchmarks/chip-read-benchmark.json`

Smoke test on real mirrored data:

```bash
./scripts/smoke_chip_benchmark.sh
```

Trainer-path benchmark with normalization enabled:

```bash
source .local/gdal-kakadu/env.sh
uv run geogrok-benchmark-training \
  --chips-path datasets/manifests/spacenet/chips.parquet \
  --split train \
  --modality PAN \
  --limit 32 \
  --repeat 2 \
  --warmup 2 \
  --output-dtype float32 \
  --clip-min 0 \
  --clip-max 2047 \
  --scale-max 2047
```

This benchmark reports read time and transform time separately so you can see
where throughput drops as preprocessing gets more expensive.

## Training Dataset

The default trainer-facing wrapper is
[training.py](/nvme/development/geogrok-training/src/geogrok/data/training.py).
It is framework-light for now and returns:

- normalized `(C, H, W)` `numpy` arrays
- original chip metadata
- per-sample timing with `read_ms`, `transform_ms`, and `total_ms`

Smoke test on real mirrored data:

```bash
./scripts/smoke_training_dataset.sh
```

## Training Loop Scaffold

The repo now includes a deterministic dry-run training loop that:

- batches on-demand samples
- shuffles train epochs reproducibly from `seed + epoch`
- keeps validation deterministic
- logs per-epoch throughput and latency metrics to disk

Run it with:

```bash
source .local/gdal-kakadu/env.sh
uv run geogrok-run-baseline \
  --chips-path datasets/manifests/spacenet/chips.parquet \
  --epochs 2 \
  --batch-size 8 \
  --train-limit 32 \
  --val-limit 16 \
  --output-dtype float32 \
  --clip-min 0 \
  --clip-max 2047 \
  --scale-max 2047
```

Outputs:

- `artifacts/runs/training-dryrun/metrics.jsonl`
- `artifacts/runs/training-dryrun/summary.json`

Smoke test on real mirrored data:

```bash
./scripts/smoke_training_loop.sh
```

## Embedding Baseline

The first retrieval baseline is intentionally simple and deterministic. It:

- reads normalized PAN chips from the trainer-facing dataset
- computes a handcrafted embedding from coarse pooled intensity, gradient, and
  profile features
- evaluates nearest-neighbor retrieval by `scene_id` or `capture_id`
- supports separate query and gallery split definitions
- balances the sampled set across scenes and enforces spatially separated positives
- logs read, transform, embedding, and total latency

Run it with:

```bash
source .local/gdal-kakadu/env.sh
uv run geogrok-run-embedding-baseline \
  --chips-path datasets/manifests/spacenet/chips.parquet \
  --query-split val \
  --query-split test \
  --gallery-split val \
  --gallery-split test \
  --modality PAN \
  --limit 128 \
  --max-chips-per-scene 4 \
  --min-chips-per-scene 2 \
  --output-dtype float32 \
  --clip-min 0 \
  --clip-max 2047 \
  --scale-max 2047 \
  --positive-key scene_id \
  --min-positive-center-distance 1024
```

Outputs:

- `artifacts/runs/embedding-baseline/embeddings.npy`
- `artifacts/runs/embedding-baseline/index.parquet`
- `artifacts/runs/embedding-baseline/benchmark.json`
- `artifacts/runs/embedding-baseline/retrieval.json`

Smoke test on real mirrored data:

```bash
./scripts/smoke_embedding_baseline.sh
```

## Learned Embedding Baseline

The next-step learned baseline keeps the same retrieval protocol but replaces
the handcrafted embedding with:

- fixed PAN feature extraction from the deterministic baseline
- a shallow linear projection
- contrastive training on same-scene positive pairs from the train split
- evaluation on the harder `val/test` query-gallery protocol

Run it with:

```bash
source .local/gdal-kakadu/env.sh
uv run geogrok-run-learned-embedding \
  --chips-path datasets/manifests/spacenet/chips.parquet \
  --train-split train \
  --query-split val \
  --query-split test \
  --gallery-split val \
  --gallery-split test \
  --modality PAN \
  --train-limit 128 \
  --eval-limit 128 \
  --max-chips-per-scene 4 \
  --min-chips-per-scene 2 \
  --output-dtype float32 \
  --clip-min 0 \
  --clip-max 2047 \
  --scale-max 2047 \
  --positive-key scene_id \
  --min-positive-center-distance 1024 \
  --embedding-dim 64 \
  --epochs 20 \
  --steps-per-epoch 16 \
  --pairs-per-batch 16
```

Outputs:

- `artifacts/runs/learned-embedding-baseline/model.npz`
- `artifacts/runs/learned-embedding-baseline/train_features.npy`
- `artifacts/runs/learned-embedding-baseline/eval_embeddings.npy`
- `artifacts/runs/learned-embedding-baseline/feature_train_benchmark.json`
- `artifacts/runs/learned-embedding-baseline/feature_eval_benchmark.json`
- `artifacts/runs/learned-embedding-baseline/training.json`
- `artifacts/runs/learned-embedding-baseline/retrieval.json`

Smoke test on real mirrored data:

```bash
./scripts/smoke_learned_embedding.sh
```

Current status: this shallow learned projection trains correctly and logs useful
throughput numbers, but on the current harder retrieval protocol it does not yet
beat the deterministic baseline. That makes it a useful control, not yet a
replacement.

## Pair Mining

The repo now has an explicit pair-mining stage for harder retrieval protocols.
Instead of treating every chip from the same raster scene as positive, it:

- geolocates chip windows into real-world ground ROIs using raster georeferencing
  or RPC metadata
- works in local metric coordinates per city
- mines overlapping cross-asset positives
- mines spatially nearby non-overlap hard negatives
- writes reusable pair labels to parquet

Run it with:

```bash
source .local/gdal-kakadu/env.sh
uv run geogrok-make-pairs \
  --chips-path datasets/manifests/spacenet/chips.parquet \
  --scenes-path datasets/manifests/spacenet/scenes.parquet \
  --output-root datasets/pairs/spacenet \
  --modality PAN \
  --positive-overlap-fraction 0.5 \
  --weak-overlap-fraction 0.2 \
  --hard-negative-radius-m 800
```

Outputs:

- `datasets/pairs/spacenet/chip_rois.parquet`
- `datasets/pairs/spacenet/asset_pairs.parquet`
- `datasets/pairs/spacenet/pairs.parquet`
- `datasets/pairs/spacenet/summary.json`

Smoke test on real mirrored data:

```bash
./scripts/smoke_pairs.sh
```

Current status: the smoke protocol on Jacksonville PAN scenes mined explicit
`positive_exact`, `positive_weak`, and `negative_hard` rows from real chip ROIs.
The latest run produced 240 chip ROIs, 4 overlapping asset pairs, and 146
labeled chip pairs: 8 exact positives, 14 weak positives, and 124 hard negatives.

## CNN Embedding Baseline

The first real learned image encoder keeps the same harder retrieval protocol
but learns directly from downsampled PAN chips instead of fixed handcrafted
features. The current implementation is intentionally small:

- mean-downsample each PAN chip to `image_size x image_size`
- run a 2-layer NumPy CNN with global average pooling
- train the embedding head contrastively on same-scene positive pairs
- preserve the same retrieval and throughput reporting used by the earlier
  baselines

Run it with:

```bash
source .local/gdal-kakadu/env.sh
uv run geogrok-run-cnn-embedding \
  --chips-path datasets/manifests/spacenet/chips.parquet \
  --train-split train \
  --query-split val \
  --query-split test \
  --gallery-split val \
  --gallery-split test \
  --modality PAN \
  --train-limit 128 \
  --eval-limit 128 \
  --max-chips-per-scene 4 \
  --min-chips-per-scene 2 \
  --output-dtype float32 \
  --clip-min 0 \
  --clip-max 2047 \
  --scale-max 2047 \
  --positive-key scene_id \
  --min-positive-center-distance 1024 \
  --image-size 64 \
  --conv1-channels 8 \
  --conv2-channels 16 \
  --embedding-dim 64 \
  --epochs 20 \
  --steps-per-epoch 16 \
  --pairs-per-batch 16
```

Outputs:

- `artifacts/runs/cnn-embedding-baseline/model.npz`
- `artifacts/runs/cnn-embedding-baseline/train_images.npy`
- `artifacts/runs/cnn-embedding-baseline/eval_embeddings.npy`
- `artifacts/runs/cnn-embedding-baseline/train_index.parquet`
- `artifacts/runs/cnn-embedding-baseline/eval_index.parquet`
- `artifacts/runs/cnn-embedding-baseline/train_preprocess_benchmark.json`
- `artifacts/runs/cnn-embedding-baseline/eval_preprocess_benchmark.json`
- `artifacts/runs/cnn-embedding-baseline/training.json`
- `artifacts/runs/cnn-embedding-baseline/embedding.json`
- `artifacts/runs/cnn-embedding-baseline/retrieval.json`

Smoke test on real mirrored data:

```bash
./scripts/smoke_cnn_embedding.sh
```

Current status: the tiny CNN baseline is now wired end to end, but on the
current harder retrieval protocol it still underperforms the deterministic
baseline. The latest smoke run produced `R@1=0.000`, `R@5=0.062`, `R@10=0.156`,
and `MRR=0.061`, so this is a real learned baseline, not yet a competitive one.

## Torch Embedding Baseline

The current strongest learned retrieval baseline in the repo uses PyTorch and
the local GPU. It keeps the same harder retrieval protocol and adds:

- a small convnet encoder trained directly on downsampled PAN chips
- CUDA execution with optional mixed precision
- the same manifest sampling, split logic, and retrieval metrics as the earlier
  baselines
- explicit training and embedding throughput plus GPU memory reporting

Run it with:

```bash
source .local/gdal-kakadu/env.sh
uv run geogrok-run-torch-embedding \
  --chips-path datasets/manifests/spacenet/chips.parquet \
  --train-split train \
  --query-split val \
  --query-split test \
  --gallery-split val \
  --gallery-split test \
  --modality PAN \
  --train-limit 256 \
  --eval-limit 128 \
  --max-chips-per-scene 4 \
  --min-chips-per-scene 2 \
  --output-dtype float32 \
  --clip-min 0 \
  --clip-max 2047 \
  --scale-max 2047 \
  --positive-key scene_id \
  --min-positive-center-distance 1024 \
  --image-size 128 \
  --base-channels 48 \
  --embedding-dim 128 \
  --epochs 24 \
  --steps-per-epoch 48 \
  --pairs-per-batch 32 \
  --eval-batch-size 64 \
  --device auto \
  --amp
```

Outputs:

- `artifacts/runs/torch-embedding-baseline/model.pt`
- `artifacts/runs/torch-embedding-baseline/train_images.npy`
- `artifacts/runs/torch-embedding-baseline/eval_embeddings.npy`
- `artifacts/runs/torch-embedding-baseline/train_index.parquet`
- `artifacts/runs/torch-embedding-baseline/eval_index.parquet`
- `artifacts/runs/torch-embedding-baseline/train_preprocess_benchmark.json`
- `artifacts/runs/torch-embedding-baseline/eval_preprocess_benchmark.json`
- `artifacts/runs/torch-embedding-baseline/training.json`
- `artifacts/runs/torch-embedding-baseline/embedding.json`
- `artifacts/runs/torch-embedding-baseline/retrieval.json`

Smoke test on real mirrored data:

```bash
./scripts/smoke_torch_embedding.sh
```

Current status: this is the first learned encoder in the repo that now beats
the deterministic baseline on the current harder protocol. The latest smoke run
on the RTX 3090 produced `R@1=0.135`, `R@5=0.229`, `R@10=0.302`, and
`MRR=0.195`, with training throughput around `12.4k images/s` and peak GPU
memory under `300 MiB`.

## Torch Pair Evaluation

The torch runner can now evaluate against explicit chip-pair labels from
`pairs.parquet` instead of using `scene_id` as an implicit relevance signal.
This is a materially harder protocol because it measures:

- `positive_exact`: strong cross-asset spatial overlap
- `positive_weak`: moderate cross-asset spatial overlap
- `negative_hard`: nearby but non-overlapping chip pairs

Run it with:

```bash
source .local/gdal-kakadu/env.sh
uv run geogrok-run-torch-embedding \
  --chips-path datasets/manifests/spacenet/chips.parquet \
  --pairs-path datasets/pairs/spacenet/pairs.parquet \
  --train-pairs-path datasets/pairs/spacenet/pairs.parquet \
  --train-split train \
  --query-split train \
  --gallery-split train \
  --modality PAN \
  --train-limit 256 \
  --image-size 128 \
  --base-channels 48 \
  --embedding-dim 128 \
  --epochs 24 \
  --steps-per-epoch 48 \
  --pairs-per-batch 32 \
  --eval-batch-size 64 \
  --device auto \
  --amp
```

When `--pairs-path` is set, `retrieval.json` switches to pair-based metrics:

- `exact_recall_at_1/5/10`
- `any_recall_at_1/5/10`
- `exact_mean_reciprocal_rank`
- `any_mean_reciprocal_rank`
- `hard_negative_at_1_rate`
- `hard_negative_in_top_5_rate`

When `--train-pairs-path` is also set, the contrastive trainer stops building
positives from `scene_id` or `capture_id` groups and instead samples
`positive_exact` / `positive_weak` rows directly from `pairs.parquet`. This is
the right mode when you want the training objective to match the overlap-based
evaluation protocol.

The mined `pairs.parquet` rows now also carry `query_split`,
`candidate_split`, `query_sensor`, `candidate_sensor`, `query_acq_time`, and
`candidate_acq_time`, and `summary.json` includes split-aware pair counts. That
makes it easier to stratify retrieval performance by held-out split without
re-reading the chip manifest.

Smoke test on real mirrored data:

```bash
./scripts/smoke_torch_pair_eval.sh
```

Current status: on the small Jacksonville smoke pair set, the latest torch run
on the RTX 3090 now trains on explicit positive pairs as well as evaluating on
them. That smoke run produced `exact_R@1=0.625`, `exact_R@5=1.000`,
`exact_R@10=1.000`, `any_R@1=0.650`, `any_R@5=0.900`, `any_R@10=0.900`,
`any_MRR=0.751`, and `hardneg@1=0.000`, with `train_sampling=explicit_pairs`
over 11 deduplicated positive training pairs. Treat that as a wiring check, not
as a benchmark claim: the smoke protocol is still `train -> train` on a tiny
pair set, so it is intentionally optimistic.

Held-out smoke test on real mirrored data:

```bash
./scripts/smoke_torch_pair_holdout.sh
```

This script mines a smaller split-aware pair set across Jacksonville
(`train`), Omaha (`val`), and UCSD (`test`), then trains on `train` pairs and
evaluates on `val/test`. The default smoke settings are intentionally capped to
keep runtime reasonable:

- `MAX_CHIPS_PER_ASSET=8`
- `TRAIN_LIMIT=512`

The latest held-out smoke run on the RTX 3090 produced `exact_R@1=0.005`,
`exact_R@5=0.055`, `exact_R@10=0.070`, `any_R@1=0.012`, `any_R@5=0.059`,
`any_R@10=0.076`, `any_MRR=0.042`, and `hardneg@1=0.025`, with 3,505 mined pair
rows and 856 held-out eval chips. That is the more honest baseline to track
for generalization. The large gap between this and the Jacksonville train/train
smoke is expected and useful: it tells you the model is still leaning heavily
on local structure rather than learning a strong cross-region retrieval space.

## Pretrained Benchmark

The repo can now benchmark a few frozen generic pretrained encoders on the same
held-out PAN chip set and score them against the same `pairs.parquet`
contract. This is intended as a control benchmark before moving to remote
sensing foundation models or teacher-student alignment.

Run it with:

```bash
source .local/gdal-kakadu/env.sh
uv sync --extra dev --extra train
uv run geogrok-benchmark-pretrained \
  --chips-path datasets/manifests/spacenet/chips.parquet \
  --pairs-path datasets/pairs/spacenet/pairs.parquet \
  --query-split val \
  --query-split test \
  --gallery-split val \
  --gallery-split test \
  --modality PAN \
  --eval-limit 512 \
  --batch-size 64 \
  --model resnet18 \
  --model resnet50 \
  --model vit_b_16 \
  --device auto \
  --amp
```

The current frozen controls include both generic and remote-sensing models:

- `resnet50`
- `resnet101`
- `resnet152`
- `remoteclip_rn50`
- `georsclip_vit_b32_ret2`
- `dinov2_vitb14`
- `dinov3_vitb16`

For the CLIP-family remote-sensing models, the repo uses official checkpoints
from Hugging Face:

- `chendelong/RemoteCLIP:RemoteCLIP-RN50.pt`
- `Zilun/GeoRSCLIP:ckpt/RS5M_ViT-B-32_RET-2.pt`

PAN chips are repeated to 3 channels and resized to `224 x 224`. The
ImageNet-style models use ImageNet normalization; the CLIP-family models use
CLIP normalization. Outputs are written under one run root with per-model
artifacts plus `summary.json` and `summary.parquet`.

Smoke test on real mirrored data:

```bash
./scripts/smoke_pretrained_benchmark.sh
```

Latest held-out smoke result on the RTX 3090 over 505 eval chips:

- `resnet152`: `exact_R@10=0.624`, `any_R@10=0.542`, `any_MRR=0.356`
- `resnet101`: `exact_R@10=0.611`, `any_R@10=0.507`, `any_MRR=0.354`
- `resnet50`: `exact_R@10=0.604`, `any_R@10=0.535`, `any_MRR=0.369`
- `dinov3_vitb16`: `exact_R@10=0.591`, `any_R@10=0.524`, `any_MRR=0.375`
- `remoteclip_rn50`: `exact_R@10=0.564`, `any_R@10=0.490`, `any_MRR=0.281`
- `dinov2_vitb14`: `exact_R@10=0.523`, `any_R@10=0.472`, `any_MRR=0.324`
- `georsclip_vit_b32_ret2`: `exact_R@10=0.523`, `any_R@10=0.465`, `any_MRR=0.308`

That result is more nuanced now. On this first PAN repeated-to-RGB setup:

- deeper ResNets help a bit on `R@10`, with `resnet152` now the best top-k model
- `dinov3_vitb16` is the strongest transformer by `MRR`
- the first two remote-sensing CLIP checkpoints still trail the best generic controls

So the next model work should still not assume an RS-pretrained model is
automatically better here; the modality gap is still dominating. The practical
reference set to beat is now `resnet152` for top-k retrieval and
`dinov3_vitb16` for early-rank retrieval quality.

## PAN Adaptation Benchmark

The repo can now benchmark a PAN-only student against a frozen teacher
embedding space on the same held-out pair protocol.

The current benchmark setup is:

- teacher: frozen pretrained encoder from `geogrok-benchmark-pretrained`
- student: PAN-only CNN over normalized `128 x 128` chips, with
  `baseline_cnn` and stronger `residual_cnn` options
- training objective:
  - explicit `positive_exact` / `positive_weak` pairs from `pairs.parquet`
  - cosine alignment to the teacher embedding
  - teacher similarity-structure matching within each batch
  - multi-view PAN consistency across two augmented student views
  - weighted exact-vs-weak positive pair matching
  - dynamic hard-negative separation against `negative_hard` pairs
  - contrastive `NT-Xent` on the fused student embedding
- evaluation: held-out `val/test` pair retrieval with the same metrics as the
  frozen benchmark

Run it with:

```bash
source .local/gdal-kakadu/env.sh
uv sync --extra dev --extra train
uv run geogrok-benchmark-pan-adapt \
  --chips-path datasets/manifests/spacenet/chips.parquet \
  --pairs-path datasets/pairs/spacenet/pairs.parquet \
  --teacher-model dinov3_vitb16 \
  --train-split train \
  --query-split val \
  --query-split test \
  --gallery-split val \
  --gallery-split test \
  --modality PAN \
  --train-limit 512 \
  --eval-limit 512 \
  --teacher-batch-size 16 \
  --student-image-size 128 \
  --student-arch residual_cnn \
  --student-base-channels 64 \
  --epochs 24 \
  --steps-per-epoch 48 \
  --pairs-per-batch 32 \
  --eval-batch-size 64 \
  --contrastive-weight 1.0 \
  --alignment-weight 1.0 \
  --structure-weight 0.5 \
  --view-consistency-weight 0.25 \
  --positive-pair-weight 0.5 \
  --hard-negative-weight 0.25 \
  --positive-exact-weight 2.0 \
  --positive-weak-weight 1.0 \
  --hard-negative-max-similarity 0.2 \
  --hard-negative-gap-scale 0.5 \
  --hard-negative-min-similarity -0.25 \
  --augmentation-min-crop-scale 0.7 \
  --augmentation-noise-std 0.02 \
  --augmentation-gamma-jitter 0.15 \
  --augmentation-blur-probability 0.2 \
  --device auto \
  --amp
```

Smoke test:

```bash
./scripts/smoke_pan_adapt_benchmark.sh
```

The smoke script now defaults to `dinov3_vitb16` plus the stronger
`residual_cnn` student.

Latest held-out smoke result with `dinov3_vitb16` as teacher and `residual_cnn`
student:

- teacher: `exact_R@10=0.591`, `any_R@10=0.524`, `any_MRR=0.375`
- student: `exact_R@10=0.121`, `any_R@10=0.125`, `any_MRR=0.065`
  Earlier dynamic-margin run without adversarial mining.

Latest held-out smoke result after adversarial same-city non-overlap mining:

- teacher: `exact_R@10=0.591`, `any_R@10=0.524`, `any_MRR=0.375`
- student: `exact_R@10=0.087`, `any_R@10=0.122`, `any_MRR=0.056`
- mined hard negatives: `30` pairs with `teacher_sim_mean=0.9113`,
  `teacher_sim_p95=0.9486`

Apples-to-apples check with the same `residual_cnn` student and `resnet152`
teacher:

- teacher: `exact_R@10=0.624`, `any_R@10=0.542`, `any_MRR=0.356`
- student: `exact_R@10=0.074`, `any_R@10=0.083`, `any_MRR=0.046`

The student is still far below the teacher. The updated recipe is helpful on
the `dinov3` teacher path, but not universally better. It shows:

- the PAN adaptation path is now implemented end to end
- weighted positives plus multi-view/structure distillation can move the
  student in the right direction
- the effect is teacher-dependent, so the adaptation objective is still not
  stable
- the hard-negative path is now active on mined adversarial negatives
  (`hardneg_mean=0.2234`), which means the training signal is finally aligned
  with confusable same-city non-overlap pairs
- the student is still recall-poor, especially on weak positives and early rank
- future work should focus on a better PAN adaptation recipe, not on adding
  more frozen controls

## Optional Chip Extraction

If you need a materialized chip corpus for benchmarking or later cache
experiments, the extractor remains available:

```bash
uv run geogrok-extract-chips \
  --chips-path datasets/manifests/spacenet/chips.parquet \
  --output-root datasets/chips/spacenet \
  --modality PAN \
  --limit 16
```

Optional smoke test:

```bash
./scripts/smoke_chip_extraction.sh
```

## Next steps

The next intended repo work after phase 0 is:

1. mirror selected imagery into `datasets/spacenet.ai/`
2. extend manifests with view-angle and off-nadir details
3. use `pairs.parquet` as the default retrieval benchmark contract
4. add the first dense-task baseline on on-demand PAN chips
