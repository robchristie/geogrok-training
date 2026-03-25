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
- [scripts/smoke_embedding_baseline.sh](/nvme/development/geogrok-training/scripts/smoke_embedding_baseline.sh):
  deterministic embedding and retrieval smoke test
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
- [src/geogrok/training/loop.py](/nvme/development/geogrok-training/src/geogrok/training/loop.py):
  deterministic batching and per-epoch throughput metrics
- [src/geogrok/training/baseline.py](/nvme/development/geogrok-training/src/geogrok/training/baseline.py):
  framework-light dry-run training runner
- [src/geogrok/retrieval/baseline.py](/nvme/development/geogrok-training/src/geogrok/retrieval/baseline.py):
  deterministic PAN embedding and retrieval baseline
- [src/geogrok/data/chips.py](/nvme/development/geogrok-training/src/geogrok/data/chips.py):
  optional chip extraction CLI
- [src/geogrok/io/gdal_env.py](/nvme/development/geogrok-training/src/geogrok/io/gdal_env.py):
  Python-side GDAL runtime activation helper

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
- logs read, transform, embedding, and total latency

Run it with:

```bash
source .local/gdal-kakadu/env.sh
uv run geogrok-run-embedding-baseline \
  --chips-path datasets/manifests/spacenet/chips.parquet \
  --split train \
  --modality PAN \
  --limit 64 \
  --output-dtype float32 \
  --clip-min 0 \
  --clip-max 2047 \
  --scale-max 2047 \
  --positive-key scene_id
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
3. add the first retrieval baseline on on-demand PAN chips
4. add the first dense-task baseline on on-demand PAN chips
