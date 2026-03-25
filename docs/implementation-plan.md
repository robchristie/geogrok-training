# Implementation Plan

## Scope

This repo is being bootstrapped around a PAN-first image intelligence stack for
high-resolution, non-orthorectified WV2/WV3 imagery. The first concrete goals
are:

- make NITF and GeoTIFF I/O reliable with a repo-local GDAL + Kakadu build
- normalize SpaceNet and related imagery into reproducible manifests
- create stable train, validation, and test splits before model work starts
- build enough data plumbing to support retrieval and dense-task baselines

The current repo already contains useful SpaceNet bucket exploration tooling
under `tools/spacenet/`. This plan turns that exploration into repeatable repo
artifacts.

## Phase 0

### 0.1 GDAL and Kakadu runtime

Deliverable: a repeatable build script at `scripts/build_gdal_kakadu.sh`.

Requirements:

- build GDAL from `third_party/gdal/`
- use Kakadu from `third_party/kakadu/`
- install GDAL applications, libraries, and Python bindings into a repo-local
  prefix
- verify that `JP2KAK` is registered and importable through `osgeo.gdal`

Default install target:

- prefix: `.local/gdal-kakadu/`
- build dir: `.build/gdal-kakadu/`

Runtime contract:

- all future repo raster readers should source the generated
  `.local/gdal-kakadu/env.sh` or use `geogrok.io.gdal_env`
- stock PyPI GDAL wheels are acceptable for experiments but not for the primary
  NITF / JP2K path

### 0.2 Dataset manifests

Deliverable: a manifest pipeline exposed as `geogrok-make-manifests`.

Current outputs:

- `assets.parquet`: normalized rows for every metadata record
- `scenes.parquet`: one preferred asset per `scene_id + modality`
- `chips.parquet`: a chip grid over selected scene assets, initially PAN-first
- `summary.json`: quick counts by split, city, and modality

Default inputs:

- `tools/spacenet/artifacts/s3-spacenet-dataset-images-ds`
- fallback: `tools/spacenet/artifacts/s3-spacenet-dataset-images.parquet`

Default outputs:

- `datasets/manifests/spacenet/`

Default bootstrap split:

- `train`: everything except held-out cities
- `val`: `Omaha`
- `test`: `UCSD`

This is only a bootstrap split, not the final benchmark definition. It is
useful immediately because it avoids random row-wise leakage and gives the repo
stable artifacts for early evaluation.

Example:

```bash
uv run geogrok-make-manifests \
  --metadata-path tools/spacenet/artifacts/s3-spacenet-dataset-images-ds \
  --output-root datasets/manifests/spacenet \
  --download-root datasets/spacenet.ai \
  --chip-size 1024 \
  --chip-stride 1024
```

### 0.3 Raw data layout

Expected directory layout:

```text
datasets/
  spacenet.ai/
    <mirrored S3 keys>
  manifests/
    spacenet/
      assets.parquet
      scenes.parquet
      chips.parquet
      summary.json
```

Manifest responsibilities:

- keep raw storage separate from normalized tabular metadata
- make local-vs-remote availability explicit
- preserve source `key`, `scene_id`, geometry bytes, and acquisition timestamps
- expose derived fields needed by the first training loop:
  `city`, `split`, `sensor`, `modality`, `capture_id`, `asset_preference_rank`
- support repeatable performance measurement against real local assets

## Phase 1

### 1.1 Retrieval baseline

Initial benchmark:

- scene retrieval on PAN chips
- chip-to-chip retrieval
- later text-to-chip retrieval with teacher embeddings

Required inputs from Phase 0:

- chip manifest
- stable split assignment
- local path resolution
- GDAL-backed on-demand chip reads from real local assets

### 1.2 Dense-task baseline

Initial benchmark:

- building- and road-centric segmentation or classification on PAN chips
- PEFT before full fine-tuning
- metrics stratified by held-out geography and view regime

The manifest layer must stay image-space native. No orthorectification should be
introduced into the primary training path unless a task explicitly needs it.
The default training path should remain manifest-only and on-demand. Materialized
chip files are a secondary option for later cache or benchmarking experiments,
not the baseline corpus format.
Performance measurement should be first-class from the start: every major data
path should have a cheap benchmark command and structured throughput report.
The trainer-facing dataset should preserve that principle by exposing separate
read and transform timings instead of only reporting end-to-end sample latency.
The first training loop should follow the same rule by logging epoch-level
throughput and latency metrics to disk for both train and validation stages.

## Immediate next repo tasks

After this patch lands, the next high-value work items are:

1. add a downloader that mirrors selected manifest rows into
   `datasets/spacenet.ai/`
2. extend manifests with off-nadir metadata once the relevant NITF tags are
   being parsed
3. add the first retrieval baseline evaluation runner on on-demand PAN chips
4. add the first dense-task baseline evaluation runner on on-demand PAN chips

## Commands

Build GDAL + Kakadu:

```bash
./scripts/build_gdal_kakadu.sh
```

Generate manifests:

```bash
uv run geogrok-make-manifests
```

Smoke-test only a small slice:

```bash
uv run geogrok-make-manifests --limit-assets 8
```

Smoke-test on-demand PAN chip reads from the manifest:

```bash
source .local/gdal-kakadu/env.sh
./scripts/smoke_on_demand_chips.sh
```

Optional materialized chip export:

```bash
source .local/gdal-kakadu/env.sh
uv run geogrok-extract-chips --limit 16 --modality PAN
```

Benchmark on-demand chip throughput:

```bash
source .local/gdal-kakadu/env.sh
uv run geogrok-benchmark-chips --limit 32 --repeat 2 --warmup 2 --modality PAN
```

Benchmark the trainer-facing path with normalization enabled:

```bash
source .local/gdal-kakadu/env.sh
uv run geogrok-benchmark-training \
  --limit 32 \
  --repeat 2 \
  --warmup 2 \
  --modality PAN \
  --output-dtype float32 \
  --clip-min 0 \
  --clip-max 2047 \
  --scale-max 2047
```

Run the deterministic dry-run training loop:

```bash
source .local/gdal-kakadu/env.sh
uv run geogrok-run-baseline \
  --epochs 2 \
  --batch-size 8 \
  --train-limit 32 \
  --val-limit 16 \
  --output-dtype float32 \
  --clip-min 0 \
  --clip-max 2047 \
  --scale-max 2047
```

Run the deterministic embedding baseline:

```bash
source .local/gdal-kakadu/env.sh
uv run geogrok-run-embedding-baseline \
  --limit 64 \
  --modality PAN \
  --output-dtype float32 \
  --clip-min 0 \
  --clip-max 2047 \
  --scale-max 2047 \
  --positive-key scene_id
```
