#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_SCRIPT="${ENV_SCRIPT:-$ROOT_DIR/.local/gdal-kakadu/env.sh}"
MANIFEST_ROOT="${MANIFEST_ROOT:-$ROOT_DIR/datasets/manifests/spacenet-training-smoke}"

if [[ ! -f "$ENV_SCRIPT" ]]; then
  echo "missing GDAL environment script: $ENV_SCRIPT" >&2
  echo "build the runtime first with ./scripts/build_gdal_kakadu.sh" >&2
  exit 1
fi

set +u
source "$ENV_SCRIPT"
set -u

uv run geogrok-make-manifests \
  --limit-assets 8 \
  --output-root "$MANIFEST_ROOT"

uv run python - <<'PY' "$MANIFEST_ROOT/chips.parquet"
from __future__ import annotations

import sys
from pathlib import Path

from geogrok.data.training import TrainingChipDataset, collate_training_samples

chips_path = Path(sys.argv[1]).resolve()
dataset = TrainingChipDataset.from_manifest(
    chips_path,
    splits=("train",),
    modalities=("PAN",),
    limit=2,
    output_dtype="float32",
    clip_min=0.0,
    clip_max=2047.0,
    scale_max=2047.0,
)
if len(dataset) != 2:
    raise SystemExit(f"expected 2 training samples, got {len(dataset)}")

samples = [dataset.sample(index) for index in range(len(dataset))]
batch = collate_training_samples(samples)

print(
    {
        "batch_shape": tuple(int(value) for value in batch.images.shape),
        "dtype": str(batch.images.dtype),
        "min": float(batch.images.min()),
        "max": float(batch.images.max()),
    }
)
for sample in samples:
    print(
        {
            "chip_id": sample.record.chip_id,
            "read_ms": round(sample.timing.read_ms, 3),
            "transform_ms": round(sample.timing.transform_ms, 3),
            "total_ms": round(sample.timing.total_ms, 3),
        }
    )
PY

echo "Training dataset smoke test passed."
