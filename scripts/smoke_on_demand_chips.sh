#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_SCRIPT="${ENV_SCRIPT:-$ROOT_DIR/.local/gdal-kakadu/env.sh}"
MANIFEST_ROOT="${MANIFEST_ROOT:-$ROOT_DIR/datasets/manifests/spacenet-ondemand-smoke}"

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

from geogrok.data.runtime import OnDemandChipDataset

chips_path = Path(sys.argv[1]).resolve()
dataset = OnDemandChipDataset.from_manifest(
    chips_path,
    splits=("train",),
    modalities=("PAN",),
    limit=2,
)
if len(dataset) != 2:
    raise SystemExit(f"expected 2 PAN samples, got {len(dataset)}")

for index in range(len(dataset)):
    sample = dataset.sample(index)
    array = sample.chip.array
    print(
        {
            "chip_id": sample.record.chip_id,
            "path": str(sample.record.local_path),
            "shape": tuple(int(value) for value in array.shape),
            "dtype": str(array.dtype),
            "min": int(array.min()),
            "max": int(array.max()),
        }
    )
PY

echo "On-demand chip smoke test passed."
