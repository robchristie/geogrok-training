#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_SCRIPT="${ENV_SCRIPT:-$ROOT_DIR/.local/gdal-kakadu/env.sh}"
MANIFEST_ROOT="${MANIFEST_ROOT:-$ROOT_DIR/datasets/manifests/spacenet-benchmark-smoke}"
REPORT_PATH="${REPORT_PATH:-$ROOT_DIR/artifacts/benchmarks/smoke-chip-read-benchmark.json}"

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

uv run geogrok-benchmark-chips \
  --chips-path "$MANIFEST_ROOT/chips.parquet" \
  --split train \
  --modality PAN \
  --limit 4 \
  --repeat 2 \
  --warmup 2 \
  --output-path "$REPORT_PATH"

uv run python - <<'PY' "$REPORT_PATH"
from __future__ import annotations

import json
import sys
from pathlib import Path

report_path = Path(sys.argv[1]).resolve()
if not report_path.exists():
    raise SystemExit(f"missing benchmark report: {report_path}")

report = json.loads(report_path.read_text(encoding="utf-8"))
required = {
    "samples_per_second",
    "megapixels_per_second",
    "mebibytes_per_second",
    "latency_ms_p50",
    "latency_ms_p95",
}
missing = sorted(required.difference(report))
if missing:
    raise SystemExit(f"benchmark report missing keys: {missing}")

print(report)
PY

echo "Chip benchmark smoke test passed."
