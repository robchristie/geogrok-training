#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_SCRIPT="${ENV_SCRIPT:-$ROOT_DIR/.local/gdal-kakadu/env.sh}"
MANIFEST_ROOT="${MANIFEST_ROOT:-$ROOT_DIR/datasets/manifests/spacenet-embedding-smoke}"
RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/artifacts/runs/embedding-baseline-smoke}"

if [[ ! -f "$ENV_SCRIPT" ]]; then
  echo "missing GDAL environment script: $ENV_SCRIPT" >&2
  echo "build the runtime first with ./scripts/build_gdal_kakadu.sh" >&2
  exit 1
fi

set +u
source "$ENV_SCRIPT"
set -u

uv run geogrok-make-manifests \
  --output-root "$MANIFEST_ROOT"

uv run geogrok-run-embedding-baseline \
  --chips-path "$MANIFEST_ROOT/chips.parquet" \
  --run-root "$RUN_ROOT" \
  --query-split val \
  --query-split test \
  --gallery-split val \
  --gallery-split test \
  --modality PAN \
  --limit 96 \
  --max-chips-per-scene 4 \
  --output-dtype float32 \
  --clip-min 0 \
  --clip-max 2047 \
  --scale-max 2047 \
  --positive-key scene_id \
  --min-positive-center-distance 1024

uv run python - <<'PY' "$RUN_ROOT"
from __future__ import annotations

import json
import sys
from pathlib import Path

run_root = Path(sys.argv[1]).resolve()
retrieval_path = run_root / "retrieval.json"
benchmark_path = run_root / "benchmark.json"

retrieval = json.loads(retrieval_path.read_text(encoding="utf-8"))
benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))

print({"retrieval": retrieval, "benchmark": benchmark})
PY

echo "Embedding baseline smoke test passed."
