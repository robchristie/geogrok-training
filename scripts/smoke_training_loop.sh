#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_SCRIPT="${ENV_SCRIPT:-$ROOT_DIR/.local/gdal-kakadu/env.sh}"
MANIFEST_ROOT="${MANIFEST_ROOT:-$ROOT_DIR/datasets/manifests/spacenet-loop-smoke}"
RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/artifacts/runs/training-loop-smoke}"

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

uv run geogrok-run-baseline \
  --chips-path "$MANIFEST_ROOT/chips.parquet" \
  --run-root "$RUN_ROOT" \
  --train-limit 8 \
  --val-limit 4 \
  --epochs 2 \
  --batch-size 2 \
  --output-dtype float32 \
  --clip-min 0 \
  --clip-max 2047 \
  --scale-max 2047

uv run python - <<'PY' "$RUN_ROOT"
from __future__ import annotations

import json
import sys
from pathlib import Path

run_root = Path(sys.argv[1]).resolve()
metrics_path = run_root / "metrics.jsonl"
summary_path = run_root / "summary.json"

if not metrics_path.exists():
    raise SystemExit(f"missing metrics file: {metrics_path}")
if not summary_path.exists():
    raise SystemExit(f"missing summary file: {summary_path}")

lines = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line]
summary = json.loads(summary_path.read_text(encoding="utf-8"))
if not lines:
    raise SystemExit("metrics.jsonl is empty")

print({"epochs_logged": len(lines), "train": summary.get("train"), "val": summary.get("val")})
PY

echo "Training loop smoke test passed."
