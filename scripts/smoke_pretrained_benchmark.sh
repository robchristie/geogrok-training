#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_SCRIPT="${ENV_SCRIPT:-$ROOT_DIR/.local/gdal-kakadu/env.sh}"
MANIFEST_ROOT="${MANIFEST_ROOT:-$ROOT_DIR/datasets/manifests/spacenet-pretrained-benchmark-smoke}"
PAIRS_ROOT="${PAIRS_ROOT:-$ROOT_DIR/datasets/pairs/spacenet-pretrained-benchmark-smoke}"
RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/artifacts/runs/pretrained-benchmark-smoke}"
MAX_CHIPS_PER_ASSET="${MAX_CHIPS_PER_ASSET:-8}"
EVAL_LIMIT="${EVAL_LIMIT:-512}"

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

uv run geogrok-make-pairs \
  --chips-path "$MANIFEST_ROOT/chips.parquet" \
  --scenes-path "$MANIFEST_ROOT/scenes.parquet" \
  --output-root "$PAIRS_ROOT" \
  --modality PAN \
  --city Jacksonville \
  --city Omaha \
  --city UCSD \
  --max-chips-per-asset "$MAX_CHIPS_PER_ASSET" \
  --positive-overlap-fraction 0.5 \
  --weak-overlap-fraction 0.2 \
  --hard-negative-radius-m 800 \
  --max-positives-per-query 2 \
  --max-hard-negatives-per-query 2

uv run geogrok-benchmark-pretrained \
  --chips-path "$MANIFEST_ROOT/chips.parquet" \
  --pairs-path "$PAIRS_ROOT/pairs.parquet" \
  --run-root "$RUN_ROOT" \
  --query-split val \
  --query-split test \
  --gallery-split val \
  --gallery-split test \
  --modality PAN \
  --eval-limit "$EVAL_LIMIT" \
  --batch-size 64 \
  --model resnet18 \
  --model resnet50 \
  --model vit_b_16 \
  --device auto \
  --amp

uv run python - <<'PY' "$RUN_ROOT"
from __future__ import annotations

import json
import sys
from pathlib import Path

run_root = Path(sys.argv[1]).resolve()
summary = json.loads((run_root / "summary.json").read_text(encoding="utf-8"))
print(summary)
PY

echo "Pretrained benchmark smoke test passed."
