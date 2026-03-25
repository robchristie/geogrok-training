#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_SCRIPT="${ENV_SCRIPT:-$ROOT_DIR/.local/gdal-kakadu/env.sh}"
MANIFEST_ROOT="${MANIFEST_ROOT:-$ROOT_DIR/datasets/manifests/spacenet-torch-embedding-smoke}"
RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/artifacts/runs/torch-embedding-smoke}"

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

uv run geogrok-run-torch-embedding \
  --chips-path "$MANIFEST_ROOT/chips.parquet" \
  --run-root "$RUN_ROOT" \
  --train-split train \
  --query-split val \
  --query-split test \
  --gallery-split val \
  --gallery-split test \
  --modality PAN \
  --train-limit 256 \
  --eval-limit 96 \
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
  --dropout 0.0 \
  --epochs 24 \
  --steps-per-epoch 48 \
  --pairs-per-batch 32 \
  --eval-batch-size 64 \
  --learning-rate 0.001 \
  --temperature 0.1 \
  --weight-decay 1e-4 \
  --device auto \
  --amp \
  --seed 42

uv run python - <<'PY' "$RUN_ROOT"
from __future__ import annotations

import json
import sys
from pathlib import Path

run_root = Path(sys.argv[1]).resolve()
retrieval = json.loads((run_root / "retrieval.json").read_text(encoding="utf-8"))
training = json.loads((run_root / "training.json").read_text(encoding="utf-8"))
embedding = json.loads((run_root / "embedding.json").read_text(encoding="utf-8"))

print({"retrieval": retrieval, "training": training, "embedding": embedding})
PY

echo "Torch embedding smoke test passed."
