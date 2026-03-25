#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_SCRIPT="${ENV_SCRIPT:-$ROOT_DIR/.local/gdal-kakadu/env.sh}"
MANIFEST_ROOT="${MANIFEST_ROOT:-$ROOT_DIR/datasets/manifests/spacenet-pairs-smoke}"
PAIRS_ROOT="${PAIRS_ROOT:-$ROOT_DIR/datasets/pairs/spacenet-pairs-smoke}"

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
  --limit-assets 8 \
  --limit-asset-pairs 4 \
  --max-chips-per-asset 48 \
  --positive-overlap-fraction 0.5 \
  --weak-overlap-fraction 0.2 \
  --hard-negative-radius-m 800 \
  --max-positives-per-query 2 \
  --max-hard-negatives-per-query 2

uv run python - <<'PY' "$PAIRS_ROOT"
from __future__ import annotations

import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
print(summary)
PY

echo "Pair mining smoke test passed."
