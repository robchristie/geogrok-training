#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_SCRIPT="${ENV_SCRIPT:-$ROOT_DIR/.local/gdal-kakadu/env.sh}"
MANIFEST_ROOT="${MANIFEST_ROOT:-$ROOT_DIR/datasets/manifests/spacenet-chip-smoke}"
CHIP_ROOT="${CHIP_ROOT:-$ROOT_DIR/datasets/chips/spacenet-chip-smoke}"

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

uv run geogrok-extract-chips \
  --chips-path "$MANIFEST_ROOT/chips.parquet" \
  --output-root "$CHIP_ROOT" \
  --modality PAN \
  --limit 2 \
  --overwrite

uv run python - <<'PY' "$CHIP_ROOT"
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

root = Path(sys.argv[1]).resolve()
index_path = root / "index.parquet"
if not index_path.exists():
    raise SystemExit(f"missing chip index: {index_path}")

frame = pd.read_parquet(index_path)
if frame.empty:
    raise SystemExit("chip extraction produced no rows")

missing = [path for path in frame["output_path"] if not Path(path).exists()]
if missing:
    raise SystemExit(f"missing extracted chips: {missing[:3]}")

print({"rows": len(frame), "write_status": frame["write_status"].value_counts().to_dict()})
print(frame[["chip_id", "output_path"]].head(2).to_string(index=False))
PY

echo "Chip extraction smoke test passed."
