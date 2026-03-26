#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REFERENCE_PYKDU_DIR="${REFERENCE_PYKDU_DIR:-$ROOT_DIR/reference/geogrok/libs/py/pykdu}"
KDU_SRC_ROOT="${KDU_SRC_ROOT:-$ROOT_DIR/third_party/kakadu}"
KDU_PLATFORM="${KDU_PLATFORM:-Linux-x86-64-gcc}"
KDU_SOURCE_LIB_DIR="${KDU_SOURCE_LIB_DIR:-$KDU_SRC_ROOT/lib/$KDU_PLATFORM}"
KDU_STAGE_ROOT="${KDU_STAGE_ROOT:-$ROOT_DIR/.build/reference-pykdu-kakadu}"
KDU_STAGE_LIB_DIR="$KDU_STAGE_ROOT/lib"
GDAL_ENV_SCRIPT="${GDAL_ENV_SCRIPT:-$ROOT_DIR/.local/gdal-kakadu/env.sh}"

require_path() {
  local path="$1"
  local description="$2"
  if [[ ! -e "$path" ]]; then
    echo "missing ${description}: $path" >&2
    exit 1
  fi
}

require_command() {
  local name="$1"
  if ! command -v "$name" >/dev/null 2>&1; then
    echo "missing required command: $name" >&2
    exit 1
  fi
}

require_command uv
require_command cmake

require_path "$REFERENCE_PYKDU_DIR/pyproject.toml" "reference pykdu project"
require_path "$KDU_SRC_ROOT/coresys" "Kakadu source tree"
require_path "$KDU_SOURCE_LIB_DIR" "built Kakadu library directory"

if [[ ! -f "$GDAL_ENV_SCRIPT" ]]; then
  echo "missing GDAL/Kakadu env script: $GDAL_ENV_SCRIPT" >&2
  echo "Run ./scripts/build_gdal_kakadu.sh first." >&2
  exit 1
fi

mkdir -p "$KDU_STAGE_LIB_DIR"

found_any=0
for source_lib in "$KDU_SOURCE_LIB_DIR"/libkdu*.so; do
  if [[ ! -e "$source_lib" ]]; then
    continue
  fi
  found_any=1
  target_lib="$KDU_STAGE_LIB_DIR/$(basename "$source_lib")"
  ln -sfn "$source_lib" "$target_lib"
done

if [[ "$found_any" -ne 1 ]]; then
  echo "no Kakadu shared libraries found under: $KDU_SOURCE_LIB_DIR" >&2
  echo "Run ./scripts/build_gdal_kakadu.sh first so libkdu*.so exists." >&2
  exit 1
fi

set +u
source "$GDAL_ENV_SCRIPT"
set -u

export KDU_SRC_ROOT
export KDU_ROOT="$KDU_STAGE_ROOT"
export CMAKE_ARGS="${CMAKE_ARGS:-} -DKDU_SRC_ROOT=$KDU_SRC_ROOT -DKDU_ROOT=$KDU_ROOT"
PROJECT_PYTHON="$(uv run python - <<'PY'
import sys
print(sys.executable)
PY
)"

echo "Installing reference pykdu from: $REFERENCE_PYKDU_DIR"
echo "Using Kakadu source root: $KDU_SRC_ROOT"
echo "Using staged Kakadu lib root: $KDU_ROOT"
echo "Using repo Python: $PROJECT_PYTHON"

uv pip install --python "$PROJECT_PYTHON" -e "$REFERENCE_PYKDU_DIR"

"$PROJECT_PYTHON" - <<'PY'
import numpy as np
from pykdu import Decoder, Encoder

image = (np.arange(64 * 64, dtype=np.uint16).reshape(64, 64) % 2048).astype(np.uint16)
encoded = Encoder(params="Creversible=yes", container="j2k", bit_depth=11).encode(image)
decoded = Decoder().decode(encoded)

if decoded.shape != image.shape:
    raise SystemExit(f"shape mismatch: {decoded.shape} != {image.shape}")
if decoded.dtype != image.dtype:
    raise SystemExit(f"dtype mismatch: {decoded.dtype} != {image.dtype}")
if not np.array_equal(decoded, image):
    raise SystemExit("round-trip mismatch for reference pykdu install")

print("reference pykdu install OK", len(encoded))
PY

cat <<EOF
reference pykdu install complete.
reference package: $REFERENCE_PYKDU_DIR
editable install source: $REFERENCE_PYKDU_DIR
project python: $PROJECT_PYTHON
KDU_SRC_ROOT: $KDU_SRC_ROOT
KDU_ROOT: $KDU_ROOT
smoke: passed
EOF
