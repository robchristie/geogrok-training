#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GDAL_ENV_SCRIPT="${GDAL_ENV_SCRIPT:-$ROOT_DIR/.local/gdal-kakadu/env.sh}"

if [[ ! -f "$GDAL_ENV_SCRIPT" ]]; then
  echo "missing GDAL/Kakadu env script: $GDAL_ENV_SCRIPT" >&2
  echo "Run ./scripts/build_gdal_kakadu.sh first." >&2
  exit 1
fi

set +u
source "$GDAL_ENV_SCRIPT"
set -u

PROJECT_PYTHON="$(uv run python - <<'PY'
import sys
print(sys.executable)
PY
)"

"$PROJECT_PYTHON" - <<'PY'
import numpy as np

try:
    import pykdu
except ModuleNotFoundError as exc:
    raise SystemExit(
        "pykdu is not installed in this repo environment. Run ./scripts/install_reference_pykdu.sh first."
    ) from exc

from pykdu import Decoder, Encoder

image = (np.arange(32 * 32, dtype=np.uint16).reshape(32, 32) % 4096).astype(np.uint16)
encoded = Encoder(params="Creversible=yes", container="j2k", bit_depth=12).encode(image)
decoded = Decoder().decode(encoded)

if not np.array_equal(decoded, image):
    raise SystemExit("reference pykdu check failed: decode does not match input")

print("pykdu import:", pykdu.__file__)
print("encoded bytes:", len(encoded))
print("reference pykdu runtime check passed.")
PY
