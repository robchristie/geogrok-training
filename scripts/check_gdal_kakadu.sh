#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREFIX_DIR="${PREFIX_DIR:-$ROOT_DIR/.local/gdal-kakadu}"
PYTHON_ENV_DIR="${PYTHON_ENV_DIR:-$ROOT_DIR/.build/gdal-kakadu-python}"
ENV_SCRIPT="${ENV_SCRIPT:-$PREFIX_DIR/env.sh}"

if [[ ! -f "$ENV_SCRIPT" ]]; then
  echo "missing GDAL environment script: $ENV_SCRIPT" >&2
  echo "build the runtime first with ./scripts/build_gdal_kakadu.sh" >&2
  exit 1
fi

if [[ ! -x "$PREFIX_DIR/bin/gdalinfo" ]]; then
  echo "missing gdalinfo at $PREFIX_DIR/bin/gdalinfo" >&2
  exit 1
fi

if [[ ! -x "$PYTHON_ENV_DIR/bin/python" ]]; then
  echo "missing Python build environment at $PYTHON_ENV_DIR/bin/python" >&2
  exit 1
fi

set +u
source "$ENV_SCRIPT"
set -u

echo "Checking gdalinfo driver visibility"
"$PREFIX_DIR/bin/gdalinfo" --formats | grep -q "JP2KAK"
"$PREFIX_DIR/bin/gdalinfo" --formats | grep -q "NITF"

echo "Checking Python binding driver visibility"
"$PYTHON_ENV_DIR/bin/python" - <<'PY'
from osgeo import gdal

gdal.UseExceptions()
version = gdal.VersionInfo("--version")
jp2kak = gdal.GetDriverByName("JP2KAK")
nitf = gdal.GetDriverByName("NITF")

if jp2kak is None:
    raise SystemExit("JP2KAK driver is not available from Python")
if nitf is None:
    raise SystemExit("NITF driver is not available from Python")

print(version)
print("JP2KAK", jp2kak.ShortName)
print("NITF", nitf.ShortName)
PY

echo "GDAL/Kakadu runtime check passed."
