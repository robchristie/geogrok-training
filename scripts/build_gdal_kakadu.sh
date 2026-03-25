#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GDAL_SOURCE_DIR="${GDAL_SOURCE_DIR:-$ROOT_DIR/third_party/gdal}"
KDU_ROOT="${KDU_ROOT:-$ROOT_DIR/third_party/kakadu}"
PREFIX_DIR="${PREFIX_DIR:-$ROOT_DIR/.local/gdal-kakadu}"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/.build/gdal-kakadu}"
PYTHON_ENV_DIR="${PYTHON_ENV_DIR:-$ROOT_DIR/.build/gdal-kakadu-python}"
BUILD_JOBS="${BUILD_JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 8)}"
KDU_PLATFORM="${KDU_PLATFORM:-Linux-x86-64-gcc}"
SWIG_EXECUTABLE="${SWIG_EXECUTABLE:-}"
SWIG_DIR="${SWIG_DIR:-}"

require_command() {
  local command_name="$1"
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "missing required command: $command_name" >&2
    exit 1
  fi
}

require_command cmake
require_command make
require_command uv
require_command patchelf

if [[ ! -d "$GDAL_SOURCE_DIR" ]]; then
  echo "missing GDAL source tree at $GDAL_SOURCE_DIR" >&2
  exit 1
fi

if [[ ! -d "$KDU_ROOT" ]]; then
  echo "missing Kakadu source tree at $KDU_ROOT" >&2
  exit 1
fi

KDU_LIB_DIR="$KDU_ROOT/lib/$KDU_PLATFORM"

discover_swig() {
  if [[ -z "$SWIG_EXECUTABLE" ]]; then
    if command -v swig >/dev/null 2>&1; then
      SWIG_EXECUTABLE="$(command -v swig)"
    elif [[ -x "$HOME/.cache/rattler/cache/pkgs/swig-4.4.1-h793e66c_0/bin/swig" ]]; then
      SWIG_EXECUTABLE="$HOME/.cache/rattler/cache/pkgs/swig-4.4.1-h793e66c_0/bin/swig"
    else
      echo "missing required command: swig" >&2
      echo "Set SWIG_EXECUTABLE to a local SWIG binary if it is not on PATH." >&2
      exit 1
    fi
  fi

  if [[ -z "$SWIG_DIR" ]]; then
    local swig_prefix
    swig_prefix="$(cd "$(dirname "$SWIG_EXECUTABLE")/.." && pwd)"
    if [[ -f "$swig_prefix/share/swig/swig.swg" ]]; then
      SWIG_DIR="$swig_prefix/share/swig"
    else
      local candidate
      candidate="$(find "$swig_prefix/share/swig" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
      if [[ -n "$candidate" && -f "$candidate/swig.swg" ]]; then
        SWIG_DIR="$candidate"
      else
        echo "unable to determine SWIG_DIR under $swig_prefix/share/swig" >&2
        exit 1
      fi
    fi
  fi
}

ensure_kakadu_runtime() {
  local makefile_path
  makefile_path="$KDU_ROOT/managed/make/Makefile-$KDU_PLATFORM"

  mkdir -p "$KDU_LIB_DIR"

  if ! compgen -G "$KDU_LIB_DIR/libkdu_v*.so" >/dev/null; then
    echo "building Kakadu core library"
    make -C "$KDU_ROOT/coresys/make" -f "Makefile-$KDU_PLATFORM"
  fi

  if ! compgen -G "$KDU_LIB_DIR/libkdu_a*.so" >/dev/null; then
    echo "building Kakadu auxiliary library"
    if [[ ! -f "$makefile_path" ]]; then
      echo "missing Kakadu managed makefile: $makefile_path" >&2
      exit 1
    fi
    make -C "$KDU_ROOT/managed/make" -f "Makefile-$KDU_PLATFORM" all_but_jni
  fi
}

ensure_python_build_env() {
  if [[ ! -x "$PYTHON_ENV_DIR/bin/python" ]]; then
    uv venv "$PYTHON_ENV_DIR" --python 3.13 >/dev/null
  fi
  uv pip install --python "$PYTHON_ENV_DIR/bin/python" \
    "numpy>=2.0.0" \
    "setuptools>=70.0.0" \
    "wheel>=0.45.0" >/dev/null
}

ensure_kakadu_runtime
ensure_python_build_env
discover_swig

PYTHON_BIN="$PYTHON_ENV_DIR/bin/python"
export SWIG_LIB="$SWIG_DIR"
export PATH="$(dirname "$SWIG_EXECUTABLE"):$PATH"
HOST_PYTHON_PREFIX="$("$PYTHON_BIN" - <<'PY'
import sys
print(sys.base_prefix)
PY
)"
HOST_PYTHON_LIB_DIR=""
if [[ -d "$HOST_PYTHON_PREFIX/lib" ]]; then
  HOST_PYTHON_LIB_DIR="$HOST_PYTHON_PREFIX/lib"
fi
HOST_PROJ_DIR=""
if [[ -d "$HOST_PYTHON_PREFIX/share/proj" ]]; then
  HOST_PROJ_DIR="$HOST_PYTHON_PREFIX/share/proj"
fi

mkdir -p "$BUILD_DIR" "$PREFIX_DIR"

cmake -S "$GDAL_SOURCE_DIR" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PREFIX_DIR" \
  -DCMAKE_INSTALL_RPATH="$PREFIX_DIR/lib;$PREFIX_DIR/lib64;$KDU_LIB_DIR" \
  -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
  -DCMAKE_PREFIX_PATH="$HOST_PYTHON_PREFIX" \
  -DGDAL_USE_INTERNAL_LIBS_WHEN_POSSIBLE=ON \
  -DGDAL_USE_KDU=ON \
  -DKDU_ROOT="$KDU_ROOT" \
  -DGDAL_ENABLE_DRIVER_JP2KAK=ON \
  -DGDAL_ENABLE_DRIVER_JPIPKAK=OFF \
  -DGDAL_ENABLE_DRIVER_GTIFF=ON \
  -DGDAL_ENABLE_DRIVER_MEM=ON \
  -DGDAL_ENABLE_DRIVER_NITF=ON \
  -DGDAL_ENABLE_DRIVER_VRT=ON \
  -DBUILD_APPS=ON \
  -DBUILD_JAVA_BINDINGS=OFF \
  -DBUILD_CSHARP_BINDINGS=OFF \
  -DBUILD_PYTHON_BINDINGS=ON \
  -DGDAL_PYTHON_INSTALL_PREFIX="$PREFIX_DIR" \
  -DSWIG_EXECUTABLE="$SWIG_EXECUTABLE" \
  -DSWIG_DIR="$SWIG_DIR" \
  -DPython_FIND_VIRTUALENV=ONLY \
  -DPython_ROOT="$PYTHON_ENV_DIR" \
  -DPython_EXECUTABLE="$PYTHON_BIN"

cmake --build "$BUILD_DIR" --parallel "$BUILD_JOBS"
cmake --build "$BUILD_DIR" --target install

SITE_PACKAGES="$("$PYTHON_BIN" - <<'PY' "$PREFIX_DIR"
from pathlib import Path
import sys

prefix = Path(sys.argv[1])
version = f"{sys.version_info.major}.{sys.version_info.minor}"
candidates = [
    prefix / "lib" / f"python{version}" / "site-packages",
    prefix / "lib64" / f"python{version}" / "site-packages",
]
for candidate in candidates:
    if candidate.exists():
        print(candidate)
        raise SystemExit(0)
print(candidates[0])
PY
)"

PYTHON_EXTENSION_RPATH="$PREFIX_DIR/lib:$PREFIX_DIR/lib64:$KDU_LIB_DIR"
if [[ -n "$HOST_PYTHON_LIB_DIR" ]]; then
  PYTHON_EXTENSION_RPATH="$PYTHON_EXTENSION_RPATH:$HOST_PYTHON_LIB_DIR"
fi

while IFS= read -r extension_path; do
  patchelf --force-rpath --set-rpath "$PYTHON_EXTENSION_RPATH" "$extension_path"
done < <(find "$SITE_PACKAGES/osgeo" -maxdepth 1 -type f -name '_*.so' | sort)

ENV_SCRIPT="$PREFIX_DIR/env.sh"
cat >"$ENV_SCRIPT" <<EOF
export GDAL_KAKADU_PREFIX="$PREFIX_DIR"
export PATH="$PREFIX_DIR/bin:\${PATH}"
export GDAL_DATA="$PREFIX_DIR/share/gdal"
export PROJ_DATA="${HOST_PROJ_DIR:-$PREFIX_DIR/share/proj}"
export PROJ_LIB="${HOST_PROJ_DIR:-$PREFIX_DIR/share/proj}"
export LD_LIBRARY_PATH="$PREFIX_DIR/lib:$PREFIX_DIR/lib64:$KDU_LIB_DIR${HOST_PYTHON_LIB_DIR:+:$HOST_PYTHON_LIB_DIR}:\${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$SITE_PACKAGES:\${PYTHONPATH:-}"
EOF

set +u
source "$ENV_SCRIPT"
set -u

"$PREFIX_DIR/bin/gdalinfo" --formats | grep -q "JP2KAK"
"$PYTHON_BIN" - <<'PY'
from osgeo import gdal

gdal.UseExceptions()
driver = gdal.GetDriverByName("JP2KAK")
if driver is None:
    raise SystemExit("JP2KAK driver is not available in Python bindings")
print("GDAL version:", gdal.VersionInfo("--version"))
print("JP2KAK driver:", driver.ShortName)
PY

cat <<EOF
GDAL + Kakadu build complete.
prefix: $PREFIX_DIR
build: $BUILD_DIR
python env: $PYTHON_ENV_DIR
host python prefix: $HOST_PYTHON_PREFIX
swig executable: $SWIG_EXECUTABLE
swig dir: $SWIG_DIR
activate with:
  source "$ENV_SCRIPT"
EOF
