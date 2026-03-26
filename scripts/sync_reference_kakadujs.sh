#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REFERENCE_ROOT="${REFERENCE_ROOT:-$ROOT_DIR/reference/geogrok}"
REFERENCE_UI_DIR="${REFERENCE_UI_DIR:-$REFERENCE_ROOT/services/ui}"
REFERENCE_KAKADUJS_DIR="${REFERENCE_KAKADUJS_DIR:-$REFERENCE_ROOT/third_party/kakadujs}"
REFERENCE_KAKADU_LINK="${REFERENCE_KAKADU_LINK:-$REFERENCE_ROOT/third_party/kakadu}"
SOURCE_KAKADU_DIR="${SOURCE_KAKADU_DIR:-$ROOT_DIR/third_party/kakadu}"
TARGET_DIR="${TARGET_DIR:-$ROOT_DIR/web/static/kakadujs}"
REFERENCE_STATIC_DIR="${REFERENCE_STATIC_DIR:-$REFERENCE_UI_DIR/static/kakadujs}"
REFERENCE_ARTIFACT_DIR="${REFERENCE_ARTIFACT_DIR:-$REFERENCE_ROOT/artifacts/wasm/kakadujs}"

require_command() {
  local command_name="$1"
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "missing required command: $command_name" >&2
    exit 1
  fi
}

require_path() {
  local path="$1"
  local description="$2"
  if [[ ! -e "$path" ]]; then
    echo "missing ${description}: $path" >&2
    exit 1
  fi
}

copy_assets() {
  local source_dir="$1"
  mkdir -p "$TARGET_DIR"
  cp -f "$source_dir/kakadujs.js" "$TARGET_DIR/kakadujs.js"
  cp -f "$source_dir/kakadujs.wasm" "$TARGET_DIR/kakadujs.wasm"
  echo "Copied Kakadu WASM artifacts to:"
  echo "  $TARGET_DIR/kakadujs.js"
  echo "  $TARGET_DIR/kakadujs.wasm"
}

require_command git
require_command npm
require_path "$REFERENCE_ROOT/.gitmodules" "reference geogrok checkout"
require_path "$SOURCE_KAKADU_DIR/Enabling_HT.txt" "top-level Kakadu source tree"

if [[ -f "$REFERENCE_STATIC_DIR/kakadujs.js" && -f "$REFERENCE_STATIC_DIR/kakadujs.wasm" ]]; then
  copy_assets "$REFERENCE_STATIC_DIR"
  exit 0
fi

if [[ -f "$REFERENCE_ARTIFACT_DIR/kakadujs.js" && -f "$REFERENCE_ARTIFACT_DIR/kakadujs.wasm" ]]; then
  copy_assets "$REFERENCE_ARTIFACT_DIR"
  exit 0
fi

if [[ ! -d "$REFERENCE_KAKADUJS_DIR/.git" && ! -f "$REFERENCE_KAKADUJS_DIR/.git" ]]; then
  (
    cd "$REFERENCE_ROOT"
    git submodule update --init --recursive third_party/kakadujs
  )
fi

if [[ -L "$REFERENCE_KAKADU_LINK" ]]; then
  :
elif [[ -e "$REFERENCE_KAKADU_LINK" ]]; then
  echo "reference Kakadu path already exists and is not a symlink: $REFERENCE_KAKADU_LINK" >&2
  echo "Remove it or set REFERENCE_KAKADU_LINK to a different bridge location." >&2
  exit 1
else
  ln -s "$SOURCE_KAKADU_DIR" "$REFERENCE_KAKADU_LINK"
fi

(
  cd "$REFERENCE_ROOT"
  ./tooling/third_party/ensure_kakadujs.sh
  services/ui/scripts/build-kakadujs-wasm.sh
)

if [[ -f "$REFERENCE_STATIC_DIR/kakadujs.js" && -f "$REFERENCE_STATIC_DIR/kakadujs.wasm" ]]; then
  copy_assets "$REFERENCE_STATIC_DIR"
  exit 0
fi

if [[ -f "$REFERENCE_ARTIFACT_DIR/kakadujs.js" && -f "$REFERENCE_ARTIFACT_DIR/kakadujs.wasm" ]]; then
  copy_assets "$REFERENCE_ARTIFACT_DIR"
  exit 0
fi

echo "failed to produce Kakadu WASM artifacts via reference/geogrok" >&2
exit 1
