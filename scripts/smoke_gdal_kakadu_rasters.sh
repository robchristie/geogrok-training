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

if [[ ! -x "$PYTHON_ENV_DIR/bin/python" ]]; then
  echo "missing Python build environment at $PYTHON_ENV_DIR/bin/python" >&2
  exit 1
fi

set +u
source "$ENV_SCRIPT"
set -u

"$PYTHON_ENV_DIR/bin/python" - "$ROOT_DIR" <<'PY'
from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from pathlib import Path

from osgeo import gdal


def search_roots(root_dir: Path) -> list[Path]:
    roots = [root_dir / "datasets" / "spacenet.ai"]
    extra = os.environ.get("RASTER_SEARCH_ROOTS", "")
    for token in extra.split(":"):
        if token:
            roots.append(Path(token).expanduser())

    deduped: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        resolved = root.resolve(strict=False)
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(root)
    return deduped


def candidate_files(
    root_dir: Path,
    *,
    env_var: str,
    dataset_suffixes: tuple[str, ...],
    fallback: tuple[Path, ...],
) -> Iterator[Path]:
    explicit = os.environ.get(env_var)
    seen: set[Path] = set()

    def emit(candidate: Path) -> Iterator[Path]:
        resolved = candidate.resolve(strict=False)
        if resolved in seen:
            return
        seen.add(resolved)
        yield candidate

    if explicit:
        yield from emit(Path(explicit).expanduser())

    for root in search_roots(root_dir):
        if not root.exists():
            continue
        for suffix in dataset_suffixes:
            for candidate in root.rglob(f"*{suffix}"):
                yield from emit(candidate)

    for candidate in fallback:
        yield from emit(candidate)


def describe_source(path: Path, *, root_dir: Path) -> str:
    dataset_root = root_dir / "datasets" / "spacenet.ai"
    if path.is_relative_to(dataset_root):
        return "dataset"
    if path.is_relative_to(root_dir / "third_party" / "gdal"):
        return "fixture"
    return "explicit"


def open_first_working(
    *,
    root_dir: Path,
    label: str,
    env_var: str,
    dataset_suffixes: tuple[str, ...],
    fallback: tuple[Path, ...],
    expected_driver: str,
) -> Path:
    errors: list[str] = []
    for candidate in candidate_files(
        root_dir,
        env_var=env_var,
        dataset_suffixes=dataset_suffixes,
        fallback=fallback,
    ):
        if not candidate.exists():
            errors.append(f"{candidate}: missing")
            continue

        try:
            dataset = gdal.Open(str(candidate), gdal.GA_ReadOnly)
            if dataset is None:
                raise RuntimeError("gdal.Open returned None")

            driver_name = dataset.GetDriver().ShortName
            if driver_name != expected_driver:
                raise RuntimeError(f"expected driver {expected_driver}, got {driver_name}")

            x_size = dataset.RasterXSize
            y_size = dataset.RasterYSize
            if x_size <= 0 or y_size <= 0:
                raise RuntimeError(f"invalid raster size {x_size}x{y_size}")

            band = dataset.GetRasterBand(1)
            if band is None:
                raise RuntimeError("dataset has no readable band 1")

            window_x = min(64, x_size)
            window_y = min(64, y_size)
            payload = band.ReadRaster(0, 0, window_x, window_y)
            if not payload:
                raise RuntimeError("ReadRaster returned no data")

            checksum = band.Checksum(0, 0, window_x, window_y)
            print(
                f"{label}: {candidate} "
                f"(source={describe_source(candidate, root_dir=root_dir)}, "
                f"driver={driver_name}, size={x_size}x{y_size}, "
                f"bands={dataset.RasterCount}, band1={gdal.GetDataTypeName(band.DataType)}, "
                f"checksum64={checksum})"
            )
            return candidate
        except Exception as exc:
            errors.append(f"{candidate}: {exc}")

    detail = "\n".join(f"  - {entry}" for entry in errors[:10])
    raise SystemExit(f"unable to open a working {label} sample\n{detail}")


def main() -> int:
    root_dir = Path(sys.argv[1]).resolve()
    gdal.UseExceptions()

    geotiff_fallback = (
        root_dir / "third_party" / "gdal" / "autotest" / "cpp" / "data" / "byte.tif",
        root_dir / "third_party" / "gdal" / "autotest" / "gdrivers" / "data" / "byte.tif",
    )
    nitf_fallback = (
        root_dir / "third_party" / "gdal" / "autotest" / "gdrivers" / "data" / "nitf" / "two_images_jp2.ntf",
        root_dir / "third_party" / "gdal" / "autotest" / "gdrivers" / "data" / "nitf" / "rgb.ntf",
    )

    open_first_working(
        root_dir=root_dir,
        label="GeoTIFF",
        env_var="GEOTIFF_PATH",
        dataset_suffixes=(".tif", ".TIF", ".tiff", ".TIFF"),
        fallback=geotiff_fallback,
        expected_driver="GTiff",
    )
    open_first_working(
        root_dir=root_dir,
        label="NITF",
        env_var="NITF_PATH",
        dataset_suffixes=(".ntf", ".NTF", ".nitf", ".NITF"),
        fallback=nitf_fallback,
        expected_driver="NITF",
    )

    print("GDAL/Kakadu raster smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PY
