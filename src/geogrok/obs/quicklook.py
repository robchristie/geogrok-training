from __future__ import annotations

import importlib
import io
from pathlib import Path
from typing import Any

import numpy as np

from geogrok.io.raster import normalize_chip_array, read_chip_array

from .data import chip_record


def require_pillow() -> Any:
    try:
        image_module = importlib.import_module("PIL.Image")
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Pillow is not installed in this repo environment. "
            "Run `uv sync --extra dev --extra obs`."
        ) from exc
    return image_module


def chip_quicklook_png_bytes(
    chip: dict[str, Any],
    *,
    size: int = 256,
    gdal_prefix: Path | None = None,
    pmin: float = 2.0,
    pmax: float = 98.0,
) -> bytes:
    image = chip_quicklook_image(
        chip,
        size=size,
        gdal_prefix=gdal_prefix,
        pmin=pmin,
        pmax=pmax,
    )
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def chip_quicklook_image(
    chip: dict[str, Any],
    *,
    size: int = 256,
    gdal_prefix: Path | None = None,
    pmin: float = 2.0,
    pmax: float = 98.0,
) -> Any:
    raster = read_chip_array(
        chip["local_path"],
        x0=int(chip["x0"]),
        y0=int(chip["y0"]),
        width=int(chip["width"]),
        height=int(chip["height"]),
        prefix=gdal_prefix,
    )
    array = normalize_chip_array(raster.array)
    display = normalize_for_display(array, pmin=pmin, pmax=pmax)
    image = to_pillow_image(display)
    resample = getattr(require_pillow(), "Resampling", None)
    if size > 0 and (image.width != size or image.height != size):
        if resample is not None:
            image = image.resize((size, size), resample.BILINEAR)
        else:
            image = image.resize((size, size))
    return image


def chip_quicklook_from_frame(
    chips_frame: Any,
    *,
    chip_id: str,
    size: int = 256,
    gdal_prefix: Path | None = None,
    pmin: float = 2.0,
    pmax: float = 98.0,
) -> bytes:
    chip = chip_record(chips_frame, chip_id)
    return chip_quicklook_png_bytes(
        chip,
        size=size,
        gdal_prefix=gdal_prefix,
        pmin=pmin,
        pmax=pmax,
    )


def pair_quicklook_png_bytes(
    chips_frame: Any,
    *,
    query_chip_id: str,
    candidate_chip_id: str,
    size: int = 256,
    gap: int = 12,
    gdal_prefix: Path | None = None,
    pmin: float = 2.0,
    pmax: float = 98.0,
) -> bytes:
    query_chip = chip_record(chips_frame, query_chip_id)
    candidate_chip = chip_record(chips_frame, candidate_chip_id)
    query_image = chip_quicklook_image(
        query_chip,
        size=size,
        gdal_prefix=gdal_prefix,
        pmin=pmin,
        pmax=pmax,
    )
    candidate_image = chip_quicklook_image(
        candidate_chip,
        size=size,
        gdal_prefix=gdal_prefix,
        pmin=pmin,
        pmax=pmax,
    )
    image_module = require_pillow()
    composite = image_module.new(
        "L",
        (
            query_image.width + candidate_image.width + gap,
            max(query_image.height, candidate_image.height),
        ),
        color=18,
    )
    composite.paste(query_image, (0, 0))
    composite.paste(candidate_image, (query_image.width + gap, 0))
    buffer = io.BytesIO()
    composite.save(buffer, format="PNG")
    return buffer.getvalue()


def normalize_for_display(
    array: np.ndarray,
    *,
    pmin: float,
    pmax: float,
) -> np.ndarray:
    if array.ndim != 3:
        raise ValueError(f"Expected (C, H, W) array, got shape {array.shape}")
    first_band = np.asarray(array[0], dtype=np.float32)
    low = float(np.percentile(first_band, pmin))
    high = float(np.percentile(first_band, pmax))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(first_band.min())
        high = float(first_band.max()) if first_band.max() > first_band.min() else low + 1.0
    scaled = np.clip((first_band - low) / max(high - low, 1e-6), 0.0, 1.0)
    return np.rint(scaled * 255.0).astype(np.uint8)


def to_pillow_image(array: np.ndarray) -> Any:
    image_module = require_pillow()
    if array.ndim == 2:
        return image_module.fromarray(array, mode="L")
    raise ValueError(f"Expected 2D grayscale array, got shape {array.shape}")
