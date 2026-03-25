from __future__ import annotations

import importlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from geogrok.io.gdal_env import activate, discover_runtime


@dataclass(frozen=True)
class PixelWindow:
    x0: int
    y0: int
    width: int
    height: int


@dataclass(frozen=True)
class RasterMetadata:
    path: Path
    driver: str
    raster_x: int
    raster_y: int
    band_count: int
    band_dtypes: tuple[str, ...]
    projection_wkt: str | None
    geotransform: tuple[float, ...] | None
    metadata_domains: tuple[str, ...]
    has_rpc_metadata: bool


@dataclass(frozen=True)
class RasterChip:
    path: Path
    window: PixelWindow
    band_count: int
    band_dtypes: tuple[str, ...]
    data: bytes


@dataclass(frozen=True)
class RasterArrayChip:
    path: Path
    window: PixelWindow
    band_count: int
    band_dtypes: tuple[str, ...]
    array: np.ndarray


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_gdal_prefix() -> Path:
    configured = Path.cwd() / ".local" / "gdal-kakadu"
    if configured.exists():
        return configured.resolve()
    return (repo_root() / ".local" / "gdal-kakadu").resolve()


def load_gdal(prefix: str | Path | None = None) -> Any:
    resolved_prefix = Path(prefix) if prefix is not None else default_gdal_prefix()
    activate(resolved_prefix)
    try:
        gdal = importlib.import_module("osgeo.gdal")
    except Exception as exc:
        runtime = discover_runtime(resolved_prefix)
        env_script = runtime.prefix / "env.sh"
        raise RuntimeError(
            "Unable to import osgeo.gdal. Source the GDAL runtime environment before "
            f"running Python commands that use raster I/O: source {env_script}"
        ) from exc

    gdal.UseExceptions()
    return gdal


def inspect_raster(path: str | Path, *, prefix: str | Path | None = None) -> RasterMetadata:
    dataset, gdal = open_dataset(path, prefix=prefix)
    try:
        return _metadata_from_dataset(path=Path(path), dataset=dataset, gdal=gdal)
    finally:
        dataset = None


def read_chip(
    path: str | Path,
    *,
    x0: int,
    y0: int,
    width: int,
    height: int,
    prefix: str | Path | None = None,
) -> RasterChip:
    window = PixelWindow(x0=x0, y0=y0, width=width, height=height)
    dataset, gdal = open_dataset(path, prefix=prefix)
    try:
        _validate_window(window, dataset.RasterXSize, dataset.RasterYSize)
        payload = dataset.ReadRaster(window.x0, window.y0, window.width, window.height)
        if not payload:
            raise RuntimeError(f"ReadRaster returned no data for {path}")
        band_dtypes = tuple(
            gdal.GetDataTypeName(dataset.GetRasterBand(index).DataType)
            for index in range(1, dataset.RasterCount + 1)
        )
        return RasterChip(
            path=Path(path).expanduser().resolve(),
            window=window,
            band_count=dataset.RasterCount,
            band_dtypes=band_dtypes,
            data=payload,
        )
    finally:
        dataset = None


def read_chip_array(
    path: str | Path,
    *,
    x0: int,
    y0: int,
    width: int,
    height: int,
    prefix: str | Path | None = None,
) -> RasterArrayChip:
    window = PixelWindow(x0=x0, y0=y0, width=width, height=height)
    dataset, gdal = open_dataset(path, prefix=prefix)
    try:
        _validate_window(window, dataset.RasterXSize, dataset.RasterYSize)
        raw = dataset.ReadAsArray(window.x0, window.y0, window.width, window.height)
        if raw is None:
            raise RuntimeError(f"ReadAsArray returned no data for {path}")
        array = normalize_chip_array(raw)
        band_dtypes = tuple(
            gdal.GetDataTypeName(dataset.GetRasterBand(index).DataType)
            for index in range(1, dataset.RasterCount + 1)
        )
        return RasterArrayChip(
            path=Path(path).expanduser().resolve(),
            window=window,
            band_count=dataset.RasterCount,
            band_dtypes=band_dtypes,
            array=array,
        )
    finally:
        dataset = None


def extract_chip_to_geotiff(
    path: str | Path,
    *,
    output_path: str | Path,
    x0: int,
    y0: int,
    width: int,
    height: int,
    prefix: str | Path | None = None,
    creation_options: tuple[str, ...] = ("TILED=YES", "COMPRESS=LZW", "BIGTIFF=IF_SAFER"),
) -> Path:
    window = PixelWindow(x0=x0, y0=y0, width=width, height=height)
    dataset, gdal = open_dataset(path, prefix=prefix)
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        _validate_window(window, dataset.RasterXSize, dataset.RasterYSize)
        options = gdal.TranslateOptions(
            format="GTiff",
            srcWin=[window.x0, window.y0, window.width, window.height],
            creationOptions=list(creation_options),
        )
        translated = gdal.Translate(str(output), dataset, options=options)
        if translated is None:
            raise RuntimeError(f"gdal.Translate failed for {path} -> {output}")
    finally:
        dataset = None
        translated = None

    return output


def open_dataset(
    path: str | Path,
    *,
    prefix: str | Path | None = None,
) -> tuple[Any, Any]:
    source = Path(path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(source)

    gdal = load_gdal(prefix)
    dataset = gdal.Open(str(source), gdal.GA_ReadOnly)
    if dataset is None:
        raise RuntimeError(f"Unable to open raster: {source}")
    return dataset, gdal


def _metadata_from_dataset(path: Path, *, dataset: Any, gdal: Any) -> RasterMetadata:
    projection_wkt = dataset.GetProjectionRef() or None
    geotransform_value = dataset.GetGeoTransform(can_return_null=True)
    geotransform = _normalize_geotransform(geotransform_value)
    domains = tuple(sorted(dataset.GetMetadataDomainList() or ()))
    band_dtypes = tuple(
        gdal.GetDataTypeName(dataset.GetRasterBand(index).DataType)
        for index in range(1, dataset.RasterCount + 1)
    )
    return RasterMetadata(
        path=path.expanduser().resolve(),
        driver=dataset.GetDriver().ShortName,
        raster_x=dataset.RasterXSize,
        raster_y=dataset.RasterYSize,
        band_count=dataset.RasterCount,
        band_dtypes=band_dtypes,
        projection_wkt=projection_wkt,
        geotransform=geotransform,
        metadata_domains=domains,
        has_rpc_metadata=bool(dataset.GetMetadata("RPC")),
    )


def _validate_window(window: PixelWindow, raster_x: int, raster_y: int) -> None:
    if window.width <= 0 or window.height <= 0:
        raise ValueError("Chip window must have positive width and height.")
    if window.x0 < 0 or window.y0 < 0:
        raise ValueError("Chip window origin must be non-negative.")
    if window.x0 + window.width > raster_x or window.y0 + window.height > raster_y:
        raise ValueError(
            f"Chip window {window} exceeds raster bounds {raster_x}x{raster_y}."
        )


def _normalize_geotransform(value: Sequence[float] | None) -> tuple[float, ...] | None:
    if value is None:
        return None
    return tuple(float(component) for component in value)


def normalize_chip_array(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 2:
        return array[np.newaxis, :, :]
    if array.ndim == 3:
        return array
    raise ValueError(f"Expected a 2D or 3D raster array, got shape {array.shape}.")
