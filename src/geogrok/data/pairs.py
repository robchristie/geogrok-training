from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from shapely import STRtree, from_wkb
from shapely.geometry import Polygon

from geogrok.io.raster import load_gdal

DEFAULT_OUTPUT_ROOT = Path("datasets/pairs/spacenet")
WGS84_A = 6378137.0
WGS84_E2 = 6.69437999014e-3


@dataclass(frozen=True)
class PairMiningSummary:
    scenes: int
    chips: int
    chip_rois: int
    asset_pairs: int
    pair_rows: int
    elapsed_seconds: float
    by_label: dict[str, int]
    by_city: dict[str, int]
    by_modality: dict[str, int]
    by_query_split: dict[str, int]
    by_candidate_split: dict[str, int]
    by_split_pair_label: dict[str, int]


@dataclass(frozen=True)
class LocalOrigin:
    city: str
    lon_deg: float
    lat_deg: float
    ecef_x: float
    ecef_y: float
    ecef_z: float


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build explicit chip-pair retrieval labels.")
    parser.add_argument(
        "--chips-path",
        type=Path,
        default=Path("datasets/manifests/spacenet/chips.parquet"),
    )
    parser.add_argument(
        "--scenes-path",
        type=Path,
        help="Scene manifest. Defaults to chips-path sibling scenes.parquet.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--gdal-prefix",
        type=Path,
        help="Override the local GDAL/Kakadu install prefix.",
    )
    parser.add_argument("--split", action="append", help="Restrict chips/scenes to this split.")
    parser.add_argument(
        "--modality",
        action="append",
        default=["PAN"],
        help="Restrict to this modality. Repeat to add more modalities.",
    )
    parser.add_argument("--city", action="append", help="Restrict to this city.")
    parser.add_argument("--sensor", action="append", help="Restrict to this sensor.")
    parser.add_argument(
        "--limit-assets",
        type=int,
        help="Optional cap after scene filtering, before overlap mining.",
    )
    parser.add_argument(
        "--limit-asset-pairs",
        type=int,
        help="Optional cap on overlapping asset pairs.",
    )
    parser.add_argument(
        "--max-chips-per-asset",
        type=int,
        default=256,
        help="Cap chip ROI extraction per asset to keep pair mining tractable.",
    )
    parser.add_argument(
        "--require-different-capture",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require positive candidates to come from a different capture_id.",
    )
    parser.add_argument(
        "--min-time-delta-seconds",
        type=float,
        default=0.0,
        help="Minimum absolute acquisition time difference between paired assets.",
    )
    parser.add_argument(
        "--positive-overlap-fraction",
        type=float,
        default=0.5,
        help="Minimum overlap/min(area) for an exact positive.",
    )
    parser.add_argument(
        "--weak-overlap-fraction",
        type=float,
        default=0.2,
        help="Minimum overlap/min(area) for a weak positive.",
    )
    parser.add_argument(
        "--hard-negative-radius-m",
        type=float,
        default=800.0,
        help="Search radius for spatially nearby hard negatives.",
    )
    parser.add_argument(
        "--max-positives-per-query",
        type=int,
        default=4,
        help="Maximum exact or weak positives to keep per query direction.",
    )
    parser.add_argument(
        "--max-hard-negatives-per-query",
        type=int,
        default=4,
        help="Maximum hard negatives to keep per query direction.",
    )
    return parser.parse_args(argv)


def resolve_scenes_path(chips_path: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    candidate = chips_path.resolve().with_name("scenes.parquet")
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Unable to infer scenes.parquet next to {chips_path}")


def load_frames(chips_path: Path, scenes_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    chips = pd.read_parquet(chips_path)
    scenes = pd.read_parquet(scenes_path)
    return chips, scenes


def filter_frame(frame: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    result = frame.copy()
    if "local_exists" in result:
        result = result[result["local_exists"].fillna(False)]
    if args.split:
        allowed = {token.lower() for token in args.split}
        result = result[result["split"].astype(str).str.lower().isin(allowed)]
    if args.modality:
        allowed = {token.upper() for token in args.modality}
        result = result[result["modality"].astype(str).str.upper().isin(allowed)]
    if args.city:
        allowed = {token.lower() for token in args.city}
        result = result[result["city"].astype(str).str.lower().isin(allowed)]
    if args.sensor:
        allowed = {token.upper() for token in args.sensor}
        result = result[result["sensor"].astype(str).str.upper().isin(allowed)]
    return result.reset_index(drop=True)


def build_city_origins(scenes: pd.DataFrame) -> dict[str, LocalOrigin]:
    origins: dict[str, LocalOrigin] = {}
    for city, group in scenes.groupby("city", sort=True):
        geometries = [from_wkb(value) for value in group["geom"] if value is not None]
        if not geometries:
            continue
        centroids = [geometry.centroid for geometry in geometries]
        lon_deg = float(np.mean([point.x for point in centroids]))
        lat_deg = float(np.mean([point.y for point in centroids]))
        ecef_x, ecef_y, ecef_z = geodetic_to_ecef(lon_deg, lat_deg)
        origins[str(city)] = LocalOrigin(
            city=str(city),
            lon_deg=lon_deg,
            lat_deg=lat_deg,
            ecef_x=ecef_x,
            ecef_y=ecef_y,
            ecef_z=ecef_z,
        )
    return origins


def find_asset_pairs(
    scenes: pd.DataFrame,
    *,
    require_different_capture: bool,
    min_time_delta_seconds: float,
    limit_asset_pairs: int | None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _group_key, group in scenes.groupby(["city", "modality"], sort=True):
        geometries = [from_wkb(value) for value in group["geom"]]
        tree = STRtree(geometries)
        records = group.to_dict("records")
        for index, query_row in enumerate(records):
            candidates = tree.query(geometries[index], predicate="intersects")
            for candidate_index in candidates:
                candidate_index = int(candidate_index)
                if candidate_index <= index:
                    continue
                candidate_row = records[candidate_index]
                if require_different_capture and (
                    optional_string(query_row.get("capture_id"))
                    == optional_string(candidate_row.get("capture_id"))
                ):
                    continue
                time_delta_seconds = timestamp_delta_seconds(
                    query_row.get("acq_time"),
                    candidate_row.get("acq_time"),
                )
                if time_delta_seconds < min_time_delta_seconds:
                    continue
                rows.append(
                    {
                        "query_asset_id": str(query_row["asset_id"]),
                        "candidate_asset_id": str(candidate_row["asset_id"]),
                        "query_scene_id": str(query_row["scene_id"]),
                        "candidate_scene_id": str(candidate_row["scene_id"]),
                        "query_capture_id": optional_string(query_row.get("capture_id")),
                        "candidate_capture_id": optional_string(candidate_row.get("capture_id")),
                        "city": str(query_row["city"]),
                        "modality": str(query_row["modality"]),
                        "time_delta_seconds": time_delta_seconds,
                    }
                )
                if limit_asset_pairs is not None and len(rows) >= limit_asset_pairs:
                    return pd.DataFrame(rows)
    return pd.DataFrame(
        rows,
        columns=[
            "query_asset_id",
            "candidate_asset_id",
            "query_scene_id",
            "candidate_scene_id",
            "query_capture_id",
            "candidate_capture_id",
            "city",
            "modality",
            "time_delta_seconds",
        ],
    )


def geolocate_chip_rois(
    chips: pd.DataFrame,
    scenes: pd.DataFrame,
    *,
    asset_pairs: pd.DataFrame,
    city_origins: dict[str, LocalOrigin],
    gdal_prefix: Path | None,
    max_chips_per_asset: int | None,
) -> pd.DataFrame:
    if asset_pairs.empty:
        return pd.DataFrame()
    gdal = load_gdal(gdal_prefix)
    asset_city = scenes.set_index("asset_id")["city"].astype(str).to_dict()
    asset_ids = sorted(
        set(asset_pairs["query_asset_id"].astype(str)).union(asset_pairs["candidate_asset_id"].astype(str))
    )
    asset_chip_frames = {
        asset_id: frame.sort_values(["chip_id"]).reset_index(drop=True)
        for asset_id, frame in chips.groupby("asset_id", sort=False)
        if asset_id in asset_ids
    }

    roi_rows: list[dict[str, object]] = []
    for asset_id in asset_ids:
        asset_frame = asset_chip_frames.get(asset_id)
        if asset_frame is None or asset_frame.empty:
            continue
        if max_chips_per_asset is not None:
            asset_frame = asset_frame.head(max_chips_per_asset).reset_index(drop=True)
        local_path = Path(str(asset_frame.iloc[0]["local_path"])).expanduser().resolve()
        city = asset_city.get(str(asset_id))
        origin = city_origins.get(str(city))
        if origin is None:
            continue

        dataset = gdal.Open(str(local_path), gdal.GA_ReadOnly)
        if dataset is None:
            continue
        transformer = make_wgs84_transformer(gdal, dataset)
        try:
            for record in asset_frame.to_dict("records"):
                roi = chip_ground_roi(record, transformer, origin)
                if roi is None:
                    continue
                roi_rows.append(roi)
        finally:
            dataset = None
            transformer = None
    return pd.DataFrame(roi_rows)


def make_wgs84_transformer(gdal: Any, dataset: Any) -> Any:
    options = ["DST_SRS=EPSG:4326"]
    if dataset.GetMetadata("RPC"):
        options.append("METHOD=RPC")
    transformer = gdal.Transformer(dataset, None, options)
    if transformer is None:
        raise RuntimeError("Unable to build GDAL transformer for chip geolocation.")
    return transformer


def chip_ground_roi(
    record: dict[str, object],
    transformer: Any,
    origin: LocalOrigin,
) -> dict[str, object] | None:
    x0 = as_float(record["x0"])
    y0 = as_float(record["y0"])
    x1 = as_float(record["x1"])
    y1 = as_float(record["y1"])
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    lon_lat: list[tuple[float, float]] = []
    metric_xy: list[tuple[float, float]] = []

    for pixel_x, pixel_y in corners:
        ok, (lon_deg, lat_deg, _height) = transformer.TransformPoint(False, pixel_x, pixel_y, 0.0)
        if not ok:
            return None
        lon_lat.append((float(lon_deg), float(lat_deg)))
        metric_xy.append(geodetic_to_local_xy(float(lon_deg), float(lat_deg), origin))

    metric_polygon = Polygon(metric_xy)
    wgs84_polygon = Polygon(lon_lat)
    if metric_polygon.is_empty or not metric_polygon.is_valid or metric_polygon.area <= 0.0:
        return None

    center_x = as_float(record["x0"]) + as_float(record["width"]) / 2.0
    center_y = as_float(record["y0"]) + as_float(record["height"]) / 2.0
    ok, (center_lon, center_lat, _center_height) = transformer.TransformPoint(
        False, center_x, center_y, 0.0
    )
    if not ok:
        return None
    center_east_m, center_north_m = geodetic_to_local_xy(
        float(center_lon),
        float(center_lat),
        origin,
    )

    return {
        "chip_id": str(record["chip_id"]),
        "asset_id": str(record["asset_id"]),
        "capture_id": optional_string(record.get("capture_id")),
        "scene_id": str(record["scene_id"]),
        "split": str(record["split"]),
        "city": str(record["city"]),
        "modality": str(record["modality"]),
        "sensor": optional_string(record.get("sensor")),
        "local_path": str(record["local_path"]),
        "acq_time": record.get("acq_time"),
        "x0": as_int(record["x0"]),
        "y0": as_int(record["y0"]),
        "x1": as_int(record["x1"]),
        "y1": as_int(record["y1"]),
        "width": as_int(record["width"]),
        "height": as_int(record["height"]),
        "center_lon_deg": float(center_lon),
        "center_lat_deg": float(center_lat),
        "center_east_m": center_east_m,
        "center_north_m": center_north_m,
        "area_m2": float(metric_polygon.area),
        "footprint_wgs84_wkb": wgs84_polygon.wkb,
        "footprint_metric_wkb": metric_polygon.wkb,
    }


def build_pairs(
    chip_rois: pd.DataFrame,
    asset_pairs: pd.DataFrame,
    *,
    positive_overlap_fraction: float,
    weak_overlap_fraction: float,
    hard_negative_radius_m: float,
    max_positives_per_query: int,
    max_hard_negatives_per_query: int,
) -> pd.DataFrame:
    if chip_rois.empty or asset_pairs.empty:
        return pd.DataFrame()

    chips_by_asset = {
        asset_id: frame.reset_index(drop=True)
        for asset_id, frame in chip_rois.groupby("asset_id", sort=False)
    }

    pair_rows: list[dict[str, object]] = []
    for asset_pair in asset_pairs.to_dict("records"):
        query_asset = chips_by_asset.get(str(asset_pair["query_asset_id"]))
        candidate_asset = chips_by_asset.get(str(asset_pair["candidate_asset_id"]))
        if query_asset is None or candidate_asset is None:
            continue
        pair_rows.extend(
            mine_directional_pairs(
                query_asset,
                candidate_asset,
                asset_pair=asset_pair,
                positive_overlap_fraction=positive_overlap_fraction,
                weak_overlap_fraction=weak_overlap_fraction,
                hard_negative_radius_m=hard_negative_radius_m,
                max_positives_per_query=max_positives_per_query,
                max_hard_negatives_per_query=max_hard_negatives_per_query,
            )
        )
        pair_rows.extend(
            mine_directional_pairs(
                candidate_asset,
                query_asset,
                asset_pair={
                    **asset_pair,
                    "query_asset_id": asset_pair["candidate_asset_id"],
                    "candidate_asset_id": asset_pair["query_asset_id"],
                    "query_scene_id": asset_pair["candidate_scene_id"],
                    "candidate_scene_id": asset_pair["query_scene_id"],
                    "query_capture_id": asset_pair["candidate_capture_id"],
                    "candidate_capture_id": asset_pair["query_capture_id"],
                },
                positive_overlap_fraction=positive_overlap_fraction,
                weak_overlap_fraction=weak_overlap_fraction,
                hard_negative_radius_m=hard_negative_radius_m,
                max_positives_per_query=max_positives_per_query,
                max_hard_negatives_per_query=max_hard_negatives_per_query,
            )
        )
    return pd.DataFrame(pair_rows)


def mine_directional_pairs(
    query_frame: pd.DataFrame,
    candidate_frame: pd.DataFrame,
    *,
    asset_pair: dict[str, object],
    positive_overlap_fraction: float,
    weak_overlap_fraction: float,
    hard_negative_radius_m: float,
    max_positives_per_query: int,
    max_hard_negatives_per_query: int,
) -> list[dict[str, object]]:
    candidate_polygons = [from_wkb(value) for value in candidate_frame["footprint_metric_wkb"]]
    candidate_centers = candidate_frame[["center_east_m", "center_north_m"]].to_numpy(
        dtype=np.float64
    )
    tree = STRtree(candidate_polygons)

    pair_rows: list[dict[str, object]] = []
    for query_row in query_frame.to_dict("records"):
        query_polygon = from_wkb(query_row["footprint_metric_wkb"])
        query_area = float(query_row["area_m2"])
        query_center = np.array(
            [float(query_row["center_east_m"]), float(query_row["center_north_m"])],
            dtype=np.float64,
        )

        exact_candidates: list[dict[str, object]] = []
        weak_candidates: list[dict[str, object]] = []
        hard_negative_candidates: list[dict[str, object]] = []
        positive_candidate_indices: set[int] = set()

        for candidate_index in tree.query(query_polygon, predicate="intersects"):
            candidate_index = int(candidate_index)
            candidate_polygon = candidate_polygons[candidate_index]
            intersection_area = float(query_polygon.intersection(candidate_polygon).area)
            if intersection_area <= 0.0:
                continue
            candidate_row = candidate_frame.iloc[candidate_index]
            candidate_area = float(candidate_row["area_m2"])
            overlap_fraction = intersection_area / min(query_area, candidate_area)
            union_area = query_area + candidate_area - intersection_area
            overlap_iou = intersection_area / union_area if union_area > 0.0 else 0.0
            center_distance_m = metric_distance(query_center, candidate_centers[candidate_index])
            candidate = build_pair_row(
                query_row,
                candidate_row.to_dict(),
                asset_pair=asset_pair,
                pair_label=(
                    "positive_exact"
                    if overlap_fraction >= positive_overlap_fraction
                    else "positive_weak"
                ),
                overlap_fraction=overlap_fraction,
                overlap_iou=overlap_iou,
                center_distance_m=center_distance_m,
            )
            if overlap_fraction >= positive_overlap_fraction:
                exact_candidates.append(candidate)
                positive_candidate_indices.add(candidate_index)
            elif overlap_fraction >= weak_overlap_fraction:
                weak_candidates.append(candidate)
                positive_candidate_indices.add(candidate_index)

        nearby_query = query_polygon.buffer(hard_negative_radius_m)
        for candidate_index in tree.query(nearby_query):
            candidate_index = int(candidate_index)
            if candidate_index in positive_candidate_indices:
                continue
            candidate_row = candidate_frame.iloc[candidate_index]
            center_distance_m = metric_distance(query_center, candidate_centers[candidate_index])
            if center_distance_m > hard_negative_radius_m:
                continue
            candidate_polygon = candidate_polygons[candidate_index]
            intersection_area = float(query_polygon.intersection(candidate_polygon).area)
            candidate_area = float(candidate_row["area_m2"])
            overlap_fraction = intersection_area / min(query_area, candidate_area)
            if overlap_fraction >= weak_overlap_fraction:
                continue
            union_area = query_area + candidate_area - intersection_area
            overlap_iou = intersection_area / union_area if union_area > 0.0 else 0.0
            hard_negative_candidates.append(
                build_pair_row(
                    query_row,
                    candidate_row.to_dict(),
                    asset_pair=asset_pair,
                    pair_label="negative_hard",
                    overlap_fraction=overlap_fraction,
                    overlap_iou=overlap_iou,
                    center_distance_m=center_distance_m,
                )
            )

        exact_candidates.sort(
            key=lambda row: (-float(row["overlap_fraction"]), float(row["center_distance_m"]))
        )
        weak_candidates.sort(
            key=lambda row: (-float(row["overlap_fraction"]), float(row["center_distance_m"]))
        )
        hard_negative_candidates.sort(key=lambda row: float(row["center_distance_m"]))

        pair_rows.extend(exact_candidates[:max_positives_per_query])
        pair_rows.extend(weak_candidates[:max_positives_per_query])
        pair_rows.extend(hard_negative_candidates[:max_hard_negatives_per_query])

    return pair_rows


def build_pair_row(
    query_row: dict[str, object],
    candidate_row: dict[str, object],
    *,
    asset_pair: dict[str, object],
    pair_label: str,
    overlap_fraction: float,
    overlap_iou: float,
    center_distance_m: float,
) -> dict[str, object]:
    return {
        "query_chip_id": str(query_row["chip_id"]),
        "candidate_chip_id": str(candidate_row["chip_id"]),
        "query_asset_id": str(query_row["asset_id"]),
        "candidate_asset_id": str(candidate_row["asset_id"]),
        "query_scene_id": str(query_row["scene_id"]),
        "candidate_scene_id": str(candidate_row["scene_id"]),
        "query_capture_id": optional_string(query_row.get("capture_id")),
        "candidate_capture_id": optional_string(candidate_row.get("capture_id")),
        "query_split": str(query_row["split"]),
        "candidate_split": str(candidate_row["split"]),
        "city": str(query_row["city"]),
        "modality": str(query_row["modality"]),
        "query_sensor": optional_string(query_row.get("sensor")),
        "candidate_sensor": optional_string(candidate_row.get("sensor")),
        "query_acq_time": query_row.get("acq_time"),
        "candidate_acq_time": candidate_row.get("acq_time"),
        "pair_label": pair_label,
        "pair_group": "positive" if pair_label.startswith("positive") else "negative",
        "time_delta_seconds": as_float(asset_pair["time_delta_seconds"]),
        "overlap_fraction": float(overlap_fraction),
        "overlap_iou": float(overlap_iou),
        "center_distance_m": float(center_distance_m),
    }


def geodetic_to_ecef(
    lon_deg: float,
    lat_deg: float,
    height_m: float = 0.0,
) -> tuple[float, float, float]:
    lon_rad = math.radians(lon_deg)
    lat_rad = math.radians(lat_deg)
    sin_lat = math.sin(lat_rad)
    cos_lat = math.cos(lat_rad)
    sin_lon = math.sin(lon_rad)
    cos_lon = math.cos(lon_rad)
    radius = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    x = (radius + height_m) * cos_lat * cos_lon
    y = (radius + height_m) * cos_lat * sin_lon
    z = (radius * (1.0 - WGS84_E2) + height_m) * sin_lat
    return x, y, z


def geodetic_to_local_xy(
    lon_deg: float,
    lat_deg: float,
    origin: LocalOrigin,
) -> tuple[float, float]:
    x, y, z = geodetic_to_ecef(lon_deg, lat_deg)
    dx = x - origin.ecef_x
    dy = y - origin.ecef_y
    dz = z - origin.ecef_z

    lon0 = math.radians(origin.lon_deg)
    lat0 = math.radians(origin.lat_deg)
    sin_lon0 = math.sin(lon0)
    cos_lon0 = math.cos(lon0)
    sin_lat0 = math.sin(lat0)
    cos_lat0 = math.cos(lat0)

    east = -sin_lon0 * dx + cos_lon0 * dy
    north = -sin_lat0 * cos_lon0 * dx - sin_lat0 * sin_lon0 * dy + cos_lat0 * dz
    return float(east), float(north)


def metric_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def timestamp_delta_seconds(a: object, b: object) -> float:
    if a is None or b is None or pd.isna(a) or pd.isna(b):
        return 0.0
    a_ts = to_timestamp(a)
    b_ts = to_timestamp(b)
    return float(abs((a_ts - b_ts).total_seconds()))


def optional_string(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    return str(value)


def as_float(value: object) -> float:
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise TypeError(f"Value is not numeric: {value!r}")


def as_int(value: object) -> int:
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating, str)):
        return int(value)
    raise TypeError(f"Value is not integer-like: {value!r}")


def to_timestamp(value: object) -> pd.Timestamp:
    if isinstance(value, pd.Timestamp):
        return value
    if isinstance(value, np.datetime64):
        return pd.Timestamp(value)
    if isinstance(value, (datetime, date, int, float, str)):
        return pd.Timestamp(value)
    raise TypeError(f"Value is not timestamp-like: {value!r}")


def write_outputs(
    *,
    chip_rois: pd.DataFrame,
    asset_pairs: pd.DataFrame,
    pairs: pd.DataFrame,
    summary: PairMiningSummary,
    output_root: Path,
) -> dict[str, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    chip_rois_path = output_root / "chip_rois.parquet"
    asset_pairs_path = output_root / "asset_pairs.parquet"
    pairs_path = output_root / "pairs.parquet"
    summary_path = output_root / "summary.json"

    chip_rois.to_parquet(chip_rois_path, index=False, compression="zstd")
    asset_pairs.to_parquet(asset_pairs_path, index=False, compression="zstd")
    pairs.to_parquet(pairs_path, index=False, compression="zstd")
    summary_path.write_text(json.dumps(asdict(summary), indent=2, sort_keys=True), encoding="utf-8")
    return {
        "chip_rois": chip_rois_path,
        "asset_pairs": asset_pairs_path,
        "pairs": pairs_path,
        "summary": summary_path,
    }


def build_summary(
    *,
    scenes: pd.DataFrame,
    chips: pd.DataFrame,
    chip_rois: pd.DataFrame,
    asset_pairs: pd.DataFrame,
    pairs: pd.DataFrame,
    elapsed_seconds: float,
) -> PairMiningSummary:
    by_label = (
        pairs["pair_label"].value_counts().sort_index()
        if not pairs.empty
        else pd.Series(dtype="int64")
    )
    by_city = (
        pairs["city"].value_counts().sort_index()
        if not pairs.empty
        else pd.Series(dtype="int64")
    )
    by_modality = (
        pairs["modality"].value_counts().sort_index()
        if not pairs.empty
        else pd.Series(dtype="int64")
    )
    by_query_split = (
        pairs["query_split"].value_counts().sort_index()
        if not pairs.empty
        else pd.Series(dtype="int64")
    )
    by_candidate_split = (
        pairs["candidate_split"].value_counts().sort_index()
        if not pairs.empty
        else pd.Series(dtype="int64")
    )
    by_split_pair_label = (
        pairs.groupby(["query_split", "candidate_split", "pair_label"]).size().sort_index()
        if not pairs.empty
        else pd.Series(dtype="int64")
    )
    split_pair_counts: dict[str, int] = {}
    for key, value in by_split_pair_label.items():
        if not isinstance(key, tuple) or len(key) != 3:
            continue
        if not isinstance(value, (int, np.integer)):
            continue
        query_split, candidate_split, pair_label = key
        split_pair_counts[
            f"{str(query_split)}->{str(candidate_split)}:{str(pair_label)}"
        ] = int(value)
    return PairMiningSummary(
        scenes=int(len(scenes)),
        chips=int(len(chips)),
        chip_rois=int(len(chip_rois)),
        asset_pairs=int(len(asset_pairs)),
        pair_rows=int(len(pairs)),
        elapsed_seconds=float(elapsed_seconds),
        by_label={str(key): int(value) for key, value in by_label.items()},
        by_city={str(key): int(value) for key, value in by_city.items()},
        by_modality={str(key): int(value) for key, value in by_modality.items()},
        by_query_split={str(key): int(value) for key, value in by_query_split.items()},
        by_candidate_split={str(key): int(value) for key, value in by_candidate_split.items()},
        by_split_pair_label=split_pair_counts,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    chips_path = args.chips_path.resolve()
    scenes_path = resolve_scenes_path(
        chips_path,
        args.scenes_path.resolve() if args.scenes_path else None,
    )
    output_root = args.output_root.resolve()
    gdal_prefix = args.gdal_prefix.resolve() if args.gdal_prefix else None

    start = perf_counter()
    chips, scenes = load_frames(chips_path, scenes_path)
    chips = filter_frame(chips, args)
    scenes = filter_frame(scenes, args)
    if args.limit_assets is not None:
        scenes = (
            scenes.sort_values(["city", "modality", "asset_id"])
            .head(args.limit_assets)
            .reset_index(drop=True)
        )
    selected_assets = set(scenes["asset_id"].astype(str))
    chips = chips[chips["asset_id"].isin(selected_assets)].reset_index(drop=True)

    city_origins = build_city_origins(scenes)
    asset_pairs = find_asset_pairs(
        scenes,
        require_different_capture=args.require_different_capture,
        min_time_delta_seconds=args.min_time_delta_seconds,
        limit_asset_pairs=args.limit_asset_pairs,
    )
    chip_rois = geolocate_chip_rois(
        chips,
        scenes,
        asset_pairs=asset_pairs,
        city_origins=city_origins,
        gdal_prefix=gdal_prefix,
        max_chips_per_asset=args.max_chips_per_asset,
    )
    pairs = build_pairs(
        chip_rois,
        asset_pairs,
        positive_overlap_fraction=args.positive_overlap_fraction,
        weak_overlap_fraction=args.weak_overlap_fraction,
        hard_negative_radius_m=args.hard_negative_radius_m,
        max_positives_per_query=args.max_positives_per_query,
        max_hard_negatives_per_query=args.max_hard_negatives_per_query,
    )
    elapsed_seconds = perf_counter() - start
    summary = build_summary(
        scenes=scenes,
        chips=chips,
        chip_rois=chip_rois,
        asset_pairs=asset_pairs,
        pairs=pairs,
        elapsed_seconds=elapsed_seconds,
    )
    paths = write_outputs(
        chip_rois=chip_rois,
        asset_pairs=asset_pairs,
        pairs=pairs,
        summary=summary,
        output_root=output_root,
    )

    print(f"chips: {chips_path}")
    print(f"scenes: {scenes_path}")
    print(f"scenes_filtered: {len(scenes):,}")
    print(f"chips_filtered: {len(chips):,}")
    print(f"chip_rois: {len(chip_rois):,} -> {paths['chip_rois']}")
    print(f"asset_pairs: {len(asset_pairs):,} -> {paths['asset_pairs']}")
    print(f"pairs: {len(pairs):,} -> {paths['pairs']}")
    print(f"summary: {paths['summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
