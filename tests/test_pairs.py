from __future__ import annotations

import math

import pandas as pd
from shapely.geometry import Polygon

from geogrok.data.pairs import (
    LocalOrigin,
    build_pairs,
    geodetic_to_ecef,
    geodetic_to_local_xy,
)


def test_geodetic_to_ecef_is_finite():
    x, y, z = geodetic_to_ecef(138.6007, -34.9285)
    assert math.isfinite(x)
    assert math.isfinite(y)
    assert math.isfinite(z)


def test_geodetic_to_local_xy_maps_origin_close_to_zero():
    lon_deg = 138.6007
    lat_deg = -34.9285
    ecef_x, ecef_y, ecef_z = geodetic_to_ecef(lon_deg, lat_deg)
    origin = LocalOrigin(
        city="Adelaide",
        lon_deg=lon_deg,
        lat_deg=lat_deg,
        ecef_x=ecef_x,
        ecef_y=ecef_y,
        ecef_z=ecef_z,
    )
    east, north = geodetic_to_local_xy(lon_deg, lat_deg, origin)
    assert abs(east) < 1e-6
    assert abs(north) < 1e-6


def test_build_pairs_emits_positive_and_hard_negative_rows():
    query_polygon = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
    positive_polygon = Polygon([(20, 0), (120, 0), (120, 100), (20, 100)])
    negative_polygon = Polygon([(300, 0), (400, 0), (400, 100), (300, 100)])

    chip_rois = pd.DataFrame(
        [
            {
                "chip_id": "q1",
                "asset_id": "asset_a",
                "capture_id": "cap_a",
                "scene_id": "scene_a",
                "split": "train",
                "city": "X",
                "modality": "PAN",
                "sensor": "WV3",
                "local_path": "/tmp/a",
                "acq_time": None,
                "x0": 0,
                "y0": 0,
                "x1": 1,
                "y1": 1,
                "width": 1,
                "height": 1,
                "center_lon_deg": 0.0,
                "center_lat_deg": 0.0,
                "center_east_m": 50.0,
                "center_north_m": 50.0,
                "area_m2": query_polygon.area,
                "footprint_wgs84_wkb": query_polygon.wkb,
                "footprint_metric_wkb": query_polygon.wkb,
            },
            {
                "chip_id": "p1",
                "asset_id": "asset_b",
                "capture_id": "cap_b",
                "scene_id": "scene_b",
                "split": "val",
                "city": "X",
                "modality": "PAN",
                "sensor": "WV3",
                "local_path": "/tmp/b",
                "acq_time": None,
                "x0": 0,
                "y0": 0,
                "x1": 1,
                "y1": 1,
                "width": 1,
                "height": 1,
                "center_lon_deg": 0.0,
                "center_lat_deg": 0.0,
                "center_east_m": 70.0,
                "center_north_m": 50.0,
                "area_m2": positive_polygon.area,
                "footprint_wgs84_wkb": positive_polygon.wkb,
                "footprint_metric_wkb": positive_polygon.wkb,
            },
            {
                "chip_id": "n1",
                "asset_id": "asset_b",
                "capture_id": "cap_b",
                "scene_id": "scene_b",
                "split": "val",
                "city": "X",
                "modality": "PAN",
                "sensor": "WV3",
                "local_path": "/tmp/b",
                "acq_time": None,
                "x0": 0,
                "y0": 0,
                "x1": 1,
                "y1": 1,
                "width": 1,
                "height": 1,
                "center_lon_deg": 0.0,
                "center_lat_deg": 0.0,
                "center_east_m": 350.0,
                "center_north_m": 50.0,
                "area_m2": negative_polygon.area,
                "footprint_wgs84_wkb": negative_polygon.wkb,
                "footprint_metric_wkb": negative_polygon.wkb,
            },
        ]
    )
    asset_pairs = pd.DataFrame(
        [
            {
                "query_asset_id": "asset_a",
                "candidate_asset_id": "asset_b",
                "query_scene_id": "scene_a",
                "candidate_scene_id": "scene_b",
                "query_capture_id": "cap_a",
                "candidate_capture_id": "cap_b",
                "city": "X",
                "modality": "PAN",
                "time_delta_seconds": 100.0,
            }
        ]
    )

    pairs = build_pairs(
        chip_rois,
        asset_pairs,
        positive_overlap_fraction=0.5,
        weak_overlap_fraction=0.2,
        hard_negative_radius_m=500.0,
        max_positives_per_query=4,
        max_hard_negatives_per_query=4,
    )

    labels = set(pairs["pair_label"].tolist())
    assert "positive_exact" in labels
    assert "negative_hard" in labels
