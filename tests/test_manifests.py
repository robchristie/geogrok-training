from __future__ import annotations

import pandas as pd

from geogrok.data.manifests import (
    asset_preference_rank,
    build_asset_manifest,
    build_chip_manifest,
    build_scene_manifest,
    extract_city,
    extract_modality,
    tile_positions,
)


def test_extract_fields_from_core3d_key(tmp_path):
    metadata = pd.DataFrame(
        [
            {
                "key": "Hosted-Datasets/CORE3D-Public-Data/Satellite-Images/Jacksonville/WV3/PAN/"
                "01MAY15WV031200015MAY01160357-P1BS-500648062030_01_P001_________AAE_0AAAAABPABQ0.NTF",
                "scene_id": None,
                "sensor_hint": "WV3",
                "raster_x": 43008,
                "raster_y": 44032,
            }
        ]
    )

    assets = build_asset_manifest(
        metadata,
        download_root=tmp_path,
        val_cities=("Omaha",),
        test_cities=("UCSD",),
        source_metadata_path=tmp_path / "metadata.parquet",
    )

    record = assets.iloc[0].to_dict()
    assert extract_city(record["key"]) == "Jacksonville"
    assert extract_modality(record["key"]) == "PAN"
    assert record["split"] == "train"
    assert record["sensor"] == "WV3"
    assert record["product_code"] == "P1BS"


def test_asset_manifest_resolves_local_path_against_bucket_mirror_root(tmp_path):
    key = (
        "Hosted-Datasets/CORE3D-Public-Data/Satellite-Images/Jacksonville/WV3/PAN/"
        "01MAY15WV031200015MAY01160357-P1BS-500648062030_01_P001_________AAE_0AAAAABPABQ0.NTF"
    )
    local_file = tmp_path / key
    local_file.parent.mkdir(parents=True, exist_ok=True)
    local_file.write_bytes(b"ntf")

    metadata = pd.DataFrame(
        [
            {
                "key": key,
                "scene_id": None,
                "sensor_hint": "WV3",
                "raster_x": 43008,
                "raster_y": 44032,
            }
        ]
    )

    assets = build_asset_manifest(
        metadata,
        download_root=tmp_path,
        val_cities=("Omaha",),
        test_cities=("UCSD",),
        source_metadata_path=tmp_path / "metadata.parquet",
    )

    record = assets.iloc[0]
    assert bool(record["local_exists"])
    assert record["local_path"] == str(local_file.resolve())


def test_scene_manifest_prefers_ntf_before_tif(tmp_path):
    metadata = pd.DataFrame(
        [
            {
                "key": "Hosted-Datasets/CORE3D-Public-Data/Satellite-Images/Jacksonville/"
                "WV3/PAN/a_scene.NTF",
                "scene_id": "scene_a",
                "raster_x": 4096,
                "raster_y": 4096,
            },
            {
                "key": "Hosted-Datasets/CORE3D-Public-Data/Satellite-Images/Jacksonville/"
                "WV3/PAN/a_scene.tif",
                "scene_id": "scene_a",
                "raster_x": 4096,
                "raster_y": 4096,
            },
        ]
    )

    assets = build_asset_manifest(
        metadata,
        download_root=tmp_path,
        val_cities=("Omaha",),
        test_cities=("UCSD",),
        source_metadata_path=tmp_path / "metadata.parquet",
    )
    scenes = build_scene_manifest(assets)

    assert len(scenes) == 1
    assert scenes.iloc[0]["key"].endswith(".NTF")
    assert asset_preference_rank("foo.NTF") < asset_preference_rank("foo.tif")


def test_chip_manifest_tiles_to_image_edge():
    scenes = pd.DataFrame(
        [
            {
                "asset_id": "asset_1",
                "capture_id": "capture_1",
                "scene_id": "scene_1",
                "split": "train",
                "city": "Jacksonville",
                "modality": "PAN",
                "sensor": "WV3",
                "local_path": "/tmp/example.ntf",
                "local_exists": False,
                "remote_uri": "s3://spacenet-dataset/example.ntf",
                "acq_time": None,
                "raster_x": 2500,
                "raster_y": 2500,
            }
        ]
    )

    chips = build_chip_manifest(
        scenes,
        chip_modalities=("PAN",),
        chip_size=1024,
        chip_stride=1024,
    )

    assert tile_positions(2500, 1024, 1024) == [0, 1024, 1476]
    assert len(chips) == 9
    assert chips["x1"].max() == 2500
    assert chips["y1"].max() == 2500
