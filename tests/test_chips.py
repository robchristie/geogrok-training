from __future__ import annotations

from pathlib import Path

import pandas as pd

from geogrok.data.chips import build_chip_output_path, select_chip_rows


def test_select_chip_rows_filters_to_local_split_and_modality():
    chips = pd.DataFrame(
        [
            {
                "chip_id": "chip_train_pan",
                "asset_id": "asset_1",
                "scene_id": "scene_1",
                "split": "train",
                "city": "Jacksonville",
                "modality": "PAN",
                "local_path": "/tmp/a.ntf",
                "local_exists": True,
                "x0": 0,
                "y0": 0,
                "width": 1024,
                "height": 1024,
            },
            {
                "chip_id": "chip_val_pan_missing",
                "asset_id": "asset_2",
                "scene_id": "scene_2",
                "split": "val",
                "city": "Omaha",
                "modality": "PAN",
                "local_path": "/tmp/b.ntf",
                "local_exists": False,
                "x0": 0,
                "y0": 0,
                "width": 1024,
                "height": 1024,
            },
            {
                "chip_id": "chip_train_ms",
                "asset_id": "asset_3",
                "scene_id": "scene_3",
                "split": "train",
                "city": "Jacksonville",
                "modality": "MS",
                "local_path": "/tmp/c.tif",
                "local_exists": True,
                "x0": 0,
                "y0": 0,
                "width": 1024,
                "height": 1024,
            },
        ]
    )

    selected = select_chip_rows(chips, splits=("train",), modalities=("PAN",), limit=None)

    assert list(selected["chip_id"]) == ["chip_train_pan"]


def test_build_chip_output_path_groups_by_split_modality_and_city():
    record = {
        "chip_id": "chip_123",
        "split": "train",
        "modality": "PAN",
        "city": "Las Vegas",
    }

    output_path = build_chip_output_path(record, Path("/tmp/chips"))

    assert output_path == Path("/tmp/chips/train/PAN/Las_Vegas/chip_123.tif")
