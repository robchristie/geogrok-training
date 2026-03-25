from __future__ import annotations

import numpy as np
import pandas as pd

from geogrok.data.runtime import OnDemandChipDataset, chip_record_from_mapping
from geogrok.io.raster import normalize_chip_array


def test_chip_record_from_mapping_normalizes_fields(tmp_path):
    record = chip_record_from_mapping(
        {
            "chip_id": "chip_1",
            "asset_id": "asset_1",
            "capture_id": None,
            "scene_id": "scene_1",
            "split": "train",
            "city": "Jacksonville",
            "modality": "PAN",
            "sensor": "WV3",
            "local_path": str(tmp_path / "example.ntf"),
            "x0": 0,
            "y0": 1024,
            "width": 512,
            "height": 512,
        }
    )

    assert record.local_path == (tmp_path / "example.ntf").resolve()
    assert record.width == 512
    assert record.height == 512


def test_on_demand_chip_dataset_from_frame_preserves_rows():
    chips = pd.DataFrame(
        [
            {
                "chip_id": "chip_train_pan",
                "asset_id": "asset_1",
                "capture_id": "capture_1",
                "scene_id": "scene_1",
                "split": "train",
                "city": "Jacksonville",
                "modality": "PAN",
                "sensor": "WV3",
                "local_path": "/tmp/a.ntf",
                "local_exists": True,
                "x0": 0,
                "y0": 0,
                "width": 1024,
                "height": 1024,
            }
        ]
    )

    dataset = OnDemandChipDataset(chips)

    assert len(dataset) == 1
    assert dataset.record(0).chip_id == "chip_train_pan"


def test_normalize_chip_array_promotes_single_band_to_chw():
    array = np.arange(16, dtype=np.uint16).reshape(4, 4)

    normalized = normalize_chip_array(array)

    assert normalized.shape == (1, 4, 4)
    assert normalized.dtype == np.uint16
