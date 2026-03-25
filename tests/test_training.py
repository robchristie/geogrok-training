from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from geogrok.data.runtime import OnDemandChipDataset
from geogrok.data.training import (
    TrainingChipDataset,
    benchmark_training_dataset,
    collate_training_samples,
    preprocess_image,
)


def test_preprocess_image_scales_and_casts():
    array = np.array([[[0, 1024], [2047, 4095]]], dtype=np.uint16)

    image = preprocess_image(
        array,
        output_dtype=np.dtype("float32"),
        clip_min=0.0,
        clip_max=2047.0,
        scale_max=2047.0,
    )

    assert image.dtype == np.float32
    assert float(image.min()) == 0.0
    assert np.isclose(float(image.max()), 1.0)


def test_training_chip_dataset_collates_samples():
    chips = pd.DataFrame(
        [
            {
                "chip_id": "chip_a",
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
                "width": 8,
                "height": 8,
            },
            {
                "chip_id": "chip_b",
                "asset_id": "asset_2",
                "capture_id": "capture_2",
                "scene_id": "scene_2",
                "split": "train",
                "city": "Jacksonville",
                "modality": "PAN",
                "sensor": "WV3",
                "local_path": "/tmp/b.ntf",
                "local_exists": True,
                "x0": 0,
                "y0": 0,
                "width": 8,
                "height": 8,
            },
        ]
    )

    class FakeOnDemandDataset(OnDemandChipDataset):
        def __init__(self) -> None:
            super().__init__(chips)

        def sample(self, index: int):
            record = self.record(index)
            array = np.full((1, 8, 8), fill_value=index + 1, dtype=np.uint16)

            class FakeChip:
                def __init__(self, local_path: Path, payload: np.ndarray) -> None:
                    self.path = local_path
                    self.array = payload

            class FakeSample:
                def __init__(self, record, chip) -> None:
                    self.record = record
                    self.chip = chip

            return FakeSample(record, FakeChip(record.local_path, array))

    dataset = TrainingChipDataset(FakeOnDemandDataset(), output_dtype="float32")
    batch = collate_training_samples([dataset.sample(0), dataset.sample(1)])

    assert batch.images.shape == (2, 1, 8, 8)
    assert batch.images.dtype == np.float32
    assert batch.records[0].chip_id == "chip_a"


def test_benchmark_training_dataset_reports_phase_metrics():
    chips = pd.DataFrame(
        [
            {
                "chip_id": "chip_a",
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
                "width": 4,
                "height": 4,
            }
        ]
    )

    class FakeOnDemandDataset(OnDemandChipDataset):
        def __init__(self) -> None:
            super().__init__(chips)

        def sample(self, index: int):
            record = self.record(index)
            array = np.full((1, 4, 4), fill_value=index + 1, dtype=np.uint16)

            class FakeChip:
                def __init__(self, local_path: Path, payload: np.ndarray) -> None:
                    self.path = local_path
                    self.array = payload

            class FakeSample:
                def __init__(self, record, chip) -> None:
                    self.record = record
                    self.chip = chip

            return FakeSample(record, FakeChip(record.local_path, array))

    dataset = TrainingChipDataset(
        FakeOnDemandDataset(),
        output_dtype="float32",
        scale_max=2047.0,
    )
    report = benchmark_training_dataset(dataset, repeat=3, warmup=1)

    assert report.samples == 3
    assert report.read_latency_ms_mean >= 0.0
    assert report.transform_latency_ms_mean >= 0.0
    assert report.total_latency_ms_mean >= 0.0
    assert report.samples_per_second > 0.0
