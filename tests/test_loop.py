from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from geogrok.data.runtime import ChipRecord
from geogrok.data.training import SampleTiming, TrainingSample
from geogrok.training.loop import epoch_indices, run_epoch


@dataclass
class FakeTrainingDataset:
    size: int

    def __len__(self) -> int:
        return self.size

    def sample(self, index: int) -> TrainingSample:
        image = np.full((1, 4, 4), fill_value=index + 1, dtype=np.float32)
        record = ChipRecord(
            chip_id=f"chip_{index}",
            asset_id=f"asset_{index}",
            capture_id=None,
            scene_id=f"scene_{index}",
            split="train",
            city="Jacksonville",
            modality="PAN",
            sensor="WV3",
            local_path=fake_path(index),
            x0=0,
            y0=0,
            width=4,
            height=4,
        )
        timing = SampleTiming(read_ms=1.0 + index, transform_ms=0.5, total_ms=1.5 + index)
        return TrainingSample(record=record, image=image, timing=timing)


def test_epoch_indices_are_deterministic():
    first = epoch_indices(10, batch_size=3, shuffle=True, seed=7, epoch=0)
    second = epoch_indices(10, batch_size=3, shuffle=True, seed=7, epoch=0)
    third = epoch_indices(10, batch_size=3, shuffle=True, seed=7, epoch=1)

    assert first == second
    assert first != third


def test_run_epoch_accumulates_samples_and_batches():
    dataset = FakeTrainingDataset(size=5)

    metrics = run_epoch(
        dataset,
        stage="train",
        epoch=0,
        batch_size=2,
        shuffle=False,
        seed=123,
    )

    assert metrics.stage == "train"
    assert metrics.samples == 5
    assert metrics.batches == 3
    assert metrics.samples_per_second > 0.0
    assert metrics.read_latency_ms_p95 >= metrics.read_latency_ms_p50


def fake_path(index: int):
    from pathlib import Path

    return Path(f"/tmp/{index}.ntf")
