from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from geogrok.data.benchmark import benchmark_dataset_reads, cycle_indices, percentile
from geogrok.data.runtime import ChipRecord, ChipSample
from geogrok.io.raster import PixelWindow, RasterArrayChip


@dataclass
class FakeDataset:
    size: int

    def __len__(self) -> int:
        return self.size

    def sample(self, index: int) -> ChipSample:
        array = np.full((1, 4, 4), fill_value=index, dtype=np.uint16)
        record = ChipRecord(
            chip_id=f"chip_{index}",
            asset_id=f"asset_{index}",
            capture_id=None,
            scene_id=f"scene_{index}",
            split="train",
            city="Jacksonville",
            modality="PAN",
            sensor="WV3",
            local_path=Path(f"/tmp/{index}.ntf"),
            x0=0,
            y0=0,
            width=4,
            height=4,
        )
        chip = RasterArrayChip(
            path=record.local_path,
            window=PixelWindow(x0=0, y0=0, width=4, height=4),
            band_count=1,
            band_dtypes=("UInt16",),
            array=array,
        )
        return ChipSample(record=record, chip=chip)


def test_cycle_indices_wraps_across_dataset_size():
    assert cycle_indices(dataset_size=3, total=8) == [0, 1, 2, 0, 1, 2, 0, 1]


def test_percentile_interpolates_values():
    values = [10.0, 20.0, 30.0, 40.0]
    assert percentile(values, 50.0) == 25.0
    assert percentile(values, 95.0) == 38.5


def test_benchmark_dataset_reads_reports_nonzero_throughput():
    dataset = FakeDataset(size=3)

    report = benchmark_dataset_reads(dataset, repeat=2, warmup=1)

    assert report.samples == 6
    assert report.warmup_samples == 1
    assert report.unique_source_files == 3
    assert report.total_pixels == 96
    assert report.total_bytes == 192
    assert report.samples_per_second > 0.0
    assert report.megapixels_per_second > 0.0
    assert report.mebibytes_per_second > 0.0
