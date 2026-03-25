from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Protocol

import numpy as np

from geogrok.data.training import (
    TrainingBatch,
    TrainingSample,
    collate_training_samples,
)


@dataclass(frozen=True)
class EpochMetrics:
    stage: str
    epoch: int
    batches: int
    samples: int
    elapsed_seconds: float
    total_pixels: int
    total_bytes: int
    samples_per_second: float
    megapixels_per_second: float
    mebibytes_per_second: float
    read_latency_ms_mean: float
    read_latency_ms_p50: float
    read_latency_ms_p95: float
    transform_latency_ms_mean: float
    transform_latency_ms_p50: float
    transform_latency_ms_p95: float
    sample_total_ms_mean: float
    sample_total_ms_p50: float
    sample_total_ms_p95: float
    batch_collate_ms_mean: float
    batch_collate_ms_p50: float
    batch_collate_ms_p95: float


class TrainingDatasetProtocol(Protocol):
    def __len__(self) -> int: ...

    def sample(self, index: int) -> TrainingSample: ...


def epoch_indices(
    dataset_size: int,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    epoch: int,
    drop_last: bool = False,
) -> list[list[int]]:
    if dataset_size < 0:
        raise ValueError("dataset_size cannot be negative.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if dataset_size == 0:
        return []

    indices = list(range(dataset_size))
    if shuffle:
        rng = np.random.default_rng(seed + epoch)
        rng.shuffle(indices)

    batches = [
        indices[start : start + batch_size] for start in range(0, len(indices), batch_size)
    ]
    if drop_last and batches and len(batches[-1]) < batch_size:
        batches = batches[:-1]
    return batches


def iterate_training_batches(
    dataset: TrainingDatasetProtocol,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    epoch: int,
    drop_last: bool = False,
) -> Iterator[tuple[list[int], TrainingBatch]]:
    for batch_indices in epoch_indices(
        len(dataset),
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        epoch=epoch,
        drop_last=drop_last,
    ):
        samples = [dataset.sample(index) for index in batch_indices]
        batch = collate_training_samples(samples)
        yield batch_indices, batch


def run_epoch(
    dataset: TrainingDatasetProtocol,
    *,
    stage: str,
    epoch: int,
    batch_size: int,
    shuffle: bool,
    seed: int,
    drop_last: bool = False,
    max_batches: int | None = None,
) -> EpochMetrics:
    read_latencies: list[float] = []
    transform_latencies: list[float] = []
    sample_total_latencies: list[float] = []
    collate_latencies: list[float] = []
    samples_seen = 0
    total_pixels = 0
    total_bytes = 0
    batches_seen = 0

    batch_index_groups = epoch_indices(
        len(dataset),
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        epoch=epoch,
        drop_last=drop_last,
    )
    if max_batches is not None:
        batch_index_groups = batch_index_groups[:max_batches]

    start = perf_counter()
    for batch_indices in batch_index_groups:
        sample_start = perf_counter()
        samples = [dataset.sample(index) for index in batch_indices]
        sample_elapsed_ms = (perf_counter() - sample_start) * 1000.0

        collate_start = perf_counter()
        batch = collate_training_samples(samples)
        collate_elapsed_ms = (perf_counter() - collate_start) * 1000.0

        collate_latencies.append(collate_elapsed_ms)
        samples_seen += len(samples)
        batches_seen += 1
        total_pixels += int(batch.images.size)
        total_bytes += int(batch.images.nbytes)

        read_latencies.extend(sample.timing.read_ms for sample in samples)
        transform_latencies.extend(sample.timing.transform_ms for sample in samples)
        sample_total_latencies.extend(sample.timing.total_ms for sample in samples)

        # Track full per-batch acquisition time so collation overhead can be compared
        # against accumulated sample timings during profiling.
        _ = sample_elapsed_ms

    elapsed_seconds = perf_counter() - start
    return EpochMetrics(
        stage=stage,
        epoch=epoch,
        batches=batches_seen,
        samples=samples_seen,
        elapsed_seconds=elapsed_seconds,
        total_pixels=total_pixels,
        total_bytes=total_bytes,
        samples_per_second=safe_rate(samples_seen, elapsed_seconds),
        megapixels_per_second=safe_rate(total_pixels / 1_000_000.0, elapsed_seconds),
        mebibytes_per_second=safe_rate(total_bytes / (1024.0 * 1024.0), elapsed_seconds),
        read_latency_ms_mean=mean(read_latencies),
        read_latency_ms_p50=percentile(read_latencies, 50.0),
        read_latency_ms_p95=percentile(read_latencies, 95.0),
        transform_latency_ms_mean=mean(transform_latencies),
        transform_latency_ms_p50=percentile(transform_latencies, 50.0),
        transform_latency_ms_p95=percentile(transform_latencies, 95.0),
        sample_total_ms_mean=mean(sample_total_latencies),
        sample_total_ms_p50=percentile(sample_total_latencies, 50.0),
        sample_total_ms_p95=percentile(sample_total_latencies, 95.0),
        batch_collate_ms_mean=mean(collate_latencies),
        batch_collate_ms_p50=percentile(collate_latencies, 50.0),
        batch_collate_ms_p95=percentile(collate_latencies, 95.0),
    )


def write_metrics_jsonl(metrics: Sequence[EpochMetrics], path: Path) -> Path:
    output_path = path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for metric in metrics:
            handle.write(json.dumps(asdict(metric), sort_keys=True))
            handle.write("\n")
    return output_path


def percentile(values: Sequence[float], percentile_rank: float) -> float:
    if not values:
        return 0.0
    if percentile_rank < 0.0 or percentile_rank > 100.0:
        raise ValueError("percentile_rank must be in [0, 100].")

    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * (percentile_rank / 100.0)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(float(value) for value in values) / len(values)


def safe_rate(value: float, seconds: float) -> float:
    if seconds <= 0.0:
        return 0.0
    return value / seconds
