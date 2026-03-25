from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from geogrok.data.runtime import ChipRecord, OnDemandChipDataset

ImageTransform = Callable[[np.ndarray, ChipRecord], np.ndarray]

DEFAULT_BENCHMARK_OUTPUT = Path("artifacts/benchmarks/training-chip-benchmark.json")


@dataclass(frozen=True)
class SampleTiming:
    read_ms: float
    transform_ms: float
    total_ms: float


@dataclass(frozen=True)
class TrainingSample:
    record: ChipRecord
    image: np.ndarray
    timing: SampleTiming


@dataclass(frozen=True)
class TrainingBatch:
    images: np.ndarray
    records: tuple[ChipRecord, ...]
    timing_ms_total: float


@dataclass(frozen=True)
class TrainingBenchmarkReport:
    samples: int
    warmup_samples: int
    unique_source_files: int
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
    total_latency_ms_mean: float
    total_latency_ms_p50: float
    total_latency_ms_p95: float


class TrainingChipDataset:
    def __init__(
        self,
        dataset: OnDemandChipDataset,
        *,
        output_dtype: str = "float32",
        clip_min: float | None = None,
        clip_max: float | None = None,
        scale_max: float | None = None,
        transforms: Sequence[ImageTransform] | None = None,
    ) -> None:
        self._dataset = dataset
        self._output_dtype = np.dtype(output_dtype)
        self._clip_min = clip_min
        self._clip_max = clip_max
        self._scale_max = scale_max
        self._transforms = tuple(transforms or ())
        validate_preprocessing(
            output_dtype=self._output_dtype,
            clip_min=self._clip_min,
            clip_max=self._clip_max,
            scale_max=self._scale_max,
        )

    @classmethod
    def from_manifest(
        cls,
        chips_path: str | Path,
        *,
        splits: tuple[str, ...] | None = None,
        modalities: tuple[str, ...] | None = None,
        limit: int | None = None,
        gdal_prefix: str | Path | None = None,
        output_dtype: str = "float32",
        clip_min: float | None = None,
        clip_max: float | None = None,
        scale_max: float | None = None,
        transforms: Sequence[ImageTransform] | None = None,
    ) -> TrainingChipDataset:
        dataset = OnDemandChipDataset.from_manifest(
            chips_path,
            splits=splits,
            modalities=modalities,
            limit=limit,
            gdal_prefix=gdal_prefix,
        )
        return cls(
            dataset,
            output_dtype=output_dtype,
            clip_min=clip_min,
            clip_max=clip_max,
            scale_max=scale_max,
            transforms=transforms,
        )

    def __len__(self) -> int:
        return len(self._dataset)

    def record(self, index: int) -> ChipRecord:
        return self._dataset.record(index)

    def sample(self, index: int) -> TrainingSample:
        total_start = perf_counter()
        read_start = perf_counter()
        base_sample = self._dataset.sample(index)
        read_ms = (perf_counter() - read_start) * 1000.0

        transform_start = perf_counter()
        image = preprocess_image(
            base_sample.chip.array,
            output_dtype=self._output_dtype,
            clip_min=self._clip_min,
            clip_max=self._clip_max,
            scale_max=self._scale_max,
        )
        for transform in self._transforms:
            image = transform(image, base_sample.record)
        transform_ms = (perf_counter() - transform_start) * 1000.0
        total_ms = (perf_counter() - total_start) * 1000.0

        return TrainingSample(
            record=base_sample.record,
            image=image,
            timing=SampleTiming(
                read_ms=read_ms,
                transform_ms=transform_ms,
                total_ms=total_ms,
            ),
        )

    def records_frame(self):
        return self._dataset.records_frame()


def preprocess_image(
    array: np.ndarray,
    *,
    output_dtype: np.dtype,
    clip_min: float | None,
    clip_max: float | None,
    scale_max: float | None,
) -> np.ndarray:
    image = np.asarray(array)

    if clip_min is not None or clip_max is not None:
        lower = clip_min if clip_min is not None else image.min()
        upper = clip_max if clip_max is not None else image.max()
        image = np.clip(image, lower, upper)

    if scale_max is not None:
        if scale_max <= 0.0:
            raise ValueError("scale_max must be positive when provided.")
        image = image.astype(np.float32, copy=False) / float(scale_max)

    return image.astype(output_dtype, copy=False)


def collate_training_samples(samples: Sequence[TrainingSample]) -> TrainingBatch:
    if not samples:
        raise ValueError("Cannot collate an empty sample list.")

    images = np.stack([sample.image for sample in samples], axis=0)
    records = tuple(sample.record for sample in samples)
    total_ms = sum(sample.timing.total_ms for sample in samples)
    return TrainingBatch(images=images, records=records, timing_ms_total=total_ms)


def validate_preprocessing(
    *,
    output_dtype: np.dtype,
    clip_min: float | None,
    clip_max: float | None,
    scale_max: float | None,
) -> None:
    if clip_min is not None and clip_max is not None and clip_min > clip_max:
        raise ValueError("clip_min cannot be greater than clip_max.")
    if scale_max is not None and scale_max <= 0.0:
        raise ValueError("scale_max must be positive when provided.")
    if output_dtype.kind not in {"u", "i", "f"}:
        raise ValueError(f"Unsupported output dtype for training images: {output_dtype}")


def benchmark_training_dataset(
    dataset: TrainingChipDataset,
    *,
    repeat: int,
    warmup: int,
) -> TrainingBenchmarkReport:
    if repeat <= 0:
        raise ValueError("--repeat must be positive.")
    if warmup < 0:
        raise ValueError("--warmup must be zero or positive.")
    if len(dataset) == 0:
        raise ValueError("Dataset is empty.")

    for index in cycle_indices(dataset_size=len(dataset), total=warmup):
        dataset.sample(index)

    read_latencies: list[float] = []
    transform_latencies: list[float] = []
    total_latencies: list[float] = []
    total_pixels = 0
    total_bytes = 0
    source_paths: set[Path] = set()

    total_samples = len(dataset) * repeat
    start = perf_counter()
    for index in cycle_indices(dataset_size=len(dataset), total=total_samples):
        sample = dataset.sample(index)
        read_latencies.append(sample.timing.read_ms)
        transform_latencies.append(sample.timing.transform_ms)
        total_latencies.append(sample.timing.total_ms)
        total_pixels += int(sample.image.size)
        total_bytes += int(sample.image.nbytes)
        source_paths.add(sample.record.local_path)
    elapsed_seconds = perf_counter() - start

    return TrainingBenchmarkReport(
        samples=total_samples,
        warmup_samples=warmup,
        unique_source_files=len(source_paths),
        elapsed_seconds=elapsed_seconds,
        total_pixels=total_pixels,
        total_bytes=total_bytes,
        samples_per_second=safe_rate(total_samples, elapsed_seconds),
        megapixels_per_second=safe_rate(total_pixels / 1_000_000.0, elapsed_seconds),
        mebibytes_per_second=safe_rate(total_bytes / (1024.0 * 1024.0), elapsed_seconds),
        read_latency_ms_mean=mean(read_latencies),
        read_latency_ms_p50=percentile(read_latencies, 50.0),
        read_latency_ms_p95=percentile(read_latencies, 95.0),
        transform_latency_ms_mean=mean(transform_latencies),
        transform_latency_ms_p50=percentile(transform_latencies, 50.0),
        transform_latency_ms_p95=percentile(transform_latencies, 95.0),
        total_latency_ms_mean=mean(total_latencies),
        total_latency_ms_p50=percentile(total_latencies, 50.0),
        total_latency_ms_p95=percentile(total_latencies, 95.0),
    )


def parse_benchmark_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark trainer-facing chip throughput.")
    parser.add_argument(
        "--chips-path",
        type=Path,
        default=Path("datasets/manifests/spacenet/chips.parquet"),
    )
    parser.add_argument(
        "--gdal-prefix",
        type=Path,
        help="Override the local GDAL/Kakadu install prefix.",
    )
    parser.add_argument(
        "--split",
        action="append",
        help="Restrict benchmark rows to this split. Repeat to add more splits.",
    )
    parser.add_argument(
        "--modality",
        action="append",
        help="Restrict benchmark rows to this modality. Repeat to add more modalities.",
    )
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--output-dtype", default="float32")
    parser.add_argument("--clip-min", type=float)
    parser.add_argument("--clip-max", type=float)
    parser.add_argument("--scale-max", type=float)
    parser.add_argument(
        "--output-path",
        type=Path,
        help=(
            "Optional JSON output path. "
            "Defaults to artifacts/benchmarks/training-chip-benchmark.json."
        ),
    )
    return parser.parse_args(argv)


def benchmark_main(argv: Sequence[str] | None = None) -> int:
    args = parse_benchmark_args(argv)
    chips_path = args.chips_path.resolve()
    gdal_prefix = args.gdal_prefix.resolve() if args.gdal_prefix is not None else None
    output_path = (
        args.output_path.resolve()
        if args.output_path is not None
        else (Path.cwd() / DEFAULT_BENCHMARK_OUTPUT).resolve()
    )

    dataset = TrainingChipDataset.from_manifest(
        chips_path,
        splits=tuple(args.split) if args.split else None,
        modalities=tuple(args.modality) if args.modality else None,
        limit=args.limit,
        gdal_prefix=gdal_prefix,
        output_dtype=args.output_dtype,
        clip_min=args.clip_min,
        clip_max=args.clip_max,
        scale_max=args.scale_max,
    )
    if len(dataset) == 0:
        raise SystemExit("No local chip rows matched the requested filters.")

    report = benchmark_training_dataset(dataset, repeat=args.repeat, warmup=args.warmup)
    write_training_benchmark(report, output_path)

    print(f"chips: {chips_path}")
    print(f"selected: {len(dataset):,}")
    print(f"samples: {report.samples:,}")
    print(f"samples/s: {report.samples_per_second:.2f}")
    print(f"MPix/s: {report.megapixels_per_second:.2f}")
    print(f"MiB/s: {report.mebibytes_per_second:.2f}")
    print(
        "read_ms: "
        f"mean={report.read_latency_ms_mean:.2f} "
        f"p50={report.read_latency_ms_p50:.2f} "
        f"p95={report.read_latency_ms_p95:.2f}"
    )
    print(
        "transform_ms: "
        f"mean={report.transform_latency_ms_mean:.2f} "
        f"p50={report.transform_latency_ms_p50:.2f} "
        f"p95={report.transform_latency_ms_p95:.2f}"
    )
    print(
        "total_ms: "
        f"mean={report.total_latency_ms_mean:.2f} "
        f"p50={report.total_latency_ms_p50:.2f} "
        f"p95={report.total_latency_ms_p95:.2f}"
    )
    print(f"report: {output_path}")
    return 0


def write_training_benchmark(report: TrainingBenchmarkReport, path: Path) -> Path:
    output_path = path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(report), indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def cycle_indices(*, dataset_size: int, total: int) -> list[int]:
    if dataset_size <= 0:
        raise ValueError("dataset_size must be positive.")
    return [index % dataset_size for index in range(total)]


def percentile(values: Sequence[float], percentile_rank: float) -> float:
    if not values:
        raise ValueError("Cannot compute a percentile from an empty sequence.")
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
        raise ValueError("Cannot compute a mean from an empty sequence.")
    return sum(float(value) for value in values) / len(values)


def safe_rate(value: float, seconds: float) -> float:
    if seconds <= 0.0:
        return 0.0
    return value / seconds
