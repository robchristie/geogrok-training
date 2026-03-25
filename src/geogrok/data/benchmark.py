from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Protocol

from geogrok.data.runtime import ChipSample, OnDemandChipDataset

DEFAULT_CHIPS_MANIFEST = Path("datasets/manifests/spacenet/chips.parquet")
DEFAULT_OUTPUT_PATH = Path("artifacts/benchmarks/chip-read-benchmark.json")


@dataclass(frozen=True)
class ChipReadBenchmarkReport:
    samples: int
    warmup_samples: int
    unique_source_files: int
    elapsed_seconds: float
    total_pixels: int
    total_bytes: int
    samples_per_second: float
    megapixels_per_second: float
    mebibytes_per_second: float
    latency_ms_mean: float
    latency_ms_min: float
    latency_ms_p50: float
    latency_ms_p95: float
    latency_ms_max: float


class ChipSampleDataset(Protocol):
    def __len__(self) -> int: ...

    def sample(self, index: int) -> ChipSample: ...


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark on-demand chip read throughput.")
    parser.add_argument("--chips-path", type=Path, default=DEFAULT_CHIPS_MANIFEST)
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
    parser.add_argument(
        "--limit",
        type=int,
        default=32,
        help="Maximum distinct chip rows to benchmark.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of benchmark passes over the selected rows.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Number of initial reads to run before timed measurement.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        help=(
            "Optional JSON output path. "
            "Defaults to artifacts/benchmarks/chip-read-benchmark.json."
        ),
    )
    return parser.parse_args(argv)


def benchmark_dataset_reads(
    dataset: ChipSampleDataset,
    *,
    repeat: int,
    warmup: int,
) -> ChipReadBenchmarkReport:
    if repeat <= 0:
        raise ValueError("--repeat must be positive.")
    if warmup < 0:
        raise ValueError("--warmup must be zero or positive.")
    if len(dataset) == 0:
        raise ValueError("Dataset is empty.")

    total_samples = len(dataset) * repeat
    warmup_indices = cycle_indices(dataset_size=len(dataset), total=warmup)
    measured_indices = cycle_indices(dataset_size=len(dataset), total=total_samples)

    for index in warmup_indices:
        dataset.sample(index)

    latencies_ms: list[float] = []
    total_pixels = 0
    total_bytes = 0
    source_paths: set[Path] = set()

    start = perf_counter()
    for index in measured_indices:
        sample_start = perf_counter()
        sample = dataset.sample(index)
        sample_elapsed_ms = (perf_counter() - sample_start) * 1000.0
        latencies_ms.append(sample_elapsed_ms)

        array = sample.chip.array
        total_pixels += int(array.size)
        total_bytes += int(array.nbytes)
        source_paths.add(sample.record.local_path)
    elapsed_seconds = perf_counter() - start

    return ChipReadBenchmarkReport(
        samples=total_samples,
        warmup_samples=warmup,
        unique_source_files=len(source_paths),
        elapsed_seconds=elapsed_seconds,
        total_pixels=total_pixels,
        total_bytes=total_bytes,
        samples_per_second=safe_rate(total_samples, elapsed_seconds),
        megapixels_per_second=safe_rate(total_pixels / 1_000_000.0, elapsed_seconds),
        mebibytes_per_second=safe_rate(total_bytes / (1024.0 * 1024.0), elapsed_seconds),
        latency_ms_mean=(sum(latencies_ms) / len(latencies_ms)),
        latency_ms_min=min(latencies_ms),
        latency_ms_p50=percentile(latencies_ms, 50.0),
        latency_ms_p95=percentile(latencies_ms, 95.0),
        latency_ms_max=max(latencies_ms),
    )


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


def safe_rate(value: float, seconds: float) -> float:
    if seconds <= 0.0:
        return 0.0
    return value / seconds


def write_report(report: ChipReadBenchmarkReport, path: Path) -> Path:
    output_path = path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(report), indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    chips_path = args.chips_path.resolve()
    output_path = (
        args.output_path.resolve()
        if args.output_path is not None
        else (Path.cwd() / DEFAULT_OUTPUT_PATH).resolve()
    )
    gdal_prefix = args.gdal_prefix.resolve() if args.gdal_prefix is not None else None

    dataset = OnDemandChipDataset.from_manifest(
        chips_path,
        splits=tuple(args.split) if args.split else None,
        modalities=tuple(args.modality) if args.modality else None,
        limit=args.limit,
        gdal_prefix=gdal_prefix,
    )
    if len(dataset) == 0:
        raise SystemExit("No local chip rows matched the requested filters.")

    report = benchmark_dataset_reads(dataset, repeat=args.repeat, warmup=args.warmup)
    write_report(report, output_path)

    print(f"chips: {chips_path}")
    print(f"selected: {len(dataset):,}")
    print(f"samples: {report.samples:,}")
    print(f"samples/s: {report.samples_per_second:.2f}")
    print(f"MPix/s: {report.megapixels_per_second:.2f}")
    print(f"MiB/s: {report.mebibytes_per_second:.2f}")
    print(
        "latency_ms: "
        f"mean={report.latency_ms_mean:.2f} "
        f"p50={report.latency_ms_p50:.2f} "
        f"p95={report.latency_ms_p95:.2f} "
        f"max={report.latency_ms_max:.2f}"
    )
    print(f"report: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
