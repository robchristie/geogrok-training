from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict

from geogrok.data.training import TrainingChipDataset
from geogrok.training.loop import EpochMetrics, run_epoch, write_metrics_jsonl

DEFAULT_RUN_ROOT = Path("artifacts/runs/training-dryrun")


class StageSummary(TypedDict):
    epochs: int
    samples_per_second_mean: float
    megapixels_per_second_mean: float
    read_latency_ms_p95_mean: float
    transform_latency_ms_p95_mean: float
    sample_total_ms_p95_mean: float


class RunSummary(TypedDict):
    train_dataset_samples: int
    val_dataset_samples: int
    train: StageSummary
    val: StageSummary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a deterministic training-loop dry run over on-demand chips."
    )
    parser.add_argument(
        "--chips-path",
        type=Path,
        default=Path("datasets/manifests/spacenet/chips.parquet"),
    )
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument(
        "--gdal-prefix",
        type=Path,
        help="Override the local GDAL/Kakadu install prefix.",
    )
    parser.add_argument(
        "--train-split",
        action="append",
        default=["train"],
        help="Training split to include. Repeat to add more splits.",
    )
    parser.add_argument(
        "--val-split",
        action="append",
        default=["val"],
        help="Validation split to include. Repeat to add more splits.",
    )
    parser.add_argument(
        "--modality",
        action="append",
        default=["PAN"],
        help="Restrict to this modality. Repeat to add more modalities.",
    )
    parser.add_argument("--train-limit", type=int, default=32)
    parser.add_argument("--val-limit", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dtype", default="float32")
    parser.add_argument("--clip-min", type=float, default=0.0)
    parser.add_argument("--clip-max", type=float, default=2047.0)
    parser.add_argument("--scale-max", type=float, default=2047.0)
    return parser.parse_args(argv)


def build_dataset(
    chips_path: Path,
    *,
    splits: Sequence[str],
    modalities: Sequence[str],
    limit: int,
    gdal_prefix: Path | None,
    output_dtype: str,
    clip_min: float | None,
    clip_max: float | None,
    scale_max: float | None,
) -> TrainingChipDataset:
    return TrainingChipDataset.from_manifest(
        chips_path,
        splits=tuple(splits),
        modalities=tuple(modalities),
        limit=limit,
        gdal_prefix=gdal_prefix,
        output_dtype=output_dtype,
        clip_min=clip_min,
        clip_max=clip_max,
        scale_max=scale_max,
    )


def run_training_dryrun(args: argparse.Namespace) -> tuple[list[EpochMetrics], RunSummary]:
    chips_path = args.chips_path.resolve()
    gdal_prefix = args.gdal_prefix.resolve() if args.gdal_prefix is not None else None

    train_dataset = build_dataset(
        chips_path,
        splits=args.train_split,
        modalities=args.modality,
        limit=args.train_limit,
        gdal_prefix=gdal_prefix,
        output_dtype=args.output_dtype,
        clip_min=args.clip_min,
        clip_max=args.clip_max,
        scale_max=args.scale_max,
    )
    val_dataset = build_dataset(
        chips_path,
        splits=args.val_split,
        modalities=args.modality,
        limit=args.val_limit,
        gdal_prefix=gdal_prefix,
        output_dtype=args.output_dtype,
        clip_min=args.clip_min,
        clip_max=args.clip_max,
        scale_max=args.scale_max,
    )

    metrics: list[EpochMetrics] = []
    for epoch in range(args.epochs):
        if len(train_dataset) > 0:
            metrics.append(
                run_epoch(
                    train_dataset,
                    stage="train",
                    epoch=epoch,
                    batch_size=args.batch_size,
                    shuffle=True,
                    seed=args.seed,
                )
            )
        if len(val_dataset) > 0:
            metrics.append(
                run_epoch(
                    val_dataset,
                    stage="val",
                    epoch=epoch,
                    batch_size=args.batch_size,
                    shuffle=False,
                    seed=args.seed,
                )
            )

    summary = build_summary(
        metrics,
        train_samples=len(train_dataset),
        val_samples=len(val_dataset),
    )
    return metrics, summary


def build_summary(
    metrics: Sequence[EpochMetrics],
    *,
    train_samples: int,
    val_samples: int,
) -> RunSummary:
    by_stage: dict[str, list[EpochMetrics]] = {}
    for metric in metrics:
        by_stage.setdefault(metric.stage, []).append(metric)

    def stage_summary(stage: str) -> StageSummary:
        stage_metrics = by_stage.get(stage, [])
        return {
            "epochs": len(stage_metrics),
            "samples_per_second_mean": _mean(metric.samples_per_second for metric in stage_metrics),
            "megapixels_per_second_mean": _mean(
                metric.megapixels_per_second for metric in stage_metrics
            ),
            "read_latency_ms_p95_mean": _mean(
                metric.read_latency_ms_p95 for metric in stage_metrics
            ),
            "transform_latency_ms_p95_mean": _mean(
                metric.transform_latency_ms_p95 for metric in stage_metrics
            ),
            "sample_total_ms_p95_mean": _mean(
                metric.sample_total_ms_p95 for metric in stage_metrics
            ),
        }

    return {
        "train_dataset_samples": train_samples,
        "val_dataset_samples": val_samples,
        "train": stage_summary("train"),
        "val": stage_summary("val"),
    }


def write_summary(summary: RunSummary, path: Path) -> Path:
    output_path = path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_root = args.run_root.resolve()
    metrics_path = run_root / "metrics.jsonl"
    summary_path = run_root / "summary.json"

    metrics, summary = run_training_dryrun(args)
    write_metrics_jsonl(metrics, metrics_path)
    write_summary(summary, summary_path)

    for metric in metrics:
        print(
            f"{metric.stage} epoch={metric.epoch} "
            f"samples={metric.samples} "
            f"samples/s={metric.samples_per_second:.2f} "
            f"MPix/s={metric.megapixels_per_second:.2f} "
            f"read_p95_ms={metric.read_latency_ms_p95:.2f} "
            f"transform_p95_ms={metric.transform_latency_ms_p95:.2f} "
            f"total_p95_ms={metric.sample_total_ms_p95:.2f}"
        )
    print(f"metrics: {metrics_path}")
    print(f"summary: {summary_path}")
    return 0


def _mean(values) -> float:
    collected = [float(value) for value in values]
    if not collected:
        return 0.0
    return sum(collected) / len(collected)


if __name__ == "__main__":
    raise SystemExit(main())
