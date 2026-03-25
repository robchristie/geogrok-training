from __future__ import annotations

from geogrok.training.baseline import build_summary
from geogrok.training.loop import EpochMetrics


def test_build_summary_groups_metrics_by_stage():
    metrics = [
        EpochMetrics(
            stage="train",
            epoch=0,
            batches=2,
            samples=8,
            elapsed_seconds=1.0,
            total_pixels=100,
            total_bytes=400,
            samples_per_second=8.0,
            megapixels_per_second=1.0,
            mebibytes_per_second=2.0,
            read_latency_ms_mean=1.0,
            read_latency_ms_p50=1.0,
            read_latency_ms_p95=1.5,
            transform_latency_ms_mean=0.5,
            transform_latency_ms_p50=0.5,
            transform_latency_ms_p95=0.75,
            sample_total_ms_mean=1.5,
            sample_total_ms_p50=1.5,
            sample_total_ms_p95=2.0,
            batch_collate_ms_mean=0.1,
            batch_collate_ms_p50=0.1,
            batch_collate_ms_p95=0.2,
        ),
        EpochMetrics(
            stage="val",
            epoch=0,
            batches=1,
            samples=4,
            elapsed_seconds=1.0,
            total_pixels=50,
            total_bytes=200,
            samples_per_second=4.0,
            megapixels_per_second=0.5,
            mebibytes_per_second=1.0,
            read_latency_ms_mean=2.0,
            read_latency_ms_p50=2.0,
            read_latency_ms_p95=2.5,
            transform_latency_ms_mean=0.2,
            transform_latency_ms_p50=0.2,
            transform_latency_ms_p95=0.3,
            sample_total_ms_mean=2.2,
            sample_total_ms_p50=2.2,
            sample_total_ms_p95=2.8,
            batch_collate_ms_mean=0.05,
            batch_collate_ms_p50=0.05,
            batch_collate_ms_p95=0.1,
        ),
    ]

    summary = build_summary(metrics, train_samples=8, val_samples=4)

    assert summary["train_dataset_samples"] == 8
    assert summary["val_dataset_samples"] == 4
    assert summary["train"]["epochs"] == 1
    assert summary["val"]["epochs"] == 1
