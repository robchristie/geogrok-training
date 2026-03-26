from __future__ import annotations

import json
from pathlib import Path

from geogrok.obs.run_index import collect_run_summaries


def test_collect_run_summaries_reads_pan_adapt_metrics(tmp_path: Path):
    run_root = tmp_path / "runs"
    run_dir = run_root / "pan-adapt-smoke"
    run_dir.mkdir(parents=True)
    summary = {
        "teacher_model": "dinov3_vitb16",
        "teacher_eval_retrieval": {
            "exact_recall_at_10": 0.59,
            "any_recall_at_10": 0.52,
            "any_mean_reciprocal_rank": 0.37,
        },
        "student_eval_retrieval": {
            "exact_recall_at_10": 0.08,
            "any_recall_at_10": 0.12,
            "any_mean_reciprocal_rank": 0.05,
        },
    }
    student_training = {"student_arch": "residual_cnn"}
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (run_dir / "student_training.json").write_text(
        json.dumps(student_training),
        encoding="utf-8",
    )

    runs = collect_run_summaries(run_root)

    assert len(runs) == 1
    assert runs[0].run_kind == "pan_adapt_benchmark"
    assert runs[0].teacher_model == "dinov3_vitb16"
    assert runs[0].student_model == "residual_cnn"
    assert runs[0].metrics["student.exact_recall_at_10"] == 0.08


def test_collect_run_summaries_reads_pretrained_best_metrics(tmp_path: Path):
    run_root = tmp_path / "runs"
    run_dir = run_root / "pretrained-benchmark-smoke"
    run_dir.mkdir(parents=True)
    summary = {
        "models": [
            {
                "model_name": "resnet50",
                "exact_recall_at_10": 0.60,
                "any_recall_at_10": 0.53,
                "any_mean_reciprocal_rank": 0.36,
            },
            {
                "model_name": "resnet152",
                "exact_recall_at_10": 0.62,
                "any_recall_at_10": 0.54,
                "any_mean_reciprocal_rank": 0.35,
            },
        ]
    }
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    runs = collect_run_summaries(run_root)

    assert len(runs) == 1
    assert runs[0].run_kind == "pretrained_benchmark"
    assert runs[0].metrics["best.exact_recall_at_10"] == 0.62
