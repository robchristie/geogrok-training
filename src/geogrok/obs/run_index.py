from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DEFAULT_RUN_ROOT = Path("artifacts/runs")


@dataclass(frozen=True)
class RunSummary:
    run_id: str
    run_root: str
    summary_path: str
    run_kind: str
    teacher_model: str | None
    student_model: str | None
    metrics: dict[str, float]
    raw_summary: dict[str, Any]


def infer_run_kind(summary: dict[str, Any]) -> str:
    if "teacher_eval_retrieval" in summary and "student_eval_retrieval" in summary:
        return "pan_adapt_benchmark"
    if "models" in summary:
        return "pretrained_benchmark"
    if "train" in summary:
        return "training_loop"
    return "unknown"


def summarize_metrics(summary: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if "models" in summary and isinstance(summary["models"], list):
        ranked_models = [
            model
            for model in summary["models"]
            if isinstance(model, dict)
            and isinstance(model.get("exact_recall_at_10"), int | float)
        ]
        if ranked_models:
            best = max(
                ranked_models,
                key=lambda model: (
                    float(model.get("exact_recall_at_10", 0.0)),
                    float(model.get("any_mean_reciprocal_rank", 0.0)),
                ),
            )
            for key in ("exact_recall_at_10", "any_recall_at_10", "any_mean_reciprocal_rank"):
                value = best.get(key)
                if isinstance(value, int | float):
                    metrics[f"best.{key}"] = float(value)
    if "student_eval_retrieval" in summary:
        student = summary["student_eval_retrieval"]
        if isinstance(student, dict):
            for key in ("exact_recall_at_10", "any_recall_at_10", "any_mean_reciprocal_rank"):
                value = student.get(key)
                if isinstance(value, int | float):
                    metrics[f"student.{key}"] = float(value)
    if "teacher_eval_retrieval" in summary:
        teacher = summary["teacher_eval_retrieval"]
        if isinstance(teacher, dict):
            for key in ("exact_recall_at_10", "any_recall_at_10", "any_mean_reciprocal_rank"):
                value = teacher.get(key)
                if isinstance(value, int | float):
                    metrics[f"teacher.{key}"] = float(value)
    if "train" in summary and isinstance(summary["train"], dict):
        train = summary["train"]
        for key in ("samples_per_second_mean", "megapixels_per_second_mean"):
            value = train.get(key)
            if isinstance(value, int | float):
                metrics[f"train.{key}"] = float(value)
    return metrics


def load_run_summary(summary_path: Path) -> RunSummary:
    run_root = summary_path.parent
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    run_kind = infer_run_kind(summary)
    teacher_model = summary.get("teacher_model")
    student_model = None
    if run_kind == "pan_adapt_benchmark":
        student_training_path = run_root / "student_training.json"
        if student_training_path.exists():
            training = json.loads(student_training_path.read_text(encoding="utf-8"))
            student_arch = training.get("student_arch")
            student_model = str(student_arch) if student_arch else None
    return RunSummary(
        run_id=run_root.name,
        run_root=str(run_root.resolve()),
        summary_path=str(summary_path.resolve()),
        run_kind=run_kind,
        teacher_model=str(teacher_model) if teacher_model is not None else None,
        student_model=student_model,
        metrics=summarize_metrics(summary),
        raw_summary=summary,
    )


def collect_run_summaries(run_root: Path = DEFAULT_RUN_ROOT) -> list[RunSummary]:
    root = run_root.expanduser().resolve()
    if not root.exists():
        return []
    summaries: list[RunSummary] = []
    for summary_path in sorted(root.glob("*/summary.json")):
        summaries.append(load_run_summary(summary_path))
    return summaries


def find_run_summary(run_id: str, run_root: Path = DEFAULT_RUN_ROOT) -> RunSummary | None:
    for summary in collect_run_summaries(run_root):
        if summary.run_id == run_id:
            return summary
    return None


def collect_run_summary_dicts(run_root: Path = DEFAULT_RUN_ROOT) -> list[dict[str, Any]]:
    return [asdict(summary) for summary in collect_run_summaries(run_root)]
