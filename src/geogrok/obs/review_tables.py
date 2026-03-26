from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from geogrok.obs.data import default_data_paths
from geogrok.obs.run_index import RunSummary
from geogrok.retrieval.baseline import row_normalize


@dataclass(frozen=True)
class RunSelection:
    selection_id: str
    label: str
    embeddings_path: Path
    index_path: Path
    retrieval_path: Path
    query_splits: tuple[str, ...]
    gallery_splits: tuple[str, ...]


def available_run_selections(run: RunSummary) -> list[RunSelection]:
    run_root = Path(run.run_root)
    if run.run_kind == "pretrained_benchmark":
        models = run.raw_summary.get("models", [])
        if not isinstance(models, list):
            return []
        query_splits = tuple(
            str(value) for value in run.raw_summary.get("query_splits", ["val", "test"])
        )
        gallery_splits = tuple(
            str(value) for value in run.raw_summary.get("gallery_splits", ["val", "test"])
        )
        selections: list[RunSelection] = []
        for model in models:
            if not isinstance(model, dict):
                continue
            model_name = str(model.get("model_name", "")).strip()
            if not model_name:
                continue
            model_root = run_root / model_name
            selections.append(
                RunSelection(
                    selection_id=model_name,
                    label=model_name,
                    embeddings_path=model_root / "embeddings.npy",
                    index_path=model_root / "index.parquet",
                    retrieval_path=model_root / "retrieval.json",
                    query_splits=query_splits,
                    gallery_splits=gallery_splits,
                )
            )
        return selections
    if run.run_kind == "pan_adapt_benchmark":
        return [
            RunSelection(
                selection_id="student",
                label="student",
                embeddings_path=run_root / "student_eval_embeddings.npy",
                index_path=run_root / "student_eval_index.parquet",
                retrieval_path=run_root / "student_eval_retrieval.json",
                query_splits=selection_splits(
                    run_root / "student_eval_retrieval.json",
                    "query_splits",
                ),
                gallery_splits=selection_splits(
                    run_root / "student_eval_retrieval.json",
                    "gallery_splits",
                ),
            ),
            RunSelection(
                selection_id="teacher",
                label=str(run.teacher_model or "teacher"),
                embeddings_path=run_root / "teacher_eval_embeddings.npy",
                index_path=run_root / "teacher_eval_index.parquet",
                retrieval_path=run_root / "teacher_eval_retrieval.json",
                query_splits=selection_splits(
                    run_root / "teacher_eval_retrieval.json",
                    "query_splits",
                ),
                gallery_splits=selection_splits(
                    run_root / "teacher_eval_retrieval.json",
                    "gallery_splits",
                ),
            ),
        ]
    return []


def selection_splits(path: Path, key: str) -> tuple[str, ...]:
    if not path.exists():
        return ("val", "test")
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload.get(key)
    if not isinstance(values, list) or not values:
        return ("val", "test")
    return tuple(str(value) for value in values)


def default_selection_id(run: RunSummary) -> str | None:
    selections = available_run_selections(run)
    if not selections:
        return None
    if run.run_kind == "pretrained_benchmark":
        models = run.raw_summary.get("models", [])
        if isinstance(models, list) and models:
            ranked = [
                model
                for model in models
                if isinstance(model, dict) and isinstance(model.get("model_name"), str)
            ]
            if ranked:
                best = max(
                    ranked,
                    key=lambda model: (
                        float(model.get("exact_recall_at_10", 0.0)),
                        float(model.get("any_mean_reciprocal_rank", 0.0)),
                    ),
                )
                return str(best["model_name"])
    return selections[0].selection_id


def describe_run(run: RunSummary) -> dict[str, Any]:
    return {
        "run_id": run.run_id,
        "run_root": run.run_root,
        "run_kind": run.run_kind,
        "teacher_model": run.teacher_model,
        "student_model": run.student_model,
        "metrics": run.metrics,
        "available_selections": [
            {
                "selection_id": selection.selection_id,
                "label": selection.label,
                "query_splits": list(selection.query_splits),
                "gallery_splits": list(selection.gallery_splits),
            }
            for selection in available_run_selections(run)
            if selection.embeddings_path.exists() and selection.index_path.exists()
        ],
        "default_selection_id": default_selection_id(run),
    }


def failure_response(
    run: RunSummary,
    *,
    selection_id: str | None = None,
    top_k: int = 10,
    limit: int = 24,
) -> dict[str, Any]:
    selection, pairs_path, ranked_pairs = ranked_pairs_for_run_selection(run, selection_id)
    queues = build_failure_queues(ranked_pairs, top_k=top_k, limit=limit)
    return {
        "run_id": run.run_id,
        "run_kind": run.run_kind,
        "selection": {
            "selection_id": selection.selection_id,
            "label": selection.label,
            "top_k": top_k,
            "query_splits": list(selection.query_splits),
            "gallery_splits": list(selection.gallery_splits),
            "pairs_path": str(pairs_path),
        },
        "queue_counts": {
            "false_negatives": int(len(queues["false_negatives_all"])),
            "false_positives": int(len(queues["false_positives_all"])),
        },
        "false_negatives": queues["false_negatives"],
        "false_positives": queues["false_positives"],
    }


def disagreement_response(
    run: RunSummary,
    *,
    limit: int = 24,
) -> dict[str, Any]:
    if run.run_kind != "pan_adapt_benchmark":
        raise ValueError("Disagreement review is only available for pan-adapt benchmark runs.")
    teacher_selection, pairs_path, teacher_ranked = ranked_pairs_for_run_selection(run, "teacher")
    student_selection, _, student_ranked = ranked_pairs_for_run_selection(run, "student")
    queues = build_disagreement_queues(
        teacher_ranked,
        student_ranked,
        limit=limit,
    )
    return {
        "run_id": run.run_id,
        "run_kind": run.run_kind,
        "pairs_path": str(pairs_path),
        "teacher_selection": {
            "selection_id": teacher_selection.selection_id,
            "label": teacher_selection.label,
            "query_splits": list(teacher_selection.query_splits),
            "gallery_splits": list(teacher_selection.gallery_splits),
        },
        "student_selection": {
            "selection_id": student_selection.selection_id,
            "label": student_selection.label,
            "query_splits": list(student_selection.query_splits),
            "gallery_splits": list(student_selection.gallery_splits),
        },
        "queue_counts": {
            "teacher_ahead_positives": int(len(queues["teacher_ahead_positives_all"])),
            "student_ahead_positives": int(len(queues["student_ahead_positives_all"])),
            "student_confused_negatives": int(len(queues["student_confused_negatives_all"])),
            "teacher_confused_negatives": int(len(queues["teacher_confused_negatives_all"])),
        },
        "teacher_ahead_positives": queues["teacher_ahead_positives"],
        "student_ahead_positives": queues["student_ahead_positives"],
        "student_confused_negatives": queues["student_confused_negatives"],
        "teacher_confused_negatives": queues["teacher_confused_negatives"],
    }


def resolve_selection(run: RunSummary, selection_id: str | None) -> RunSelection:
    selections = available_run_selections(run)
    if not selections:
        raise FileNotFoundError(f"No reviewable selections for run {run.run_id}")
    chosen_id = selection_id or default_selection_id(run)
    for selection in selections:
        if selection.selection_id == chosen_id:
            if not selection.embeddings_path.exists():
                raise FileNotFoundError(selection.embeddings_path)
            if not selection.index_path.exists():
                raise FileNotFoundError(selection.index_path)
            return selection
    raise KeyError(chosen_id)


def pairs_path_for_run(run: RunSummary) -> Path:
    raw_value = run.raw_summary.get("pairs_path")
    if isinstance(raw_value, str):
        candidate = Path(raw_value)
        if candidate.exists():
            return candidate.resolve()
    return default_data_paths().pairs_path


def ranked_pairs_for_run_selection(
    run: RunSummary,
    selection_id: str | None,
) -> tuple[RunSelection, Path, pd.DataFrame]:
    selection = resolve_selection(run, selection_id)
    pairs_path = pairs_path_for_run(run)
    embeddings = np.load(selection.embeddings_path)
    metadata = pd.read_parquet(selection.index_path)
    pairs = pd.read_parquet(pairs_path)
    ranked_pairs = ranked_pair_rows(
        embeddings,
        metadata,
        pairs,
        query_splits=selection.query_splits,
        gallery_splits=selection.gallery_splits,
    )
    return selection, pairs_path, ranked_pairs


def ranked_pair_rows(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    pairs: pd.DataFrame,
    *,
    query_splits: tuple[str, ...],
    gallery_splits: tuple[str, ...],
) -> pd.DataFrame:
    if len(embeddings) != len(metadata):
        raise ValueError("Embedding matrix and metadata row count must match.")
    if len(embeddings) == 0 or pairs.empty:
        return pd.DataFrame()

    frame = metadata.reset_index(drop=True).copy()
    frame["chip_id"] = frame["chip_id"].astype(str)
    frame["split_normalized"] = frame["split"].astype(str).str.lower()
    chip_index = {chip_id: int(index) for index, chip_id in enumerate(frame["chip_id"])}

    query_tokens = {split.lower() for split in query_splits}
    gallery_tokens = {split.lower() for split in gallery_splits}
    query_indices = [
        int(index)
        for index in frame.index
        if str(frame.iloc[index]["split_normalized"]) in query_tokens
    ]
    gallery_indices = [
        int(index)
        for index in frame.index
        if str(frame.iloc[index]["split_normalized"]) in gallery_tokens
    ]

    pair_frame = pairs.copy()
    pair_frame["query_chip_id"] = pair_frame["query_chip_id"].astype(str)
    pair_frame["candidate_chip_id"] = pair_frame["candidate_chip_id"].astype(str)
    pair_frame["query_index"] = pair_frame["query_chip_id"].map(chip_index)
    pair_frame["candidate_index"] = pair_frame["candidate_chip_id"].map(chip_index)
    pair_frame = pair_frame.dropna(subset=["query_index", "candidate_index"]).copy()
    if pair_frame.empty:
        return pd.DataFrame()
    pair_frame["query_index"] = pair_frame["query_index"].astype(int)
    pair_frame["candidate_index"] = pair_frame["candidate_index"].astype(int)
    pair_frame = pair_frame[
        pair_frame["query_index"].isin(query_indices)
        & pair_frame["candidate_index"].isin(gallery_indices)
    ].reset_index(drop=True)
    if pair_frame.empty:
        return pd.DataFrame()

    matrix = row_normalize(np.asarray(embeddings, dtype=np.float32))
    similarity = matrix @ matrix.T
    records: list[dict[str, Any]] = []
    for query_index, query_pairs in pair_frame.groupby("query_index", sort=True):
        query_index = int(query_index)
        ranking = sorted(
            [candidate for candidate in gallery_indices if candidate != query_index],
            key=lambda candidate: float(similarity[query_index, candidate]),
            reverse=True,
        )
        rank_by_candidate = {candidate: rank for rank, candidate in enumerate(ranking, start=1)}
        for row in query_pairs.to_dict(orient="records"):
            candidate_index = int(row["candidate_index"])
            rank = rank_by_candidate.get(candidate_index)
            if rank is None:
                continue
            records.append(
                {
                    "pair_key": f"{row['query_chip_id']}__{row['candidate_chip_id']}",
                    "query_chip_id": str(row["query_chip_id"]),
                    "candidate_chip_id": str(row["candidate_chip_id"]),
                    "pair_label": str(row["pair_label"]),
                    "pair_group": str(row.get("pair_group", "")),
                    "query_split": str(row.get("query_split", "")),
                    "candidate_split": str(row.get("candidate_split", "")),
                    "city": str(row.get("city", "")),
                    "modality": str(row.get("modality", "")),
                    "overlap_fraction": float(row.get("overlap_fraction", 0.0)),
                    "overlap_iou": float(row.get("overlap_iou", 0.0)),
                    "time_delta_seconds": float(row.get("time_delta_seconds", 0.0)),
                    "center_distance_m": float(row.get("center_distance_m", 0.0)),
                    "rank": int(rank),
                    "similarity": float(similarity[query_index, candidate_index]),
                }
            )
    return pd.DataFrame(records)


def build_failure_queues(
    ranked_pairs: pd.DataFrame,
    *,
    top_k: int,
    limit: int,
) -> dict[str, Any]:
    if ranked_pairs.empty:
        return {
            "false_negatives_all": [],
            "false_positives_all": [],
            "false_negatives": [],
            "false_positives": [],
        }

    positives = ranked_pairs[
        ranked_pairs["pair_label"].isin(["positive_exact", "positive_weak"])
    ].copy()
    positives["label_priority"] = positives["pair_label"].map(
        {"positive_exact": 0, "positive_weak": 1}
    ).fillna(9)
    false_negatives = positives[positives["rank"] > top_k].sort_values(
        ["label_priority", "rank", "similarity"],
        ascending=[True, True, False],
    )

    negatives = ranked_pairs[ranked_pairs["pair_label"] == "negative_hard"].copy()
    false_positives = negatives[negatives["rank"] <= top_k].sort_values(
        ["rank", "similarity"],
        ascending=[True, False],
    )

    return {
        "false_negatives_all": false_negatives.to_dict(orient="records"),
        "false_positives_all": false_positives.to_dict(orient="records"),
        "false_negatives": false_negatives.head(limit).to_dict(orient="records"),
        "false_positives": false_positives.head(limit).to_dict(orient="records"),
    }


def build_disagreement_queues(
    teacher_ranked: pd.DataFrame,
    student_ranked: pd.DataFrame,
    *,
    limit: int,
) -> dict[str, Any]:
    if teacher_ranked.empty or student_ranked.empty:
        return {
            "teacher_ahead_positives_all": [],
            "student_ahead_positives_all": [],
            "student_confused_negatives_all": [],
            "teacher_confused_negatives_all": [],
            "teacher_ahead_positives": [],
            "student_ahead_positives": [],
            "student_confused_negatives": [],
            "teacher_confused_negatives": [],
        }

    join_columns = [
        "pair_key",
        "query_chip_id",
        "candidate_chip_id",
        "pair_label",
        "pair_group",
        "query_split",
        "candidate_split",
        "city",
        "modality",
        "overlap_fraction",
        "overlap_iou",
        "time_delta_seconds",
        "center_distance_m",
    ]
    teacher = teacher_ranked[join_columns + ["rank", "similarity"]].rename(
        columns={
            "rank": "teacher_rank",
            "similarity": "teacher_similarity",
        }
    )
    student = student_ranked[join_columns + ["rank", "similarity"]].rename(
        columns={
            "rank": "student_rank",
            "similarity": "student_similarity",
        }
    )
    merged = teacher.merge(student, how="inner", on=join_columns)
    if merged.empty:
        return {
            "teacher_ahead_positives_all": [],
            "student_ahead_positives_all": [],
            "student_confused_negatives_all": [],
            "teacher_confused_negatives_all": [],
            "teacher_ahead_positives": [],
            "student_ahead_positives": [],
            "student_confused_negatives": [],
            "teacher_confused_negatives": [],
        }

    merged["teacher_rank_advantage"] = merged["student_rank"] - merged["teacher_rank"]
    merged["student_rank_advantage"] = merged["teacher_rank"] - merged["student_rank"]
    merged["teacher_similarity_advantage"] = (
        merged["teacher_similarity"] - merged["student_similarity"]
    )
    merged["student_similarity_advantage"] = (
        merged["student_similarity"] - merged["teacher_similarity"]
    )

    positives = merged[merged["pair_label"].isin(["positive_exact", "positive_weak"])].copy()
    teacher_ahead_positives = positives[
        (positives["teacher_rank_advantage"] > 0)
        | (positives["teacher_similarity_advantage"] > 0.0)
    ].sort_values(
        ["teacher_rank_advantage", "teacher_similarity_advantage", "teacher_rank"],
        ascending=[False, False, True],
    )
    student_ahead_positives = positives[
        (positives["student_rank_advantage"] > 0)
        | (positives["student_similarity_advantage"] > 0.0)
    ].sort_values(
        ["student_rank_advantage", "student_similarity_advantage", "student_rank"],
        ascending=[False, False, True],
    )

    negatives = merged[merged["pair_label"] == "negative_hard"].copy()
    student_confused_negatives = negatives[
        (negatives["student_rank_advantage"] > 0)
        | (negatives["student_similarity_advantage"] > 0.0)
    ].sort_values(
        ["student_rank_advantage", "student_similarity_advantage", "student_rank"],
        ascending=[False, False, True],
    )
    teacher_confused_negatives = negatives[
        (negatives["teacher_rank_advantage"] > 0)
        | (negatives["teacher_similarity_advantage"] > 0.0)
    ].sort_values(
        ["teacher_rank_advantage", "teacher_similarity_advantage", "teacher_rank"],
        ascending=[False, False, True],
    )

    return {
        "teacher_ahead_positives_all": teacher_ahead_positives.to_dict(orient="records"),
        "student_ahead_positives_all": student_ahead_positives.to_dict(orient="records"),
        "student_confused_negatives_all": student_confused_negatives.to_dict(orient="records"),
        "teacher_confused_negatives_all": teacher_confused_negatives.to_dict(orient="records"),
        "teacher_ahead_positives": teacher_ahead_positives.head(limit).to_dict(orient="records"),
        "student_ahead_positives": student_ahead_positives.head(limit).to_dict(orient="records"),
        "student_confused_negatives": student_confused_negatives.head(limit).to_dict(
            orient="records"
        ),
        "teacher_confused_negatives": teacher_confused_negatives.head(limit).to_dict(
            orient="records"
        ),
    }
