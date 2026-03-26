from __future__ import annotations

import numpy as np
import pandas as pd

from geogrok.obs.review_tables import (
    build_disagreement_queues,
    build_failure_queues,
    ranked_pair_rows,
)


def test_ranked_pair_rows_and_failure_queues_surface_missed_positive_and_intrusive_negative():
    embeddings = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.9, 0.1],
        ],
        dtype=np.float32,
    )
    metadata = pd.DataFrame(
        [
            {"chip_id": "query", "split": "val"},
            {"chip_id": "positive", "split": "test"},
            {"chip_id": "negative", "split": "test"},
        ]
    )
    pairs = pd.DataFrame(
        [
            {
                "query_chip_id": "query",
                "candidate_chip_id": "positive",
                "pair_label": "positive_exact",
                "pair_group": "positive",
                "query_split": "val",
                "candidate_split": "test",
                "city": "UCSD",
                "modality": "PAN",
                "overlap_fraction": 0.7,
                "overlap_iou": 0.5,
                "time_delta_seconds": 86400.0,
                "center_distance_m": 10.0,
            },
            {
                "query_chip_id": "query",
                "candidate_chip_id": "negative",
                "pair_label": "negative_hard",
                "pair_group": "negative",
                "query_split": "val",
                "candidate_split": "test",
                "city": "UCSD",
                "modality": "PAN",
                "overlap_fraction": 0.0,
                "overlap_iou": 0.0,
                "time_delta_seconds": 86400.0,
                "center_distance_m": 1200.0,
            },
        ]
    )

    ranked = ranked_pair_rows(
        embeddings,
        metadata,
        pairs,
        query_splits=("val",),
        gallery_splits=("test",),
    )
    queues = build_failure_queues(ranked, top_k=1, limit=8)

    assert len(queues["false_negatives"]) == 1
    assert queues["false_negatives"][0]["candidate_chip_id"] == "positive"
    assert queues["false_negatives"][0]["rank"] == 2

    assert len(queues["false_positives"]) == 1
    assert queues["false_positives"][0]["candidate_chip_id"] == "negative"
    assert queues["false_positives"][0]["rank"] == 1


def test_build_disagreement_queues_surface_teacher_ahead_positive_and_student_confusion():
    base_pairs = pd.DataFrame(
        [
            {
                "query_chip_id": "query",
                "candidate_chip_id": "positive",
                "pair_label": "positive_exact",
                "pair_group": "positive",
                "query_split": "val",
                "candidate_split": "test",
                "city": "UCSD",
                "modality": "PAN",
                "overlap_fraction": 0.8,
                "overlap_iou": 0.6,
                "time_delta_seconds": 86400.0,
                "center_distance_m": 20.0,
            },
            {
                "query_chip_id": "query",
                "candidate_chip_id": "negative",
                "pair_label": "negative_hard",
                "pair_group": "negative",
                "query_split": "val",
                "candidate_split": "test",
                "city": "UCSD",
                "modality": "PAN",
                "overlap_fraction": 0.0,
                "overlap_iou": 0.0,
                "time_delta_seconds": 86400.0,
                "center_distance_m": 500.0,
            },
        ]
    )
    metadata = pd.DataFrame(
        [
            {"chip_id": "query", "split": "val"},
            {"chip_id": "positive", "split": "test"},
            {"chip_id": "negative", "split": "test"},
        ]
    )
    teacher_embeddings = np.asarray(
        [
            [1.0, 0.0],
            [0.95, 0.05],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    student_embeddings = np.asarray(
        [
            [1.0, 0.0],
            [0.2, 0.98],
            [0.9, 0.1],
        ],
        dtype=np.float32,
    )

    teacher_ranked = ranked_pair_rows(
        teacher_embeddings,
        metadata,
        base_pairs,
        query_splits=("val",),
        gallery_splits=("test",),
    )
    student_ranked = ranked_pair_rows(
        student_embeddings,
        metadata,
        base_pairs,
        query_splits=("val",),
        gallery_splits=("test",),
    )
    queues = build_disagreement_queues(teacher_ranked, student_ranked, limit=8)

    assert len(queues["teacher_ahead_positives"]) == 1
    assert queues["teacher_ahead_positives"][0]["candidate_chip_id"] == "positive"
    assert queues["teacher_ahead_positives"][0]["teacher_rank"] == 1
    assert queues["teacher_ahead_positives"][0]["student_rank"] == 2

    assert len(queues["student_confused_negatives"]) == 1
    assert queues["student_confused_negatives"][0]["candidate_chip_id"] == "negative"
    assert queues["student_confused_negatives"][0]["student_rank"] == 1
    assert queues["student_confused_negatives"][0]["teacher_rank"] == 2
