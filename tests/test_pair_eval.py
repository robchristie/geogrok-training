from __future__ import annotations

import numpy as np
import pandas as pd

from geogrok.retrieval.pair_eval import chip_ids_from_pairs, evaluate_pair_retrieval


def test_chip_ids_from_pairs_collects_query_and_candidate_ids():
    pairs = pd.DataFrame(
        [
            {"query_chip_id": "a", "candidate_chip_id": "b"},
            {"query_chip_id": "b", "candidate_chip_id": "c"},
        ]
    )
    assert chip_ids_from_pairs(pairs) == {"a", "b", "c"}


def test_evaluate_pair_retrieval_scores_exact_any_and_hard_negatives():
    embeddings = np.array(
        [
            [1.0, 0.0],   # q1
            [0.99, 0.01], # p1 exact
            [0.9, 0.1],   # w1 weak
            [0.0, 1.0],   # n1 hard negative
        ],
        dtype=np.float32,
    )
    metadata = pd.DataFrame(
        [
            {"chip_id": "q1", "split": "train"},
            {"chip_id": "p1", "split": "train"},
            {"chip_id": "w1", "split": "train"},
            {"chip_id": "n1", "split": "train"},
        ]
    )
    pairs = pd.DataFrame(
        [
            {"query_chip_id": "q1", "candidate_chip_id": "p1", "pair_label": "positive_exact"},
            {"query_chip_id": "q1", "candidate_chip_id": "w1", "pair_label": "positive_weak"},
            {"query_chip_id": "q1", "candidate_chip_id": "n1", "pair_label": "negative_hard"},
        ]
    )

    report = evaluate_pair_retrieval(
        embeddings,
        metadata,
        pairs,
        query_splits=("train",),
        gallery_splits=("train",),
    )

    assert report.exact_recall_at_1 == 1.0
    assert report.any_recall_at_1 == 1.0
    assert report.hard_negative_at_1_rate == 0.0
    assert report.queries_evaluated_exact == 1
    assert report.queries_evaluated_any == 1
