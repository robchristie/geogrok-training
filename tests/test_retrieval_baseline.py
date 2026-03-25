from __future__ import annotations

import numpy as np
import pandas as pd

from geogrok.retrieval.baseline import (
    SimplePanEmbedder,
    balanced_subset,
    evaluate_retrieval,
    reduce_profile,
)


def test_simple_pan_embedder_returns_fixed_dimension():
    image = np.ones((1, 32, 32), dtype=np.float32)
    embedder = SimplePanEmbedder(intensity_bins=8, coarse_grid=4, profile_bins=6)

    embedding = embedder.embed(image)

    assert embedding.shape == (4 * 4 + 4 * 4 + 8 + 6 + 6,)
    assert np.isclose(np.linalg.norm(embedding), 1.0)


def test_reduce_profile_aggregates_to_requested_bins():
    profile = np.arange(16, dtype=np.float32)
    reduced = reduce_profile(profile, bins=4)
    assert reduced.shape == (4,)
    assert np.allclose(reduced, np.array([1.5, 5.5, 9.5, 13.5], dtype=np.float32))


def test_evaluate_retrieval_scores_perfect_when_positives_are_nearest():
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.99, 0.01],
            [0.0, 1.0],
            [0.01, 0.99],
        ],
        dtype=np.float32,
    )
    metadata = pd.DataFrame(
        {
            "chip_id": ["a1", "a2", "b1", "b2"],
            "scene_id": ["scene_a", "scene_a", "scene_b", "scene_b"],
            "capture_id": ["cap_a", "cap_a", "cap_b", "cap_b"],
            "split": ["train", "train", "train", "train"],
            "x0": [0, 1024, 0, 1024],
            "y0": [0, 0, 0, 0],
            "width": [1024, 1024, 1024, 1024],
            "height": [1024, 1024, 1024, 1024],
        }
    )

    report = evaluate_retrieval(
        embeddings,
        metadata,
        positive_key="scene_id",
        query_splits=("train",),
        gallery_splits=("train",),
        min_positive_center_distance=1024.0,
        allow_overlap_positives=False,
    )

    assert report.queries_evaluated == 4
    assert report.recall_at_1 == 1.0
    assert report.recall_at_5 == 1.0
    assert report.mean_reciprocal_rank == 1.0


def test_balanced_subset_round_robins_across_scenes():
    frame = pd.DataFrame(
        {
            "chip_id": ["a1", "a2", "a3", "b1", "b2", "c1"],
            "scene_id": ["a", "a", "a", "b", "b", "c"],
            "split": ["train"] * 6,
            "city": ["X"] * 6,
        }
    )

    selected = balanced_subset(
        frame,
        group_key="scene_id",
        min_per_group=2,
        max_per_group=2,
        limit=5,
    )

    assert list(selected["chip_id"]) == ["a1", "b1", "a2", "b2"]
