from __future__ import annotations

import numpy as np
import pandas as pd

from geogrok.retrieval.baseline import (
    SimplePanEmbedder,
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
        }
    )

    report = evaluate_retrieval(embeddings, metadata, positive_key="scene_id")

    assert report.queries_evaluated == 4
    assert report.recall_at_1 == 1.0
    assert report.recall_at_5 == 1.0
    assert report.mean_reciprocal_rank == 1.0
