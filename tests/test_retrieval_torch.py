from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from geogrok.retrieval.torch_encoder import (
    build_explicit_positive_pairs,
    build_positive_groups,
    create_model,
    nt_xent_loss,
    sample_explicit_pair_batch,
)

torch = pytest.importorskip("torch")


def test_build_positive_groups_filters_singletons():
    groups = build_positive_groups(
        [
            {"scene_id": "a"},
            {"scene_id": "a"},
            {"scene_id": "b"},
        ],
        positive_key="scene_id",
    )
    assert len(groups) == 1
    assert np.array_equal(groups[0], np.array([0, 1], dtype=np.int64))


def test_build_explicit_positive_pairs_deduplicates_directional_rows():
    pairs, exact_pairs, weak_pairs = build_explicit_positive_pairs(
        [
            {"chip_id": "chip_a"},
            {"chip_id": "chip_b"},
            {"chip_id": "chip_c"},
        ],
        pd.DataFrame(
            [
                {
                    "query_chip_id": "chip_a",
                    "candidate_chip_id": "chip_b",
                    "pair_label": "positive_exact",
                },
                {
                    "query_chip_id": "chip_b",
                    "candidate_chip_id": "chip_a",
                    "pair_label": "positive_exact",
                },
                {
                    "query_chip_id": "chip_a",
                    "candidate_chip_id": "chip_c",
                    "pair_label": "positive_weak",
                },
            ]
        ),
    )
    assert np.array_equal(pairs, np.array([[0, 1], [0, 2]], dtype=np.int64))
    assert exact_pairs == 1
    assert weak_pairs == 1


def test_sample_explicit_pair_batch_returns_two_indices_per_pair():
    indices = sample_explicit_pair_batch(
        np.array([[0, 1], [2, 3]], dtype=np.int64),
        pairs_per_batch=3,
        rng=np.random.default_rng(42),
    )
    assert indices.shape == (6,)
    assert set(indices).issubset({0, 1, 2, 3})


def test_create_model_returns_expected_embedding_shape():
    model = create_model(
        torch,
        input_channels=1,
        base_channels=8,
        embedding_dim=16,
        dropout=0.0,
    )
    batch = torch.randn(4, 1, 64, 64, dtype=torch.float32)
    embeddings = model(batch)
    assert tuple(embeddings.shape) == (4, 16)


def test_nt_xent_loss_is_finite():
    embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
        ],
        dtype=torch.float32,
    )
    loss = nt_xent_loss(torch, embeddings, temperature=0.1)
    assert torch.isfinite(loss)
