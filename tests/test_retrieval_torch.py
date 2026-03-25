from __future__ import annotations

import numpy as np
import pytest

from geogrok.retrieval.torch_encoder import build_positive_groups, create_model, nt_xent_loss

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
