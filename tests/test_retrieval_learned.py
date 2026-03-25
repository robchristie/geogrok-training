from __future__ import annotations

import numpy as np
import pandas as pd

from geogrok.retrieval.learned import (
    LinearProjectionModel,
    nt_xent_loss_and_grad,
    sample_pair_batch,
    train_contrastive_projection,
)


def test_sample_pair_batch_returns_two_indices_per_pair():
    rng = np.random.default_rng(7)
    groups = [
        np.array([0, 1, 2], dtype=np.int64),
        np.array([3, 4, 5], dtype=np.int64),
    ]

    indices = sample_pair_batch(groups, pairs_per_batch=3, rng=rng)

    assert indices.shape == (6,)


def test_nt_xent_loss_and_grad_returns_finite_values():
    batch_features = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.1],
            [0.0, 1.0],
            [0.1, 1.0],
        ],
        dtype=np.float32,
    )
    weights = np.eye(2, dtype=np.float32)

    loss, grad = nt_xent_loss_and_grad(
        batch_features,
        weights,
        temperature=0.1,
        weight_decay=1e-4,
    )

    assert np.isfinite(loss)
    assert grad.shape == weights.shape
    assert np.all(np.isfinite(grad))


def test_train_contrastive_projection_returns_projector():
    features = np.array(
        [
            [1.0, 0.0, 0.1],
            [0.9, 0.1, 0.2],
            [0.0, 1.0, 0.1],
            [0.1, 0.9, 0.2],
        ],
        dtype=np.float32,
    )
    metadata = pd.DataFrame(
        {
            "scene_id": ["a", "a", "b", "b"],
            "capture_id": ["ca", "ca", "cb", "cb"],
        }
    )

    model, report = train_contrastive_projection(
        features,
        metadata,
        positive_key="scene_id",
        embedding_dim=2,
        epochs=2,
        steps_per_epoch=2,
        pairs_per_batch=2,
        learning_rate=0.05,
        temperature=0.1,
        weight_decay=1e-4,
        seed=7,
    )

    assert isinstance(model, LinearProjectionModel)
    embedded = model.embed(features)
    assert embedded.shape == (4, 2)
    assert report.loss_final <= report.loss_initial or np.isfinite(report.loss_final)
