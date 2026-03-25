from __future__ import annotations

import numpy as np

from geogrok.retrieval.cnn import (
    cnn_forward,
    downsample_mean,
    init_tiny_cnn,
    nt_xent_from_embeddings,
)


def test_downsample_mean_preserves_constant_image():
    image = np.full((16, 16), fill_value=7.0, dtype=np.float32)
    reduced = downsample_mean(image, target_size=4)
    assert reduced.shape == (4, 4)
    assert np.allclose(reduced, 7.0)


def test_tiny_cnn_forward_returns_expected_embedding_shape():
    rng = np.random.default_rng(3)
    model = init_tiny_cnn(
        input_channels=1,
        conv1_channels=4,
        conv2_channels=8,
        embedding_dim=16,
        rng=rng,
    )
    images = rng.normal(size=(2, 1, 64, 64)).astype(np.float32)

    embeddings, _cache = cnn_forward(model, images)

    assert embeddings.shape == (2, 16)


def test_nt_xent_from_embeddings_returns_finite_loss():
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
        ],
        dtype=np.float32,
    )
    loss, grad = nt_xent_from_embeddings(embeddings, temperature=0.1)
    assert np.isfinite(loss)
    assert grad.shape == embeddings.shape
