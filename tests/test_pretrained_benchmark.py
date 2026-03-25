from __future__ import annotations

import numpy as np
import pytest

from geogrok.retrieval.pretrained_benchmark import model_batch_inputs, normalize_multi_arg

torch = pytest.importorskip("torch")


def test_normalize_multi_arg_deduplicates_in_order():
    values = normalize_multi_arg(["resnet18", "resnet50", "resnet18"], default=("vit_b_16",))
    assert values == ("resnet18", "resnet50")


def test_model_batch_inputs_repeats_pan_channel_and_resizes():
    batch = np.random.default_rng(42).random((2, 1, 128, 128), dtype=np.float32)
    prepared = model_batch_inputs(
        torch,
        batch,
        input_size=224,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        device_name="cpu",
    )
    assert tuple(prepared.shape) == (2, 3, 224, 224)
    assert prepared.dtype == torch.float32
