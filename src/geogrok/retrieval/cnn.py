from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from geogrok.retrieval.baseline import (
    RetrievalReport,
    build_dataset,
    evaluate_retrieval,
    mean,
    percentile,
    safe_rate,
)
from geogrok.retrieval.learned import sample_pair_batch

DEFAULT_RUN_ROOT = Path("artifacts/runs/cnn-embedding-baseline")


@dataclass(frozen=True)
class PreprocessReport:
    samples: int
    image_size: int
    elapsed_seconds: float
    samples_per_second: float
    read_latency_ms_mean: float
    read_latency_ms_p95: float
    transform_latency_ms_mean: float
    transform_latency_ms_p95: float
    resize_latency_ms_mean: float
    resize_latency_ms_p95: float
    total_latency_ms_mean: float
    total_latency_ms_p95: float


@dataclass(frozen=True)
class CnnTrainingReport:
    epochs: int
    steps_per_epoch: int
    pairs_per_batch: int
    train_scenes: int
    elapsed_seconds: float
    steps_per_second: float
    pairs_per_second: float
    loss_initial: float
    loss_final: float
    loss_best: float


@dataclass(frozen=True)
class EmbeddingReport:
    samples: int
    embedding_dim: int
    elapsed_seconds: float
    samples_per_second: float
    embed_latency_ms_mean: float
    embed_latency_ms_p95: float


@dataclass(frozen=True)
class TinyCnnModel:
    conv1_w: np.ndarray
    conv1_b: np.ndarray
    conv2_w: np.ndarray
    conv2_b: np.ndarray
    proj_w: np.ndarray
    proj_b: np.ndarray

    @property
    def embedding_dim(self) -> int:
        return int(self.proj_b.shape[0])

    def state_dict(self) -> dict[str, np.ndarray]:
        return {
            "conv1_w": self.conv1_w,
            "conv1_b": self.conv1_b,
            "conv2_w": self.conv2_w,
            "conv2_b": self.conv2_b,
            "proj_w": self.proj_w,
            "proj_b": self.proj_b,
        }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a tiny CNN retrieval baseline on downsampled PAN chips."
    )
    parser.add_argument(
        "--chips-path",
        type=Path,
        default=Path("datasets/manifests/spacenet/chips.parquet"),
    )
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument(
        "--gdal-prefix",
        type=Path,
        help="Override the local GDAL/Kakadu install prefix.",
    )
    parser.add_argument(
        "--train-split",
        action="append",
        default=["train"],
        help="Split used for contrastive training. Repeat to add more splits.",
    )
    parser.add_argument(
        "--query-split",
        action="append",
        default=["val", "test"],
        help="Query split for retrieval evaluation. Repeat to add more splits.",
    )
    parser.add_argument(
        "--gallery-split",
        action="append",
        default=["val", "test"],
        help="Gallery split for retrieval evaluation. Repeat to add more splits.",
    )
    parser.add_argument(
        "--modality",
        action="append",
        default=["PAN"],
        help="Modality to include. Repeat to add more modalities.",
    )
    parser.add_argument("--train-limit", type=int, default=128)
    parser.add_argument("--eval-limit", type=int, default=128)
    parser.add_argument("--max-chips-per-scene", type=int, default=4)
    parser.add_argument("--min-chips-per-scene", type=int, default=2)
    parser.add_argument("--output-dtype", default="float32")
    parser.add_argument("--clip-min", type=float, default=0.0)
    parser.add_argument("--clip-max", type=float, default=2047.0)
    parser.add_argument("--scale-max", type=float, default=2047.0)
    parser.add_argument("--positive-key", choices=("scene_id", "capture_id"), default="scene_id")
    parser.add_argument("--min-positive-center-distance", type=float, default=1024.0)
    parser.add_argument("--allow-overlap-positives", action="store_true")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--conv1-channels", type=int, default=8)
    parser.add_argument("--conv2-channels", type=int, default=16)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--steps-per-epoch", type=int, default=16)
    parser.add_argument("--pairs-per-batch", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def extract_images(
    dataset,
    *,
    image_size: int,
) -> tuple[np.ndarray, np.ndarray, PreprocessReport, list[dict[str, object]]]:
    if image_size <= 0:
        raise ValueError("image_size must be positive.")

    records: list[dict[str, object]] = []
    images: list[np.ndarray] = []
    read_latencies: list[float] = []
    transform_latencies: list[float] = []
    resize_latencies: list[float] = []
    total_latencies: list[float] = []

    start = perf_counter()
    for index in range(len(dataset)):
        sample = dataset.sample(index)
        resize_start = perf_counter()
        resized = downsample_mean(sample.image[0], image_size)
        resize_ms = (perf_counter() - resize_start) * 1000.0
        total_ms = sample.timing.total_ms + resize_ms

        records.append(
            {
                "chip_id": sample.record.chip_id,
                "asset_id": sample.record.asset_id,
                "capture_id": sample.record.capture_id,
                "scene_id": sample.record.scene_id,
                "split": sample.record.split,
                "city": sample.record.city,
                "modality": sample.record.modality,
                "local_path": str(sample.record.local_path),
                "x0": sample.record.x0,
                "y0": sample.record.y0,
                "width": sample.record.width,
                "height": sample.record.height,
            }
        )
        images.append(resized[np.newaxis, :, :].astype(np.float32, copy=False))
        read_latencies.append(sample.timing.read_ms)
        transform_latencies.append(sample.timing.transform_ms)
        resize_latencies.append(resize_ms)
        total_latencies.append(total_ms)

    elapsed_seconds = perf_counter() - start
    image_array = np.stack(images, axis=0) if images else np.empty((0, 1, image_size, image_size))
    report = PreprocessReport(
        samples=len(records),
        image_size=image_size,
        elapsed_seconds=elapsed_seconds,
        samples_per_second=safe_rate(len(records), elapsed_seconds),
        read_latency_ms_mean=mean(read_latencies),
        read_latency_ms_p95=percentile(read_latencies, 95.0),
        transform_latency_ms_mean=mean(transform_latencies),
        transform_latency_ms_p95=percentile(transform_latencies, 95.0),
        resize_latency_ms_mean=mean(resize_latencies),
        resize_latency_ms_p95=percentile(resize_latencies, 95.0),
        total_latency_ms_mean=mean(total_latencies),
        total_latency_ms_p95=percentile(total_latencies, 95.0),
    )
    return image_array, np.arange(len(records), dtype=np.int64), report, records


def downsample_mean(image: np.ndarray, target_size: int) -> np.ndarray:
    source = np.asarray(image, dtype=np.float32)
    y_edges = np.linspace(0, source.shape[0], target_size + 1, dtype=int)
    x_edges = np.linspace(0, source.shape[1], target_size + 1, dtype=int)
    output = np.zeros((target_size, target_size), dtype=np.float32)
    for row in range(target_size):
        for col in range(target_size):
            block = source[y_edges[row] : y_edges[row + 1], x_edges[col] : x_edges[col + 1]]
            output[row, col] = float(block.mean()) if block.size else 0.0
    return output


def init_tiny_cnn(
    *,
    input_channels: int,
    conv1_channels: int,
    conv2_channels: int,
    embedding_dim: int,
    rng: np.random.Generator,
) -> TinyCnnModel:
    conv1_scale = np.sqrt(2.0 / (input_channels * 5 * 5))
    conv2_scale = np.sqrt(2.0 / (conv1_channels * 3 * 3))
    proj_scale = np.sqrt(2.0 / conv2_channels)
    return TinyCnnModel(
        conv1_w=(rng.normal(0.0, conv1_scale, size=(conv1_channels, input_channels, 5, 5))).astype(
            np.float32
        ),
        conv1_b=np.zeros(conv1_channels, dtype=np.float32),
        conv2_w=(rng.normal(0.0, conv2_scale, size=(conv2_channels, conv1_channels, 3, 3))).astype(
            np.float32
        ),
        conv2_b=np.zeros(conv2_channels, dtype=np.float32),
        proj_w=(rng.normal(0.0, proj_scale, size=(conv2_channels, embedding_dim))).astype(
            np.float32
        ),
        proj_b=np.zeros(embedding_dim, dtype=np.float32),
    )


def cnn_forward(
    model: TinyCnnModel,
    images: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    conv1, conv1_cache = conv2d_forward(images, model.conv1_w, model.conv1_b, stride=2, padding=2)
    relu1 = np.maximum(conv1, 0.0)
    conv2, conv2_cache = conv2d_forward(relu1, model.conv2_w, model.conv2_b, stride=2, padding=1)
    relu2 = np.maximum(conv2, 0.0)
    pooled = relu2.mean(axis=(2, 3))
    projected = pooled @ model.proj_w + model.proj_b
    embeddings = row_normalize(projected.astype(np.float32, copy=False))
    cache = {
        "input": images,
        "conv1": conv1,
        "relu1": relu1,
        "conv2": conv2,
        "relu2": relu2,
        "pooled": pooled,
        "projected": projected,
        "conv1_cache": conv1_cache,
        "conv2_cache": conv2_cache,
    }
    return embeddings, cache


def cnn_backward(
    model: TinyCnnModel,
    cache: dict[str, Any],
    grad_embeddings: np.ndarray,
) -> dict[str, np.ndarray]:
    projected = cache["projected"]
    pooled = cache["pooled"]
    relu2 = cache["relu2"]
    conv2 = cache["conv2"]
    conv1 = cache["conv1"]

    grad_projected = backward_row_normalize(grad_embeddings, projected)
    grad_proj_w = pooled.T @ grad_projected
    grad_proj_b = grad_projected.sum(axis=0)
    grad_pooled = grad_projected @ model.proj_w.T

    grad_relu2 = np.broadcast_to(
        grad_pooled[:, :, None, None] / (relu2.shape[2] * relu2.shape[3]),
        relu2.shape,
    ).copy()
    grad_conv2 = grad_relu2 * (conv2 > 0.0)
    grad_relu1, grad_conv2_w, grad_conv2_b = conv2d_backward(
        grad_conv2,
        cache["conv2_cache"],
        model.conv2_w,
        stride=2,
        padding=1,
    )
    grad_conv1 = grad_relu1 * (conv1 > 0.0)
    _grad_input, grad_conv1_w, grad_conv1_b = conv2d_backward(
        grad_conv1,
        cache["conv1_cache"],
        model.conv1_w,
        stride=2,
        padding=2,
    )
    return {
        "conv1_w": grad_conv1_w,
        "conv1_b": grad_conv1_b,
        "conv2_w": grad_conv2_w,
        "conv2_b": grad_conv2_b,
        "proj_w": grad_proj_w,
        "proj_b": grad_proj_b,
    }


def conv2d_forward(
    x: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    *,
    stride: int,
    padding: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    x_padded = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode="constant")
    cols = im2col(x_padded, kernel_h=weights.shape[2], kernel_w=weights.shape[3], stride=stride)
    batch_size = x.shape[0]
    out_h = (x_padded.shape[2] - weights.shape[2]) // stride + 1
    out_w = (x_padded.shape[3] - weights.shape[3]) // stride + 1
    weights_flat = weights.reshape(weights.shape[0], -1)
    out = cols @ weights_flat.T + bias
    out = out.reshape(batch_size, out_h, out_w, weights.shape[0]).transpose(0, 3, 1, 2)
    cache = {
        "x_shape": np.array(x.shape, dtype=np.int64),
        "x_padded_shape": np.array(x_padded.shape, dtype=np.int64),
        "cols": cols,
        "out_h": np.array([out_h], dtype=np.int64),
        "out_w": np.array([out_w], dtype=np.int64),
    }
    return out.astype(np.float32, copy=False), cache


def conv2d_backward(
    dout: np.ndarray,
    cache: dict[str, Any],
    weights: np.ndarray,
    *,
    stride: int,
    padding: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cols = cache["cols"]
    x_shape = tuple(int(value) for value in cache["x_shape"])
    x_padded_shape = tuple(int(value) for value in cache["x_padded_shape"])
    out_h = int(cache["out_h"][0])
    out_w = int(cache["out_w"][0])
    batch_size = x_shape[0]

    dout_cols = dout.transpose(0, 2, 3, 1).reshape(batch_size * out_h * out_w, weights.shape[0])
    weights_flat = weights.reshape(weights.shape[0], -1)

    grad_weights = dout_cols.T @ cols
    grad_weights = grad_weights.reshape(weights.shape).astype(np.float32, copy=False)
    grad_bias = dout_cols.sum(axis=0).astype(np.float32, copy=False)
    dcols = dout_cols @ weights_flat

    grad_padded = col2im(
        dcols,
        x_padded_shape=x_padded_shape,
        kernel_h=weights.shape[2],
        kernel_w=weights.shape[3],
        stride=stride,
        out_h=out_h,
        out_w=out_w,
    )
    if padding > 0:
        grad_input = grad_padded[:, :, padding:-padding, padding:-padding]
    else:
        grad_input = grad_padded
    return grad_input.astype(np.float32, copy=False), grad_weights, grad_bias


def im2col(
    x_padded: np.ndarray,
    *,
    kernel_h: int,
    kernel_w: int,
    stride: int,
) -> np.ndarray:
    batch_size, channels, _height, _width = x_padded.shape
    out_h = (x_padded.shape[2] - kernel_h) // stride + 1
    out_w = (x_padded.shape[3] - kernel_w) // stride + 1
    cols = np.empty((batch_size * out_h * out_w, channels * kernel_h * kernel_w), dtype=np.float32)
    row = 0
    for batch_index in range(batch_size):
        for out_row in range(out_h):
            y0 = out_row * stride
            for out_col in range(out_w):
                x0 = out_col * stride
                patch = x_padded[
                    batch_index,
                    :,
                    y0 : y0 + kernel_h,
                    x0 : x0 + kernel_w,
                ]
                cols[row] = patch.reshape(-1)
                row += 1
    return cols


def col2im(
    cols: np.ndarray,
    *,
    x_padded_shape: Sequence[int],
    kernel_h: int,
    kernel_w: int,
    stride: int,
    out_h: int,
    out_w: int,
) -> np.ndarray:
    batch_size, channels, height, width = (int(value) for value in x_padded_shape)
    reshaped = cols.reshape(batch_size, out_h, out_w, channels, kernel_h, kernel_w)
    reshaped = reshaped.transpose(0, 3, 1, 2, 4, 5)
    output = np.zeros((batch_size, channels, height, width), dtype=np.float32)
    for kernel_row in range(kernel_h):
        for kernel_col in range(kernel_w):
            output[
                :,
                :,
                kernel_row : kernel_row + out_h * stride : stride,
                kernel_col : kernel_col + out_w * stride : stride,
            ] += reshaped[:, :, :, :, kernel_row, kernel_col]
    return output


def backward_row_normalize(grad_normalized: np.ndarray, values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    normalized = values / norms
    projection = np.sum(grad_normalized * normalized, axis=1, keepdims=True)
    return (grad_normalized - normalized * projection) / norms


def train_tiny_cnn(
    images: np.ndarray,
    metadata_records: Sequence[dict[str, object]],
    *,
    positive_key: str,
    conv1_channels: int,
    conv2_channels: int,
    embedding_dim: int,
    epochs: int,
    steps_per_epoch: int,
    pairs_per_batch: int,
    learning_rate: float,
    temperature: float,
    weight_decay: float,
    seed: int,
) -> tuple[TinyCnnModel, CnnTrainingReport]:
    metadata = pd.DataFrame(metadata_records)
    metadata["positive_label"] = metadata[positive_key].fillna("null").astype(str)
    groups = [
        group.index.to_numpy(dtype=np.int64)
        for _, group in metadata.groupby("positive_label", sort=False)
        if len(group) >= 2
    ]
    if not groups:
        raise ValueError("No positive groups with at least two chips were found for CNN training.")

    rng = np.random.default_rng(seed)
    model = init_tiny_cnn(
        input_channels=images.shape[1],
        conv1_channels=conv1_channels,
        conv2_channels=conv2_channels,
        embedding_dim=embedding_dim,
        rng=rng,
    )
    params = model.state_dict()
    moment1 = {name: np.zeros_like(value) for name, value in params.items()}
    moment2 = {name: np.zeros_like(value) for name, value in params.items()}
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    global_step = 0
    losses: list[float] = []

    start = perf_counter()
    for _epoch in range(epochs):
        for _step in range(steps_per_epoch):
            batch_indices = sample_pair_batch(groups, pairs_per_batch=pairs_per_batch, rng=rng)
            batch_images = images[batch_indices]
            embeddings, cache = cnn_forward(model, batch_images)
            loss, grad_embeddings = nt_xent_from_embeddings(
                embeddings,
                temperature=temperature,
            )
            grads = cnn_backward(model, cache, grad_embeddings)
            for name, value in params.items():
                grad = grads[name] + weight_decay * value
                global_step += 1
                moment1[name] = beta1 * moment1[name] + (1.0 - beta1) * grad
                moment2[name] = beta2 * moment2[name] + (1.0 - beta2) * (grad * grad)
                m_hat = moment1[name] / (1.0 - beta1**global_step)
                v_hat = moment2[name] / (1.0 - beta2**global_step)
                params[name] = value - learning_rate * m_hat / (np.sqrt(v_hat) + eps)
            model = TinyCnnModel(**params)
            losses.append(loss)
    elapsed_seconds = perf_counter() - start

    total_steps = epochs * steps_per_epoch
    total_pairs = total_steps * pairs_per_batch
    report = CnnTrainingReport(
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pairs_per_batch=pairs_per_batch,
        train_scenes=len(groups),
        elapsed_seconds=elapsed_seconds,
        steps_per_second=safe_rate(total_steps, elapsed_seconds),
        pairs_per_second=safe_rate(total_pairs, elapsed_seconds),
        loss_initial=float(losses[0]),
        loss_final=float(losses[-1]),
        loss_best=float(min(losses)),
    )
    return model, report


def nt_xent_from_embeddings(
    embeddings: np.ndarray,
    *,
    temperature: float,
) -> tuple[float, np.ndarray]:
    logits = (embeddings @ embeddings.T) / temperature
    np.fill_diagonal(logits, -np.inf)

    max_logits = np.max(logits, axis=1, keepdims=True)
    stable_logits = logits - max_logits
    exp_logits = np.exp(stable_logits)
    np.fill_diagonal(exp_logits, 0.0)
    partition = exp_logits.sum(axis=1, keepdims=True)
    probs = exp_logits / np.where(partition == 0.0, 1.0, partition)

    n = embeddings.shape[0]
    positives = np.arange(n) ^ 1
    targets = np.zeros_like(probs)
    targets[np.arange(n), positives] = 1.0
    positive_prob = np.clip(probs[np.arange(n), positives], 1e-12, None)
    loss = float(-np.log(positive_prob).mean())

    grad_logits = (probs - targets) / n
    grad_embeddings = ((grad_logits + grad_logits.T) @ embeddings) / temperature
    return loss, grad_embeddings.astype(np.float32, copy=False)


def embed_images(
    model: TinyCnnModel,
    images: np.ndarray,
    *,
    batch_size: int = 32,
) -> tuple[np.ndarray, EmbeddingReport]:
    outputs: list[np.ndarray] = []
    latencies: list[float] = []
    start = perf_counter()
    for offset in range(0, len(images), batch_size):
        batch = images[offset : offset + batch_size]
        batch_start = perf_counter()
        embeddings, _cache = cnn_forward(model, batch)
        latencies.append((perf_counter() - batch_start) * 1000.0 / max(len(batch), 1))
        outputs.append(embeddings)
    elapsed_seconds = perf_counter() - start
    matrix = np.concatenate(outputs, axis=0) if outputs else np.empty((0, model.embedding_dim))
    report = EmbeddingReport(
        samples=len(matrix),
        embedding_dim=model.embedding_dim,
        elapsed_seconds=elapsed_seconds,
        samples_per_second=safe_rate(len(matrix), elapsed_seconds),
        embed_latency_ms_mean=mean(latencies),
        embed_latency_ms_p95=percentile(latencies, 95.0),
    )
    return matrix.astype(np.float32, copy=False), report


def write_outputs(
    *,
    model: TinyCnnModel,
    train_images: np.ndarray,
    eval_embeddings: np.ndarray,
    train_metadata_records: Sequence[dict[str, object]],
    eval_metadata_records: Sequence[dict[str, object]],
    train_preprocess_report: PreprocessReport,
    eval_preprocess_report: PreprocessReport,
    training_report: CnnTrainingReport,
    embedding_report: EmbeddingReport,
    retrieval_report: RetrievalReport,
    run_root: Path,
) -> dict[str, Path]:
    output_root = run_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    model_path = output_root / "model.npz"
    train_images_path = output_root / "train_images.npy"
    eval_embeddings_path = output_root / "eval_embeddings.npy"
    train_index_path = output_root / "train_index.parquet"
    eval_index_path = output_root / "eval_index.parquet"
    train_preprocess_path = output_root / "train_preprocess_benchmark.json"
    eval_preprocess_path = output_root / "eval_preprocess_benchmark.json"
    training_path = output_root / "training.json"
    embedding_path = output_root / "embedding.json"
    retrieval_path = output_root / "retrieval.json"

    state = model.state_dict()
    np.savez(
        model_path,
        conv1_w=state["conv1_w"],
        conv1_b=state["conv1_b"],
        conv2_w=state["conv2_w"],
        conv2_b=state["conv2_b"],
        proj_w=state["proj_w"],
        proj_b=state["proj_b"],
    )
    np.save(train_images_path, train_images)
    np.save(eval_embeddings_path, eval_embeddings)
    pd.DataFrame(list(train_metadata_records)).to_parquet(
        train_index_path, index=False, compression="zstd"
    )
    pd.DataFrame(list(eval_metadata_records)).to_parquet(
        eval_index_path, index=False, compression="zstd"
    )
    train_preprocess_path.write_text(
        json.dumps(asdict(train_preprocess_report), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    eval_preprocess_path.write_text(
        json.dumps(asdict(eval_preprocess_report), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    training_path.write_text(
        json.dumps(asdict(training_report), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    embedding_path.write_text(
        json.dumps(asdict(embedding_report), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    retrieval_path.write_text(
        json.dumps(asdict(retrieval_report), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {
        "model": model_path,
        "train_images": train_images_path,
        "eval_embeddings": eval_embeddings_path,
        "train_index": train_index_path,
        "eval_index": eval_index_path,
        "train_preprocess": train_preprocess_path,
        "eval_preprocess": eval_preprocess_path,
        "training": training_path,
        "embedding": embedding_path,
        "retrieval": retrieval_path,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    chips_path = args.chips_path.resolve()
    run_root = args.run_root.resolve()
    gdal_prefix = args.gdal_prefix.resolve() if args.gdal_prefix is not None else None
    train_splits = tuple(dict.fromkeys(args.train_split))
    query_splits = tuple(dict.fromkeys(args.query_split))
    gallery_splits = tuple(dict.fromkeys(args.gallery_split))
    eval_splits = tuple(sorted(set(query_splits).union(gallery_splits)))

    train_dataset = build_dataset(
        chips_path,
        splits=train_splits,
        modalities=tuple(args.modality),
        limit=args.train_limit,
        min_chips_per_scene=args.min_chips_per_scene,
        max_chips_per_scene=args.max_chips_per_scene,
        gdal_prefix=gdal_prefix,
        output_dtype=args.output_dtype,
        clip_min=args.clip_min,
        clip_max=args.clip_max,
        scale_max=args.scale_max,
    )
    eval_dataset = build_dataset(
        chips_path,
        splits=eval_splits,
        modalities=tuple(args.modality),
        limit=args.eval_limit,
        min_chips_per_scene=args.min_chips_per_scene,
        max_chips_per_scene=args.max_chips_per_scene,
        gdal_prefix=gdal_prefix,
        output_dtype=args.output_dtype,
        clip_min=args.clip_min,
        clip_max=args.clip_max,
        scale_max=args.scale_max,
    )
    if len(train_dataset) == 0 or len(eval_dataset) == 0:
        raise SystemExit("Train or evaluation dataset is empty after filtering.")

    train_images, _train_indices, train_preprocess_report, train_records = extract_images(
        train_dataset,
        image_size=args.image_size,
    )
    eval_images, _eval_indices, eval_preprocess_report, eval_records = extract_images(
        eval_dataset,
        image_size=args.image_size,
    )

    model, training_report = train_tiny_cnn(
        train_images,
        train_records,
        positive_key=args.positive_key,
        conv1_channels=args.conv1_channels,
        conv2_channels=args.conv2_channels,
        embedding_dim=args.embedding_dim,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        pairs_per_batch=args.pairs_per_batch,
        learning_rate=args.learning_rate,
        temperature=args.temperature,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )
    eval_embeddings, embedding_report = embed_images(model, eval_images)
    eval_metadata = pd.DataFrame(eval_records)
    retrieval_report = evaluate_retrieval(
        eval_embeddings,
        eval_metadata,
        positive_key=args.positive_key,
        query_splits=query_splits,
        gallery_splits=gallery_splits,
        min_positive_center_distance=args.min_positive_center_distance,
        allow_overlap_positives=args.allow_overlap_positives,
    )
    paths = write_outputs(
        model=model,
        train_images=train_images,
        eval_embeddings=eval_embeddings,
        train_metadata_records=train_records,
        eval_metadata_records=eval_records,
        train_preprocess_report=train_preprocess_report,
        eval_preprocess_report=eval_preprocess_report,
        training_report=training_report,
        embedding_report=embedding_report,
        retrieval_report=retrieval_report,
        run_root=run_root,
    )

    print(f"chips: {chips_path}")
    print(f"train_samples: {len(train_images)} eval_samples: {len(eval_images)}")
    print(
        f"protocol: query_splits={query_splits} gallery_splits={gallery_splits} "
        f"queries={retrieval_report.query_count} gallery={retrieval_report.gallery_count}"
    )
    print(
        f"preprocess_train: size={args.image_size} "
        f"samples/s={train_preprocess_report.samples_per_second:.2f} "
        f"read_p95={train_preprocess_report.read_latency_ms_p95:.2f} "
        f"resize_p95={train_preprocess_report.resize_latency_ms_p95:.2f}"
    )
    print(
        f"training: emb_dim={model.embedding_dim} "
        f"loss={training_report.loss_initial:.4f}->{training_report.loss_final:.4f} "
        f"pairs/s={training_report.pairs_per_second:.2f}"
    )
    print(
        f"embed_eval: samples/s={embedding_report.samples_per_second:.2f} "
        f"embed_p95={embedding_report.embed_latency_ms_p95:.2f}"
    )
    print(
        f"retrieval: R@1={retrieval_report.recall_at_1:.3f} "
        f"R@5={retrieval_report.recall_at_5:.3f} "
        f"R@10={retrieval_report.recall_at_10:.3f} "
        f"MRR={retrieval_report.mean_reciprocal_rank:.3f}"
    )
    print(f"model: {paths['model']}")
    print(f"train_preprocess: {paths['train_preprocess']}")
    print(f"eval_preprocess: {paths['eval_preprocess']}")
    print(f"training_report: {paths['training']}")
    print(f"embedding_report: {paths['embedding']}")
    print(f"retrieval: {paths['retrieval']}")
    return 0


def row_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return matrix / norms


if __name__ == "__main__":
    raise SystemExit(main())
