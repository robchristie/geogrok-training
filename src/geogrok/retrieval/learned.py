from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from geogrok.retrieval.baseline import (
    RetrievalReport,
    SimplePanEmbedder,
    build_dataset,
    evaluate_retrieval,
    mean,
    percentile,
    safe_rate,
)

DEFAULT_RUN_ROOT = Path("artifacts/runs/learned-embedding-baseline")


@dataclass(frozen=True)
class FeatureExtractionReport:
    samples: int
    feature_dim: int
    elapsed_seconds: float
    samples_per_second: float
    read_latency_ms_mean: float
    read_latency_ms_p95: float
    transform_latency_ms_mean: float
    transform_latency_ms_p95: float
    feature_latency_ms_mean: float
    feature_latency_ms_p95: float
    total_latency_ms_mean: float
    total_latency_ms_p95: float


@dataclass(frozen=True)
class ContrastiveTrainingReport:
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
class LinearProjectionModel:
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    embedding_dim: int

    def embed(self, features: np.ndarray) -> np.ndarray:
        standardized = (features - self.mean) / self.scale
        projected = standardized @ self.weights
        return row_normalize(projected.astype(np.float32, copy=False))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a shallow contrastive embedding model over PAN chip features."
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
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--steps-per-epoch", type=int, default=16)
    parser.add_argument("--pairs-per-batch", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def extract_feature_dataset(
    dataset,
    *,
    embedder: SimplePanEmbedder,
) -> tuple[np.ndarray, pd.DataFrame, FeatureExtractionReport]:
    records: list[dict[str, object]] = []
    features: list[np.ndarray] = []
    read_latencies: list[float] = []
    transform_latencies: list[float] = []
    feature_latencies: list[float] = []
    total_latencies: list[float] = []

    start = perf_counter()
    for index in range(len(dataset)):
        sample = dataset.sample(index)
        feature_start = perf_counter()
        feature_vector = embedder.features(sample.image)
        feature_ms = (perf_counter() - feature_start) * 1000.0
        total_ms = sample.timing.total_ms + feature_ms

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
        features.append(feature_vector)
        read_latencies.append(sample.timing.read_ms)
        transform_latencies.append(sample.timing.transform_ms)
        feature_latencies.append(feature_ms)
        total_latencies.append(total_ms)

    elapsed_seconds = perf_counter() - start
    matrix = np.stack(features, axis=0) if features else np.empty((0, embedder.embedding_dim))
    report = FeatureExtractionReport(
        samples=len(records),
        feature_dim=int(matrix.shape[1]) if matrix.size else embedder.embedding_dim,
        elapsed_seconds=elapsed_seconds,
        samples_per_second=safe_rate(len(records), elapsed_seconds),
        read_latency_ms_mean=mean(read_latencies),
        read_latency_ms_p95=percentile(read_latencies, 95.0),
        transform_latency_ms_mean=mean(transform_latencies),
        transform_latency_ms_p95=percentile(transform_latencies, 95.0),
        feature_latency_ms_mean=mean(feature_latencies),
        feature_latency_ms_p95=percentile(feature_latencies, 95.0),
        total_latency_ms_mean=mean(total_latencies),
        total_latency_ms_p95=percentile(total_latencies, 95.0),
    )
    return matrix.astype(np.float32, copy=False), pd.DataFrame(records), report


def train_contrastive_projection(
    features: np.ndarray,
    metadata: pd.DataFrame,
    *,
    positive_key: str,
    embedding_dim: int,
    epochs: int,
    steps_per_epoch: int,
    pairs_per_batch: int,
    learning_rate: float,
    temperature: float,
    weight_decay: float,
    seed: int,
) -> tuple[LinearProjectionModel, ContrastiveTrainingReport]:
    if len(features) != len(metadata):
        raise ValueError("features and metadata must have the same row count.")
    if embedding_dim <= 0:
        raise ValueError("embedding_dim must be positive.")
    if epochs <= 0 or steps_per_epoch <= 0 or pairs_per_batch <= 0:
        raise ValueError("epochs, steps_per_epoch, and pairs_per_batch must be positive.")

    frame = metadata.reset_index(drop=True).copy()
    frame["positive_label"] = frame[positive_key].fillna("null").astype(str)
    groups = [
        group.index.to_numpy(dtype=np.int64)
        for _, group in frame.groupby("positive_label", sort=False)
        if len(group) >= 2
    ]
    if not groups:
        raise ValueError("No positive groups with at least two chips were found for training.")

    x_mean = features.mean(axis=0, keepdims=True)
    x_scale = features.std(axis=0, keepdims=True)
    x_scale = np.where(x_scale < 1e-6, 1.0, x_scale)
    x = ((features - x_mean) / x_scale).astype(np.float32, copy=False)

    rng = np.random.default_rng(seed)
    weights = rng.normal(0.0, 0.05, size=(x.shape[1], embedding_dim)).astype(np.float32)
    m = np.zeros_like(weights)
    v = np.zeros_like(weights)
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    global_step = 0
    losses: list[float] = []

    start = perf_counter()
    for _epoch in range(epochs):
        for _step in range(steps_per_epoch):
            batch_indices = sample_pair_batch(groups, pairs_per_batch=pairs_per_batch, rng=rng)
            loss, grad = nt_xent_loss_and_grad(
                x[batch_indices],
                weights,
                temperature=temperature,
                weight_decay=weight_decay,
            )
            global_step += 1
            m = beta1 * m + (1.0 - beta1) * grad
            v = beta2 * v + (1.0 - beta2) * (grad * grad)
            m_hat = m / (1.0 - beta1**global_step)
            v_hat = v / (1.0 - beta2**global_step)
            weights = weights - learning_rate * m_hat / (np.sqrt(v_hat) + eps)
            losses.append(loss)
    elapsed_seconds = perf_counter() - start

    model = LinearProjectionModel(
        mean=x_mean.astype(np.float32, copy=False),
        scale=x_scale.astype(np.float32, copy=False),
        weights=weights.astype(np.float32, copy=False),
        embedding_dim=embedding_dim,
    )
    total_steps = epochs * steps_per_epoch
    total_pairs = total_steps * pairs_per_batch
    report = ContrastiveTrainingReport(
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


def sample_pair_batch(
    groups: Sequence[np.ndarray],
    *,
    pairs_per_batch: int,
    rng: np.random.Generator,
) -> np.ndarray:
    selected_groups = rng.choice(
        len(groups),
        size=pairs_per_batch,
        replace=len(groups) < pairs_per_batch,
    )
    indices: list[int] = []
    for group_index in selected_groups:
        group = groups[int(group_index)]
        pair = rng.choice(group, size=2, replace=False)
        indices.extend(int(value) for value in pair)
    return np.array(indices, dtype=np.int64)


def nt_xent_loss_and_grad(
    batch_features: np.ndarray,
    weights: np.ndarray,
    *,
    temperature: float,
    weight_decay: float,
) -> tuple[float, np.ndarray]:
    y = batch_features @ weights
    z, dy_dz = normalize_rows_with_backward(y)
    logits = (z @ z.T) / temperature
    np.fill_diagonal(logits, -np.inf)

    max_logits = np.max(logits, axis=1, keepdims=True)
    stable_logits = logits - max_logits
    exp_logits = np.exp(stable_logits)
    np.fill_diagonal(exp_logits, 0.0)
    partition = exp_logits.sum(axis=1, keepdims=True)
    probs = exp_logits / np.where(partition == 0.0, 1.0, partition)

    n = batch_features.shape[0]
    positives = np.arange(n) ^ 1
    targets = np.zeros_like(probs)
    targets[np.arange(n), positives] = 1.0

    positive_prob = probs[np.arange(n), positives]
    positive_prob = np.clip(positive_prob, 1e-12, None)
    loss = float(-np.log(positive_prob).mean() + 0.5 * weight_decay * np.sum(weights * weights))

    d_logits = (probs - targets) / n
    d_z = ((d_logits + d_logits.T) @ z) / temperature
    d_y = backward_row_normalize(d_z, y, dy_dz)
    grad = batch_features.T @ d_y + weight_decay * weights
    return loss, grad.astype(np.float32, copy=False)


def normalize_rows_with_backward(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    norms = np.linalg.norm(y, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    z = y / norms
    return z.astype(np.float32, copy=False), norms.astype(np.float32, copy=False)


def backward_row_normalize(d_z: np.ndarray, y: np.ndarray, norms: np.ndarray) -> np.ndarray:
    z = y / norms
    projection = np.sum(d_z * z, axis=1, keepdims=True)
    return (d_z - z * projection) / norms


def row_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return matrix / norms


def write_outputs(
    *,
    model: LinearProjectionModel,
    train_features: np.ndarray,
    train_metadata: pd.DataFrame,
    eval_embeddings: np.ndarray,
    eval_metadata: pd.DataFrame,
    feature_report_train: FeatureExtractionReport,
    feature_report_eval: FeatureExtractionReport,
    training_report: ContrastiveTrainingReport,
    retrieval_report: RetrievalReport,
    run_root: Path,
) -> dict[str, Path]:
    output_root = run_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    model_path = output_root / "model.npz"
    train_features_path = output_root / "train_features.npy"
    eval_embeddings_path = output_root / "eval_embeddings.npy"
    train_index_path = output_root / "train_index.parquet"
    eval_index_path = output_root / "eval_index.parquet"
    feature_train_path = output_root / "feature_train_benchmark.json"
    feature_eval_path = output_root / "feature_eval_benchmark.json"
    training_path = output_root / "training.json"
    retrieval_path = output_root / "retrieval.json"

    np.savez(
        model_path,
        mean=model.mean,
        scale=model.scale,
        weights=model.weights,
        embedding_dim=np.array([model.embedding_dim], dtype=np.int64),
    )
    np.save(train_features_path, train_features)
    np.save(eval_embeddings_path, eval_embeddings)
    train_metadata.to_parquet(train_index_path, index=False, compression="zstd")
    eval_metadata.to_parquet(eval_index_path, index=False, compression="zstd")
    feature_train_path.write_text(
        json.dumps(asdict(feature_report_train), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    feature_eval_path.write_text(
        json.dumps(asdict(feature_report_eval), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    training_path.write_text(
        json.dumps(asdict(training_report), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    retrieval_path.write_text(
        json.dumps(asdict(retrieval_report), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {
        "model": model_path,
        "train_features": train_features_path,
        "eval_embeddings": eval_embeddings_path,
        "train_index": train_index_path,
        "eval_index": eval_index_path,
        "feature_train": feature_train_path,
        "feature_eval": feature_eval_path,
        "training": training_path,
        "retrieval": retrieval_path,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    chips_path = args.chips_path.resolve()
    run_root = args.run_root.resolve()
    gdal_prefix = args.gdal_prefix.resolve() if args.gdal_prefix is not None else None
    train_splits = ordered_unique(args.train_split)
    query_splits = ordered_unique(args.query_split)
    gallery_splits = ordered_unique(args.gallery_split)
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

    feature_embedder = SimplePanEmbedder()
    train_features, train_metadata, train_feature_report = extract_feature_dataset(
        train_dataset,
        embedder=feature_embedder,
    )
    eval_features, eval_metadata, eval_feature_report = extract_feature_dataset(
        eval_dataset,
        embedder=feature_embedder,
    )

    model, training_report = train_contrastive_projection(
        train_features,
        train_metadata,
        positive_key=args.positive_key,
        embedding_dim=args.embedding_dim,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        pairs_per_batch=args.pairs_per_batch,
        learning_rate=args.learning_rate,
        temperature=args.temperature,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )
    eval_embeddings = model.embed(eval_features)
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
        train_features=train_features,
        train_metadata=train_metadata,
        eval_embeddings=eval_embeddings,
        eval_metadata=eval_metadata,
        feature_report_train=train_feature_report,
        feature_report_eval=eval_feature_report,
        training_report=training_report,
        retrieval_report=retrieval_report,
        run_root=run_root,
    )

    print(f"chips: {chips_path}")
    print(f"train_samples: {len(train_features)} eval_samples: {len(eval_features)}")
    print(
        f"protocol: query_splits={query_splits} gallery_splits={gallery_splits} "
        f"queries={retrieval_report.query_count} gallery={retrieval_report.gallery_count}"
    )
    print(
        f"train_features: dim={train_feature_report.feature_dim} "
        f"samples/s={train_feature_report.samples_per_second:.2f} "
        f"read_p95={train_feature_report.read_latency_ms_p95:.2f} "
        f"feature_p95={train_feature_report.feature_latency_ms_p95:.2f}"
    )
    print(
        f"training: emb_dim={model.embedding_dim} "
        f"loss={training_report.loss_initial:.4f}->{training_report.loss_final:.4f} "
        f"pairs/s={training_report.pairs_per_second:.2f}"
    )
    print(
        f"retrieval: R@1={retrieval_report.recall_at_1:.3f} "
        f"R@5={retrieval_report.recall_at_5:.3f} "
        f"R@10={retrieval_report.recall_at_10:.3f} "
        f"MRR={retrieval_report.mean_reciprocal_rank:.3f}"
    )
    print(f"model: {paths['model']}")
    print(f"feature_train: {paths['feature_train']}")
    print(f"feature_eval: {paths['feature_eval']}")
    print(f"training_report: {paths['training']}")
    print(f"retrieval: {paths['retrieval']}")
    return 0


def ordered_unique(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


if __name__ == "__main__":
    raise SystemExit(main())
