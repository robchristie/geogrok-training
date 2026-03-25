from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Literal

import numpy as np
import pandas as pd

from geogrok.data.training import TrainingChipDataset

DEFAULT_RUN_ROOT = Path("artifacts/runs/embedding-baseline")


@dataclass(frozen=True)
class EmbeddingTiming:
    read_ms: float
    transform_ms: float
    embed_ms: float
    total_ms: float


@dataclass(frozen=True)
class EmbeddingBenchmarkReport:
    samples: int
    embedding_dim: int
    elapsed_seconds: float
    samples_per_second: float
    read_latency_ms_mean: float
    read_latency_ms_p95: float
    transform_latency_ms_mean: float
    transform_latency_ms_p95: float
    embed_latency_ms_mean: float
    embed_latency_ms_p95: float
    total_latency_ms_mean: float
    total_latency_ms_p95: float


@dataclass(frozen=True)
class RetrievalReport:
    positive_key: str
    queries_evaluated: int
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    mean_reciprocal_rank: float


class SimplePanEmbedder:
    def __init__(
        self,
        *,
        intensity_bins: int = 16,
        coarse_grid: int = 8,
        profile_bins: int = 16,
    ) -> None:
        if intensity_bins <= 0 or coarse_grid <= 0 or profile_bins <= 0:
            raise ValueError("Embedder dimensions must be positive.")
        self.intensity_bins = intensity_bins
        self.coarse_grid = coarse_grid
        self.profile_bins = profile_bins

    @property
    def embedding_dim(self) -> int:
        coarse = self.coarse_grid * self.coarse_grid
        return coarse + coarse + self.intensity_bins + self.profile_bins + self.profile_bins

    def embed(self, image: np.ndarray) -> np.ndarray:
        if image.ndim != 3:
            raise ValueError(f"Expected image in (C, H, W), got shape {image.shape}.")

        pan = np.asarray(image[0], dtype=np.float32)
        coarse = pooled_grid(pan, self.coarse_grid, self.coarse_grid).reshape(-1)

        grad_y, grad_x = np.gradient(pan)
        grad_mag = np.hypot(grad_x, grad_y)
        coarse_grad = pooled_grid(grad_mag, self.coarse_grid, self.coarse_grid).reshape(-1)

        intensity_max = 1.0 if pan.max() <= 1.0 else float(pan.max())
        histogram = np.histogram(
            pan,
            bins=self.intensity_bins,
            range=(0.0, intensity_max),
            density=True,
        )[0].astype(np.float32)

        row_profile = reduce_profile(pan.mean(axis=1), self.profile_bins)
        col_profile = reduce_profile(pan.mean(axis=0), self.profile_bins)

        embedding = np.concatenate(
            (coarse, coarse_grad, histogram, row_profile, col_profile), dtype=np.float32
        )
        return l2_normalize(embedding)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a deterministic embedding baseline over on-demand PAN chips."
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
        "--split",
        action="append",
        default=["train"],
        help="Split to include. Repeat to add more splits.",
    )
    parser.add_argument(
        "--modality",
        action="append",
        default=["PAN"],
        help="Modality to include. Repeat to add more modalities.",
    )
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--output-dtype", default="float32")
    parser.add_argument("--clip-min", type=float, default=0.0)
    parser.add_argument("--clip-max", type=float, default=2047.0)
    parser.add_argument("--scale-max", type=float, default=2047.0)
    parser.add_argument(
        "--positive-key",
        choices=("scene_id", "capture_id"),
        default="scene_id",
    )
    return parser.parse_args(argv)


def embed_dataset(
    dataset: TrainingChipDataset,
    *,
    embedder: SimplePanEmbedder,
) -> tuple[np.ndarray, pd.DataFrame, EmbeddingBenchmarkReport]:
    records: list[dict[str, object]] = []
    embeddings: list[np.ndarray] = []
    read_latencies: list[float] = []
    transform_latencies: list[float] = []
    embed_latencies: list[float] = []
    total_latencies: list[float] = []

    start = perf_counter()
    for index in range(len(dataset)):
        sample = dataset.sample(index)
        embed_start = perf_counter()
        embedding = embedder.embed(sample.image)
        embed_ms = (perf_counter() - embed_start) * 1000.0
        total_ms = sample.timing.total_ms + embed_ms

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
            }
        )
        embeddings.append(embedding)
        read_latencies.append(sample.timing.read_ms)
        transform_latencies.append(sample.timing.transform_ms)
        embed_latencies.append(embed_ms)
        total_latencies.append(total_ms)
    elapsed_seconds = perf_counter() - start

    matrix = np.stack(embeddings, axis=0) if embeddings else np.empty((0, embedder.embedding_dim))
    benchmark = EmbeddingBenchmarkReport(
        samples=len(records),
        embedding_dim=embedder.embedding_dim,
        elapsed_seconds=elapsed_seconds,
        samples_per_second=safe_rate(len(records), elapsed_seconds),
        read_latency_ms_mean=mean(read_latencies),
        read_latency_ms_p95=percentile(read_latencies, 95.0),
        transform_latency_ms_mean=mean(transform_latencies),
        transform_latency_ms_p95=percentile(transform_latencies, 95.0),
        embed_latency_ms_mean=mean(embed_latencies),
        embed_latency_ms_p95=percentile(embed_latencies, 95.0),
        total_latency_ms_mean=mean(total_latencies),
        total_latency_ms_p95=percentile(total_latencies, 95.0),
    )
    return matrix, pd.DataFrame(records), benchmark


def evaluate_retrieval(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    *,
    positive_key: Literal["scene_id", "capture_id"],
) -> RetrievalReport:
    if len(embeddings) != len(metadata):
        raise ValueError("Embedding matrix and metadata row count must match.")
    if len(embeddings) == 0:
        return RetrievalReport(
            positive_key=positive_key,
            queries_evaluated=0,
            recall_at_1=0.0,
            recall_at_5=0.0,
            recall_at_10=0.0,
            mean_reciprocal_rank=0.0,
        )

    matrix = np.asarray(embeddings, dtype=np.float32)
    matrix = row_normalize(matrix)
    similarity = matrix @ matrix.T
    np.fill_diagonal(similarity, -np.inf)

    labels = metadata[positive_key].fillna("null").astype(str).tolist()
    recalls = {1: 0, 5: 0, 10: 0}
    reciprocal_ranks: list[float] = []
    queries_evaluated = 0

    for index, label in enumerate(labels):
        positives = [
            candidate
            for candidate, candidate_label in enumerate(labels)
            if candidate_label == label and candidate != index
        ]
        if not positives:
            continue

        ranking = np.argsort(-similarity[index])
        first_positive_rank = None
        for rank, candidate in enumerate(ranking, start=1):
            if candidate in positives:
                first_positive_rank = rank
                break
        if first_positive_rank is None:
            continue

        queries_evaluated += 1
        reciprocal_ranks.append(1.0 / first_positive_rank)
        for k in recalls:
            if first_positive_rank <= k:
                recalls[k] += 1

    if queries_evaluated == 0:
        return RetrievalReport(
            positive_key=positive_key,
            queries_evaluated=0,
            recall_at_1=0.0,
            recall_at_5=0.0,
            recall_at_10=0.0,
            mean_reciprocal_rank=0.0,
        )

    return RetrievalReport(
        positive_key=positive_key,
        queries_evaluated=queries_evaluated,
        recall_at_1=recalls[1] / queries_evaluated,
        recall_at_5=recalls[5] / queries_evaluated,
        recall_at_10=recalls[10] / queries_evaluated,
        mean_reciprocal_rank=sum(reciprocal_ranks) / len(reciprocal_ranks),
    )


def pooled_grid(image: np.ndarray, rows: int, cols: int) -> np.ndarray:
    y_edges = np.linspace(0, image.shape[0], rows + 1, dtype=int)
    x_edges = np.linspace(0, image.shape[1], cols + 1, dtype=int)
    pooled = np.zeros((rows, cols), dtype=np.float32)

    for row in range(rows):
        for col in range(cols):
            window = image[y_edges[row] : y_edges[row + 1], x_edges[col] : x_edges[col + 1]]
            pooled[row, col] = float(window.mean()) if window.size else 0.0
    return pooled


def reduce_profile(profile: np.ndarray, bins: int) -> np.ndarray:
    edges = np.linspace(0, profile.shape[0], bins + 1, dtype=int)
    reduced = np.zeros(bins, dtype=np.float32)
    for index in range(bins):
        segment = profile[edges[index] : edges[index + 1]]
        reduced[index] = float(segment.mean()) if segment.size else 0.0
    return reduced


def row_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return matrix / norms


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector
    return vector / norm


def write_outputs(
    *,
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    benchmark: EmbeddingBenchmarkReport,
    retrieval: RetrievalReport,
    run_root: Path,
) -> dict[str, Path]:
    output_root = run_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    embeddings_path = output_root / "embeddings.npy"
    index_path = output_root / "index.parquet"
    benchmark_path = output_root / "benchmark.json"
    retrieval_path = output_root / "retrieval.json"

    np.save(embeddings_path, embeddings)
    metadata.to_parquet(index_path, index=False, compression="zstd")
    benchmark_path.write_text(
        json.dumps(asdict(benchmark), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    retrieval_path.write_text(
        json.dumps(asdict(retrieval), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return {
        "embeddings": embeddings_path,
        "index": index_path,
        "benchmark": benchmark_path,
        "retrieval": retrieval_path,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    chips_path = args.chips_path.resolve()
    run_root = args.run_root.resolve()
    gdal_prefix = args.gdal_prefix.resolve() if args.gdal_prefix is not None else None

    dataset = TrainingChipDataset.from_manifest(
        chips_path,
        splits=tuple(args.split),
        modalities=tuple(args.modality),
        limit=args.limit,
        gdal_prefix=gdal_prefix,
        output_dtype=args.output_dtype,
        clip_min=args.clip_min,
        clip_max=args.clip_max,
        scale_max=args.scale_max,
    )
    if len(dataset) == 0:
        raise SystemExit("No local chip rows matched the requested filters.")

    embedder = SimplePanEmbedder()
    embeddings, metadata, benchmark = embed_dataset(dataset, embedder=embedder)
    retrieval = evaluate_retrieval(
        embeddings,
        metadata,
        positive_key=args.positive_key,
    )
    paths = write_outputs(
        embeddings=embeddings,
        metadata=metadata,
        benchmark=benchmark,
        retrieval=retrieval,
        run_root=run_root,
    )

    print(f"chips: {chips_path}")
    print(f"samples: {benchmark.samples}")
    print(f"embedding_dim: {benchmark.embedding_dim}")
    print(f"samples/s: {benchmark.samples_per_second:.2f}")
    print(
        f"latency_ms: read_p95={benchmark.read_latency_ms_p95:.2f} "
        f"transform_p95={benchmark.transform_latency_ms_p95:.2f} "
        f"embed_p95={benchmark.embed_latency_ms_p95:.2f} "
        f"total_p95={benchmark.total_latency_ms_p95:.2f}"
    )
    print(
        f"retrieval: R@1={retrieval.recall_at_1:.3f} "
        f"R@5={retrieval.recall_at_5:.3f} "
        f"R@10={retrieval.recall_at_10:.3f} "
        f"MRR={retrieval.mean_reciprocal_rank:.3f}"
    )
    print(f"embeddings: {paths['embeddings']}")
    print(f"index: {paths['index']}")
    print(f"benchmark: {paths['benchmark']}")
    print(f"retrieval: {paths['retrieval']}")
    return 0


def mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(float(value) for value in values) / len(values)


def percentile(values: Sequence[float], percentile_rank: float) -> float:
    if not values:
        return 0.0
    if percentile_rank < 0.0 or percentile_rank > 100.0:
        raise ValueError("percentile_rank must be in [0, 100].")

    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * (percentile_rank / 100.0)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def safe_rate(value: float, seconds: float) -> float:
    if seconds <= 0.0:
        return 0.0
    return value / seconds


if __name__ == "__main__":
    raise SystemExit(main())
