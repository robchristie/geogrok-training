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

from geogrok.data.runtime import OnDemandChipDataset
from geogrok.data.training import TrainingChipDataset

DEFAULT_RUN_ROOT = Path("artifacts/runs/embedding-baseline")


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
    query_splits: tuple[str, ...]
    gallery_splits: tuple[str, ...]
    query_count: int
    gallery_count: int
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

    def features(self, image: np.ndarray) -> np.ndarray:
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

        return np.concatenate(
            (coarse, coarse_grad, histogram, row_profile, col_profile), dtype=np.float32
        )

    def embed(self, image: np.ndarray) -> np.ndarray:
        return l2_normalize(self.features(image))


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
        help="Default split set used for both query and gallery when explicit sets are omitted.",
    )
    parser.add_argument(
        "--query-split",
        action="append",
        help="Query split to evaluate. Repeat to add more splits.",
    )
    parser.add_argument(
        "--gallery-split",
        action="append",
        help="Gallery split to evaluate. Repeat to add more splits.",
    )
    parser.add_argument(
        "--modality",
        action="append",
        default=["PAN"],
        help="Modality to include. Repeat to add more modalities.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=128,
        help="Maximum total chips after balanced scene sampling.",
    )
    parser.add_argument(
        "--max-chips-per-scene",
        type=int,
        default=4,
        help="Maximum chips to keep per scene before embedding.",
    )
    parser.add_argument(
        "--min-chips-per-scene",
        type=int,
        default=2,
        help="Minimum chips per scene required for retrieval evaluation.",
    )
    parser.add_argument("--output-dtype", default="float32")
    parser.add_argument("--clip-min", type=float, default=0.0)
    parser.add_argument("--clip-max", type=float, default=2047.0)
    parser.add_argument("--scale-max", type=float, default=2047.0)
    parser.add_argument(
        "--positive-key",
        choices=("scene_id", "capture_id"),
        default="scene_id",
    )
    parser.add_argument(
        "--min-positive-center-distance",
        type=float,
        default=1024.0,
        help="Minimum center distance in pixels for a valid positive match.",
    )
    parser.add_argument(
        "--allow-overlap-positives",
        action="store_true",
        help="Allow overlapping chips to count as positives.",
    )
    return parser.parse_args(argv)


def build_dataset(
    chips_path: Path,
    *,
    splits: Sequence[str],
    modalities: Sequence[str],
    limit: int | None,
    min_chips_per_scene: int,
    max_chips_per_scene: int | None,
    gdal_prefix: Path | None,
    output_dtype: str,
    clip_min: float | None,
    clip_max: float | None,
    scale_max: float | None,
) -> TrainingChipDataset:
    base_dataset = TrainingChipDataset.from_manifest(
        chips_path,
        splits=tuple(splits),
        modalities=tuple(modalities),
        limit=None,
        gdal_prefix=gdal_prefix,
        output_dtype=output_dtype,
        clip_min=clip_min,
        clip_max=clip_max,
        scale_max=scale_max,
    )
    selected_frame = balanced_subset(
        base_dataset.records_frame(),
        group_key="scene_id",
        min_per_group=min_chips_per_scene,
        max_per_group=max_chips_per_scene,
        limit=limit,
    )
    selected_runtime = OnDemandChipDataset(selected_frame, gdal_prefix=gdal_prefix)
    return TrainingChipDataset(
        selected_runtime,
        output_dtype=output_dtype,
        clip_min=clip_min,
        clip_max=clip_max,
        scale_max=scale_max,
    )


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
                "x0": sample.record.x0,
                "y0": sample.record.y0,
                "width": sample.record.width,
                "height": sample.record.height,
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
    query_splits: Sequence[str],
    gallery_splits: Sequence[str],
    min_positive_center_distance: float,
    allow_overlap_positives: bool,
) -> RetrievalReport:
    if len(embeddings) != len(metadata):
        raise ValueError("Embedding matrix and metadata row count must match.")
    if len(embeddings) == 0:
        return RetrievalReport(
            positive_key=positive_key,
            query_splits=tuple(query_splits),
            gallery_splits=tuple(gallery_splits),
            query_count=0,
            gallery_count=0,
            queries_evaluated=0,
            recall_at_1=0.0,
            recall_at_5=0.0,
            recall_at_10=0.0,
            mean_reciprocal_rank=0.0,
        )

    frame = metadata.reset_index(drop=True).copy()
    frame["positive_label"] = frame[positive_key].fillna("null").astype(str)
    frame["split_normalized"] = frame["split"].astype(str).str.lower()

    query_tokens = {split.lower() for split in query_splits}
    gallery_tokens = {split.lower() for split in gallery_splits}
    query_indices = [
        int(index)
        for index in frame.index
        if frame.iloc[index]["split_normalized"] in query_tokens
    ]
    gallery_indices = [
        int(index)
        for index in frame.index
        if frame.iloc[index]["split_normalized"] in gallery_tokens
    ]

    matrix = row_normalize(np.asarray(embeddings, dtype=np.float32))
    similarity = matrix @ matrix.T
    labels = frame["positive_label"].tolist()
    recalls = {1: 0, 5: 0, 10: 0}
    reciprocal_ranks: list[float] = []
    queries_evaluated = 0

    for index in query_indices:
        label = labels[index]
        positives = [
            candidate
            for candidate in gallery_indices
            if labels[candidate] == label
            and candidate != index
            and is_valid_positive_pair(
                frame.iloc[index],
                frame.iloc[candidate],
                min_center_distance=min_positive_center_distance,
                allow_overlap=allow_overlap_positives,
            )
        ]
        if not positives:
            continue

        allowed_gallery = [
            candidate
            for candidate in gallery_indices
            if candidate != index
            and (
                labels[candidate] != label
                or is_valid_positive_pair(
                    frame.iloc[index],
                    frame.iloc[candidate],
                    min_center_distance=min_positive_center_distance,
                    allow_overlap=allow_overlap_positives,
                )
            )
        ]
        if not allowed_gallery:
            continue

        ranking = sorted(
            allowed_gallery,
            key=lambda candidate: float(similarity[index, candidate]),
            reverse=True,
        )
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
            query_splits=tuple(query_splits),
            gallery_splits=tuple(gallery_splits),
            query_count=len(query_indices),
            gallery_count=len(gallery_indices),
            queries_evaluated=0,
            recall_at_1=0.0,
            recall_at_5=0.0,
            recall_at_10=0.0,
            mean_reciprocal_rank=0.0,
        )

    return RetrievalReport(
        positive_key=positive_key,
        query_splits=tuple(query_splits),
        gallery_splits=tuple(gallery_splits),
        query_count=len(query_indices),
        gallery_count=len(gallery_indices),
        queries_evaluated=queries_evaluated,
        recall_at_1=recalls[1] / queries_evaluated,
        recall_at_5=recalls[5] / queries_evaluated,
        recall_at_10=recalls[10] / queries_evaluated,
        mean_reciprocal_rank=sum(reciprocal_ranks) / len(reciprocal_ranks),
    )


def balanced_subset(
    frame: pd.DataFrame,
    *,
    group_key: str,
    min_per_group: int,
    max_per_group: int | None,
    limit: int | None,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    if min_per_group <= 0:
        raise ValueError("min_per_group must be positive.")
    if max_per_group is not None and max_per_group <= 0:
        raise ValueError("max_per_group must be positive when provided.")
    if max_per_group is not None and max_per_group < min_per_group:
        raise ValueError("max_per_group cannot be smaller than min_per_group.")
    if limit is not None and limit < min_per_group:
        raise ValueError("limit cannot be smaller than min_per_group.")

    groups: list[pd.DataFrame] = []
    for _, group in frame.groupby(group_key, sort=True, dropna=False):
        ordered = group.sort_values(["split", "city", "chip_id"]).reset_index(drop=True)
        if max_per_group is not None:
            ordered = ordered.head(max_per_group)
        if len(ordered) >= min_per_group:
            groups.append(ordered)

    if limit is not None:
        max_groups = max(1, limit // min_per_group)
        groups = groups[:max_groups]

    selected_rows: list[pd.Series] = []
    position = 0
    while True:
        progress = False
        for group in groups:
            if position < len(group):
                selected_rows.append(group.iloc[position])
                progress = True
                if limit is not None and len(selected_rows) >= limit:
                    return pd.DataFrame(selected_rows).reset_index(drop=True)
        if not progress:
            break
        position += 1
    return pd.DataFrame(selected_rows).reset_index(drop=True)


def is_valid_positive_pair(
    query_row: pd.Series,
    candidate_row: pd.Series,
    *,
    min_center_distance: float,
    allow_overlap: bool,
) -> bool:
    if not allow_overlap and windows_overlap(query_row, candidate_row):
        return False
    if center_distance(query_row, candidate_row) < min_center_distance:
        return False
    return True


def windows_overlap(left: pd.Series, right: pd.Series) -> bool:
    left_x0 = int(left["x0"])
    left_y0 = int(left["y0"])
    left_x1 = left_x0 + int(left["width"])
    left_y1 = left_y0 + int(left["height"])

    right_x0 = int(right["x0"])
    right_y0 = int(right["y0"])
    right_x1 = right_x0 + int(right["width"])
    right_y1 = right_y0 + int(right["height"])

    return (
        left_x0 < right_x1
        and right_x0 < left_x1
        and left_y0 < right_y1
        and right_y0 < left_y1
    )


def center_distance(left: pd.Series, right: pd.Series) -> float:
    left_cx = float(left["x0"]) + float(left["width"]) / 2.0
    left_cy = float(left["y0"]) + float(left["height"]) / 2.0
    right_cx = float(right["x0"]) + float(right["width"]) / 2.0
    right_cy = float(right["y0"]) + float(right["height"]) / 2.0
    return float(np.hypot(left_cx - right_cx, left_cy - right_cy))


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
    query_splits = tuple(args.query_split) if args.query_split else tuple(args.split)
    gallery_splits = tuple(args.gallery_split) if args.gallery_split else tuple(args.split)
    selected_splits = tuple(sorted(set(query_splits).union(gallery_splits)))

    dataset = build_dataset(
        chips_path,
        splits=selected_splits,
        modalities=tuple(args.modality),
        limit=args.limit,
        min_chips_per_scene=args.min_chips_per_scene,
        max_chips_per_scene=args.max_chips_per_scene,
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
        query_splits=query_splits,
        gallery_splits=gallery_splits,
        min_positive_center_distance=args.min_positive_center_distance,
        allow_overlap_positives=args.allow_overlap_positives,
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
    print(
        f"protocol: query_splits={query_splits} gallery_splits={gallery_splits} "
        f"queries={retrieval.query_count} gallery={retrieval.gallery_count}"
    )
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
