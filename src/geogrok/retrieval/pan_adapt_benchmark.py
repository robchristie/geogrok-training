from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from geogrok.data.runtime import OnDemandChipDataset
from geogrok.data.training import TrainingChipDataset
from geogrok.io.raster import load_gdal
from geogrok.retrieval.baseline import mean, percentile, safe_rate
from geogrok.retrieval.cnn import PreprocessReport, extract_images
from geogrok.retrieval.pair_eval import (
    PairRetrievalReport,
    chip_ids_from_pairs,
    evaluate_pair_retrieval,
)
from geogrok.retrieval.pretrained_benchmark import (
    PretrainedBenchmarkReport,
    build_eval_dataset,
    load_pretrained_model,
    normalize_multi_arg,
    require_torchvision,
    resolve_device,
)
from geogrok.retrieval.pretrained_benchmark import (
    embed_dataset as embed_teacher_dataset,
)
from geogrok.retrieval.torch_encoder import (
    TorchEmbeddingReport,
    create_model,
    embed_images_torch,
    filter_pairs_for_records,
    nt_xent_loss,
    require_torch,
)

DEFAULT_RUN_ROOT = Path("artifacts/runs/pan-adapt-benchmark")


@dataclass(frozen=True)
class DistillationTrainingReport:
    teacher_model: str
    teacher_weights: str
    student_arch: str
    train_samples: int
    train_positive_pairs: int
    train_positive_exact_pairs: int
    train_positive_weak_pairs: int
    train_hard_negative_pairs: int
    hard_negative_teacher_similarity_mean: float
    hard_negative_teacher_similarity_p95: float
    student_image_size: int
    student_base_channels: int
    student_embedding_dim: int
    contrastive_weight: float
    alignment_weight: float
    structure_weight: float
    view_consistency_weight: float
    positive_pair_weight: float
    hard_negative_weight: float
    positive_exact_weight: float
    positive_weak_weight: float
    hard_negative_max_similarity: float
    hard_negative_gap_scale: float
    hard_negative_min_similarity: float
    adversarial_negative_top_fraction: float
    adversarial_negative_max_pairs: int
    adversarial_negative_min_teacher_similarity: float
    augmentation_min_crop_scale: float
    augmentation_noise_std: float
    augmentation_gamma_jitter: float
    augmentation_blur_probability: float
    device: str
    device_name: str
    amp_enabled: bool
    epochs: int
    steps_per_epoch: int
    pairs_per_batch: int
    batch_size: int
    elapsed_seconds: float
    steps_per_second: float
    images_per_second: float
    batch_latency_ms_mean: float
    batch_latency_ms_p95: float
    loss_initial: float
    loss_final: float
    loss_best: float
    contrastive_loss_mean: float
    alignment_loss_mean: float
    structure_loss_mean: float
    view_consistency_loss_mean: float
    positive_pair_loss_mean: float
    hard_negative_loss_mean: float
    max_memory_mib: float


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark PAN-aware student adaptation against a frozen "
            "teacher embedding space."
        )
    )
    parser.add_argument(
        "--chips-path",
        type=Path,
        default=Path("datasets/manifests/spacenet/chips.parquet"),
    )
    parser.add_argument(
        "--pairs-path",
        type=Path,
        required=True,
        help="Explicit pairs.parquet used for training and held-out retrieval evaluation.",
    )
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument(
        "--gdal-prefix",
        type=Path,
        help="Override the local GDAL/Kakadu install prefix.",
    )
    parser.add_argument(
        "--teacher-model",
        default="resnet152",
        help="Frozen teacher model name from geogrok-benchmark-pretrained.",
    )
    parser.add_argument(
        "--train-split",
        action="append",
        default=None,
        help="Split used for student distillation training. Repeat to add more splits.",
    )
    parser.add_argument(
        "--query-split",
        action="append",
        default=None,
        help="Query split for held-out retrieval evaluation. Repeat to add more splits.",
    )
    parser.add_argument(
        "--gallery-split",
        action="append",
        default=None,
        help="Gallery split for held-out retrieval evaluation. Repeat to add more splits.",
    )
    parser.add_argument(
        "--modality",
        action="append",
        default=None,
        help="Modality to include. Repeat to add more modalities.",
    )
    parser.add_argument("--train-limit", type=int, default=512)
    parser.add_argument("--eval-limit", type=int, default=512)
    parser.add_argument("--output-dtype", default="float32")
    parser.add_argument("--clip-min", type=float, default=0.0)
    parser.add_argument("--clip-max", type=float, default=2047.0)
    parser.add_argument("--scale-max", type=float, default=2047.0)
    parser.add_argument("--teacher-batch-size", type=int, default=16)
    parser.add_argument("--student-image-size", type=int, default=128)
    parser.add_argument(
        "--student-arch",
        choices=("baseline_cnn", "residual_cnn"),
        default="residual_cnn",
    )
    parser.add_argument("--student-base-channels", type=int, default=48)
    parser.add_argument("--student-dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--steps-per-epoch", type=int, default=48)
    parser.add_argument("--pairs-per-batch", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--contrastive-weight", type=float, default=1.0)
    parser.add_argument("--alignment-weight", type=float, default=1.0)
    parser.add_argument("--structure-weight", type=float, default=0.5)
    parser.add_argument("--view-consistency-weight", type=float, default=0.25)
    parser.add_argument("--positive-pair-weight", type=float, default=0.5)
    parser.add_argument("--hard-negative-weight", type=float, default=0.25)
    parser.add_argument("--positive-exact-weight", type=float, default=2.0)
    parser.add_argument("--positive-weak-weight", type=float, default=1.0)
    parser.add_argument("--hard-negative-max-similarity", type=float, default=0.2)
    parser.add_argument("--hard-negative-gap-scale", type=float, default=0.5)
    parser.add_argument("--hard-negative-min-similarity", type=float, default=-0.25)
    parser.add_argument("--adversarial-negative-top-fraction", type=float, default=0.25)
    parser.add_argument("--adversarial-negative-max-pairs", type=int, default=512)
    parser.add_argument("--adversarial-negative-min-teacher-similarity", type=float, default=0.0)
    parser.add_argument("--augmentation-min-crop-scale", type=float, default=0.7)
    parser.add_argument("--augmentation-noise-std", type=float, default=0.02)
    parser.add_argument("--augmentation-gamma-jitter", type=float, default=0.15)
    parser.add_argument("--augmentation-blur-probability", type=float, default=0.2)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Training and inference device.",
    )
    parser.add_argument("--amp", action="store_true", help="Use mixed precision on CUDA.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def assert_record_alignment(
    *,
    expected_records: Sequence[dict[str, object]],
    actual_metadata: pd.DataFrame,
    field_name: str,
) -> None:
    expected = [str(record["chip_id"]) for record in expected_records]
    actual = actual_metadata["chip_id"].astype(str).tolist()
    if expected != actual:
        raise RuntimeError(f"{field_name} chip ordering does not match the student dataset.")


def create_student_model(
    torch: Any,
    *,
    arch: str,
    input_channels: int,
    base_channels: int,
    embedding_dim: int,
    dropout: float,
) -> Any:
    nn = torch.nn

    if arch == "baseline_cnn":
        return create_model(
            torch,
            input_channels=input_channels,
            base_channels=base_channels,
            embedding_dim=embedding_dim,
            dropout=dropout,
        )

    if arch != "residual_cnn":
        raise ValueError(f"Unsupported student arch: {arch}")

    class ResidualBlock(nn.Module):
        def __init__(self, channels_in: int, channels_out: int, *, stride: int) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(
                channels_in,
                channels_out,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            )
            self.bn1 = nn.BatchNorm2d(channels_out)
            self.conv2 = nn.Conv2d(
                channels_out,
                channels_out,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
            self.bn2 = nn.BatchNorm2d(channels_out)
            self.relu = nn.ReLU(inplace=True)
            if stride != 1 or channels_in != channels_out:
                self.skip = nn.Sequential(
                    nn.Conv2d(channels_in, channels_out, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(channels_out),
                )
            else:
                self.skip = nn.Identity()

        def forward(self, x: Any) -> Any:
            identity = self.skip(x)
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out = self.relu(out + identity)
            return out

    class ResidualPanEmbeddingNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(
                    input_channels,
                    base_channels,
                    kernel_size=7,
                    stride=2,
                    padding=3,
                    bias=False,
                ),
                nn.BatchNorm2d(base_channels),
                nn.ReLU(inplace=True),
            )
            self.layer1 = nn.Sequential(
                ResidualBlock(base_channels, base_channels, stride=1),
                ResidualBlock(base_channels, base_channels, stride=1),
            )
            self.layer2 = nn.Sequential(
                ResidualBlock(base_channels, base_channels * 2, stride=2),
                ResidualBlock(base_channels * 2, base_channels * 2, stride=1),
            )
            self.layer3 = nn.Sequential(
                ResidualBlock(base_channels * 2, base_channels * 4, stride=2),
                ResidualBlock(base_channels * 4, base_channels * 4, stride=1),
            )
            self.layer4 = nn.Sequential(
                ResidualBlock(base_channels * 4, base_channels * 4, stride=2),
                ResidualBlock(base_channels * 4, base_channels * 4, stride=1),
            )
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.projection = nn.Sequential(
                nn.Flatten(),
                nn.Linear(base_channels * 4, base_channels * 4),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(base_channels * 4, embedding_dim),
            )

        def forward(self, x: Any) -> Any:
            x = self.stem(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.pool(x)
            x = self.projection(x)
            return torch.nn.functional.normalize(x, dim=1)

    return ResidualPanEmbeddingNet()


def augment_pan_batch(
    torch: Any,
    images: Any,
    *,
    rng: np.random.Generator,
    min_crop_scale: float,
    noise_std: float,
    gamma_jitter: float,
    blur_probability: float,
) -> Any:
    functional = torch.nn.functional
    batch, channels, height, width = images.shape
    augmented: list[Any] = []

    for index in range(batch):
        image = images[index : index + 1]
        crop_scale = float(rng.uniform(min_crop_scale, 1.0))
        crop_height = max(16, min(height, int(round(height * crop_scale))))
        crop_width = max(16, min(width, int(round(width * crop_scale))))
        top = 0 if crop_height >= height else int(rng.integers(0, height - crop_height + 1))
        left = 0 if crop_width >= width else int(rng.integers(0, width - crop_width + 1))
        image = image[:, :, top : top + crop_height, left : left + crop_width]
        if crop_height != height or crop_width != width:
            image = functional.interpolate(
                image,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )

        if bool(rng.integers(0, 2)):
            image = torch.flip(image, dims=(3,))
        if bool(rng.integers(0, 2)):
            image = torch.flip(image, dims=(2,))

        if gamma_jitter > 0.0:
            gamma = float(rng.uniform(1.0 - gamma_jitter, 1.0 + gamma_jitter))
            gamma = max(0.5, gamma)
            image = torch.pow(torch.clamp(image, min=0.0, max=1.0), gamma)

        if noise_std > 0.0:
            noise = torch.randn(
                (1, channels, height, width),
                device=image.device,
                dtype=image.dtype,
            )
            image = image + noise * noise_std

        if rng.random() < blur_probability:
            image = functional.avg_pool2d(image, kernel_size=3, stride=1, padding=1)

        augmented.append(torch.clamp(image, min=0.0, max=1.0))

    return torch.cat(augmented, dim=0)


def teacher_structure_loss(torch: Any, student_embeddings: Any, teacher_embeddings: Any) -> Any:
    functional = torch.nn.functional
    student_similarity = student_embeddings @ student_embeddings.T
    teacher_similarity = teacher_embeddings @ teacher_embeddings.T
    off_diagonal_mask = ~torch.eye(
        student_similarity.shape[0],
        dtype=torch.bool,
        device=student_similarity.device,
    )
    return functional.mse_loss(
        student_similarity[off_diagonal_mask],
        teacher_similarity[off_diagonal_mask],
    )


def build_pan_adapt_training_dataset(
    chips_path: Path,
    *,
    splits: Sequence[str],
    modalities: Sequence[str],
    pair_frame: pd.DataFrame,
    limit: int | None,
    gdal_prefix: Path | None,
    output_dtype: str,
    clip_min: float,
    clip_max: float,
    scale_max: float,
) -> tuple[TrainingChipDataset, pd.DataFrame]:
    dataset = TrainingChipDataset.from_manifest(
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
    records_frame = dataset.records_frame().copy()
    positive_pairs = filter_pairs_for_records(
        pair_frame,
        records_frame,
        pair_labels={"positive_exact", "positive_weak"},
    )
    if positive_pairs.empty:
        raise SystemExit("No positive training pairs overlap with the requested training splits.")
    negative_pairs = filter_pairs_for_records(
        pair_frame,
        records_frame,
        pair_labels={"negative_hard"},
    )

    train_chip_ids = chip_ids_from_pairs(positive_pairs)
    if not negative_pairs.empty:
        train_chip_ids = train_chip_ids.union(chip_ids_from_pairs(negative_pairs))
    records_frame = (
        records_frame[records_frame["chip_id"].astype(str).isin(train_chip_ids)]
        .sort_values(["asset_id", "chip_id"])
        .reset_index(drop=True)
    )
    if limit is not None and len(records_frame) > limit:
        limited_frame = records_frame.head(limit).reset_index(drop=True)
        positive_pairs = filter_pairs_for_records(
            positive_pairs,
            limited_frame,
            pair_labels={"positive_exact", "positive_weak"},
        )
        if positive_pairs.empty:
            raise SystemExit(
                "Training chip limit removed all positive pairs. "
                "Increase --train-limit or disable it."
            )
        negative_pairs = filter_pairs_for_records(
            negative_pairs,
            limited_frame,
            pair_labels={"negative_hard"},
        )
        limited_chip_ids = chip_ids_from_pairs(positive_pairs)
        if not negative_pairs.empty:
            limited_chip_ids = limited_chip_ids.union(chip_ids_from_pairs(negative_pairs))
        records_frame = (
            limited_frame[limited_frame["chip_id"].astype(str).isin(limited_chip_ids)]
            .reset_index(drop=True)
        )

    dataset = TrainingChipDataset(
        OnDemandChipDataset(records_frame, gdal_prefix=gdal_prefix),
        output_dtype=output_dtype,
        clip_min=clip_min,
        clip_max=clip_max,
        scale_max=scale_max,
    )
    train_pairs = filter_pairs_for_records(
        pair_frame,
        dataset.records_frame(),
        pair_labels={"positive_exact", "positive_weak", "negative_hard"},
    )
    filtered_positives = filter_pairs_for_records(
        train_pairs,
        dataset.records_frame(),
        pair_labels={"positive_exact", "positive_weak"},
    )
    if filtered_positives.empty:
        raise SystemExit("No positive training pairs remain after dataset filtering.")
    return dataset, train_pairs


def build_weighted_positive_pairs(
    metadata_records: Sequence[dict[str, object]],
    pair_frame: pd.DataFrame,
    *,
    exact_weight: float,
    weak_weight: float,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    metadata = pd.DataFrame(metadata_records).reset_index(drop=True).copy()
    if metadata.empty or pair_frame.empty:
        empty_pairs = np.empty((0, 2), dtype=np.int64)
        empty_weights = np.empty((0,), dtype=np.float32)
        return empty_pairs, empty_weights, 0, 0

    metadata["chip_id"] = metadata["chip_id"].astype(str)
    chip_index = {chip_id: int(index) for index, chip_id in enumerate(metadata["chip_id"])}
    positives = pair_frame[
        pair_frame["pair_label"].isin(["positive_exact", "positive_weak"])
    ].copy()
    if positives.empty:
        empty_pairs = np.empty((0, 2), dtype=np.int64)
        empty_weights = np.empty((0,), dtype=np.float32)
        return empty_pairs, empty_weights, 0, 0

    positives["query_chip_id"] = positives["query_chip_id"].astype(str)
    positives["candidate_chip_id"] = positives["candidate_chip_id"].astype(str)
    positives["query_index"] = positives["query_chip_id"].map(chip_index)
    positives["candidate_index"] = positives["candidate_chip_id"].map(chip_index)
    positives = positives.dropna(subset=["query_index", "candidate_index"]).copy()
    positives["query_index"] = positives["query_index"].astype(int)
    positives["candidate_index"] = positives["candidate_index"].astype(int)
    positives = positives[positives["query_index"] != positives["candidate_index"]].copy()
    if positives.empty:
        empty_pairs = np.empty((0, 2), dtype=np.int64)
        empty_weights = np.empty((0,), dtype=np.float32)
        return empty_pairs, empty_weights, 0, 0

    positives["pair_priority"] = np.where(positives["pair_label"] == "positive_exact", 0, 1)
    positives["left_index"] = positives[["query_index", "candidate_index"]].min(axis=1)
    positives["right_index"] = positives[["query_index", "candidate_index"]].max(axis=1)
    positives = (
        positives.sort_values(["pair_priority", "left_index", "right_index"])
        .drop_duplicates(subset=["left_index", "right_index"], keep="first")
        .reset_index(drop=True)
    )
    positives["sample_weight"] = np.where(
        positives["pair_label"] == "positive_exact",
        exact_weight,
        weak_weight,
    ).astype(np.float32)

    pair_indices = positives[["left_index", "right_index"]].to_numpy(dtype=np.int64)
    pair_weights = positives["sample_weight"].to_numpy(dtype=np.float32)
    exact_pairs = int((positives["pair_label"] == "positive_exact").sum())
    weak_pairs = int((positives["pair_label"] == "positive_weak").sum())
    return pair_indices, pair_weights, exact_pairs, weak_pairs


def build_negative_pairs(
    metadata_records: Sequence[dict[str, object]],
    pair_frame: pd.DataFrame,
) -> np.ndarray:
    metadata = pd.DataFrame(metadata_records).reset_index(drop=True).copy()
    if metadata.empty or pair_frame.empty:
        return np.empty((0, 2), dtype=np.int64)

    metadata["chip_id"] = metadata["chip_id"].astype(str)
    chip_index = {chip_id: int(index) for index, chip_id in enumerate(metadata["chip_id"])}
    negatives = pair_frame[pair_frame["pair_label"] == "negative_hard"].copy()
    if negatives.empty:
        return np.empty((0, 2), dtype=np.int64)

    negatives["query_chip_id"] = negatives["query_chip_id"].astype(str)
    negatives["candidate_chip_id"] = negatives["candidate_chip_id"].astype(str)
    negatives["query_index"] = negatives["query_chip_id"].map(chip_index)
    negatives["candidate_index"] = negatives["candidate_chip_id"].map(chip_index)
    negatives = negatives.dropna(subset=["query_index", "candidate_index"]).copy()
    negatives["query_index"] = negatives["query_index"].astype(int)
    negatives["candidate_index"] = negatives["candidate_index"].astype(int)
    negatives = negatives[negatives["query_index"] != negatives["candidate_index"]].copy()
    if negatives.empty:
        return np.empty((0, 2), dtype=np.int64)

    negatives["left_index"] = negatives[["query_index", "candidate_index"]].min(axis=1)
    negatives["right_index"] = negatives[["query_index", "candidate_index"]].max(axis=1)
    negatives = negatives.drop_duplicates(
        subset=["left_index", "right_index"]
    ).reset_index(drop=True)
    return negatives[["left_index", "right_index"]].to_numpy(dtype=np.int64)


def mine_adversarial_negative_pairs(
    metadata_records: Sequence[dict[str, object]],
    pair_frame: pd.DataFrame,
    teacher_embeddings: np.ndarray,
    *,
    top_fraction: float,
    max_pairs: int,
    min_teacher_similarity: float,
) -> tuple[np.ndarray, float, float]:
    metadata = pd.DataFrame(metadata_records).reset_index(drop=True).copy()
    if metadata.empty or pair_frame.empty:
        return np.empty((0, 2), dtype=np.int64), 0.0, 0.0
    if not 0.0 < top_fraction <= 1.0:
        raise ValueError("top_fraction must be within (0, 1].")
    if max_pairs <= 0:
        raise ValueError("max_pairs must be positive.")
    if len(teacher_embeddings) != len(metadata):
        raise ValueError("teacher_embeddings row count must match metadata_records.")

    embeddings = np.asarray(teacher_embeddings, dtype=np.float32)
    embeddings = embeddings / np.clip(
        np.linalg.norm(embeddings, axis=1, keepdims=True),
        1e-12,
        None,
    )

    metadata["chip_id"] = metadata["chip_id"].astype(str)
    chip_index = {chip_id: int(index) for index, chip_id in enumerate(metadata["chip_id"])}
    negatives = pair_frame[pair_frame["pair_label"] == "negative_hard"].copy()
    if negatives.empty:
        return np.empty((0, 2), dtype=np.int64), 0.0, 0.0

    negatives["query_chip_id"] = negatives["query_chip_id"].astype(str)
    negatives["candidate_chip_id"] = negatives["candidate_chip_id"].astype(str)
    negatives["query_index"] = negatives["query_chip_id"].map(chip_index)
    negatives["candidate_index"] = negatives["candidate_chip_id"].map(chip_index)
    negatives = negatives.dropna(subset=["query_index", "candidate_index"]).copy()
    negatives["query_index"] = negatives["query_index"].astype(int)
    negatives["candidate_index"] = negatives["candidate_index"].astype(int)
    negatives = negatives[negatives["query_index"] != negatives["candidate_index"]].copy()
    if negatives.empty:
        return np.empty((0, 2), dtype=np.int64), 0.0, 0.0

    negatives["left_index"] = negatives[["query_index", "candidate_index"]].min(axis=1)
    negatives["right_index"] = negatives[["query_index", "candidate_index"]].max(axis=1)
    negatives = negatives.drop_duplicates(
        subset=["left_index", "right_index"]
    ).reset_index(drop=True)
    if negatives.empty:
        return np.empty((0, 2), dtype=np.int64), 0.0, 0.0

    left_embeddings = embeddings[negatives["left_index"].to_numpy(dtype=np.int64)]
    right_embeddings = embeddings[negatives["right_index"].to_numpy(dtype=np.int64)]
    negatives["teacher_similarity"] = np.sum(left_embeddings * right_embeddings, axis=1)
    negatives = negatives[negatives["teacher_similarity"] >= min_teacher_similarity].copy()
    if negatives.empty:
        return np.empty((0, 2), dtype=np.int64), 0.0, 0.0

    negatives = negatives.sort_values("teacher_similarity", ascending=False).reset_index(drop=True)
    target_count = max(1, int(np.ceil(len(negatives) * top_fraction)))
    selected = negatives.head(min(max_pairs, target_count)).copy()
    similarities = selected["teacher_similarity"].to_numpy(dtype=np.float32)
    pair_indices = selected[["left_index", "right_index"]].to_numpy(dtype=np.int64)
    return (
        pair_indices,
        float(similarities.mean()) if len(similarities) > 0 else 0.0,
        percentile(similarities.tolist(), 95.0),
    )


def summarize_pair_teacher_similarities(
    pair_indices: np.ndarray,
    teacher_embeddings: np.ndarray,
) -> tuple[float, float]:
    if len(pair_indices) == 0:
        return 0.0, 0.0
    embeddings = np.asarray(teacher_embeddings, dtype=np.float32)
    embeddings = embeddings / np.clip(
        np.linalg.norm(embeddings, axis=1, keepdims=True),
        1e-12,
        None,
    )
    left = embeddings[pair_indices[:, 0]]
    right = embeddings[pair_indices[:, 1]]
    similarities = np.sum(left * right, axis=1)
    return float(similarities.mean()), percentile(similarities.tolist(), 95.0)


def sample_weighted_pairs(
    pairs: np.ndarray,
    *,
    pair_weights: np.ndarray | None,
    pairs_per_batch: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("pairs must have shape (n, 2).")
    probabilities = None
    if pair_weights is not None:
        weights = np.asarray(pair_weights, dtype=np.float64)
        if len(weights) != len(pairs):
            raise ValueError("pair_weights must have the same length as pairs.")
        total_weight = float(weights.sum())
        if total_weight <= 0.0:
            raise ValueError("pair_weights must sum to a positive value.")
        probabilities = weights / total_weight
    selected_pairs = rng.choice(
        len(pairs),
        size=pairs_per_batch,
        replace=len(pairs) < pairs_per_batch,
        p=probabilities,
    )
    return pairs[selected_pairs], selected_pairs.astype(np.int64, copy=False)


def train_student_with_teacher(
    student_images: np.ndarray,
    teacher_targets: np.ndarray,
    positive_pairs: np.ndarray,
    positive_pair_weights: np.ndarray,
    negative_pairs: np.ndarray,
    *,
    positive_exact_pairs: int,
    positive_weak_pairs: int,
    hard_negative_teacher_similarity_mean: float,
    hard_negative_teacher_similarity_p95: float,
    student_arch: str,
    base_channels: int,
    dropout: float,
    epochs: int,
    steps_per_epoch: int,
    pairs_per_batch: int,
    learning_rate: float,
    temperature: float,
    weight_decay: float,
    contrastive_weight: float,
    alignment_weight: float,
    structure_weight: float,
    view_consistency_weight: float,
    positive_pair_weight: float,
    hard_negative_weight: float,
    positive_exact_weight: float,
    positive_weak_weight: float,
    hard_negative_max_similarity: float,
    hard_negative_gap_scale: float,
    hard_negative_min_similarity: float,
    adversarial_negative_top_fraction: float,
    adversarial_negative_max_pairs: int,
    adversarial_negative_min_teacher_similarity: float,
    augmentation_min_crop_scale: float,
    augmentation_noise_std: float,
    augmentation_gamma_jitter: float,
    augmentation_blur_probability: float,
    device_name: str,
    amp_enabled: bool,
    seed: int,
    teacher_model: str,
    teacher_weights: str,
    student_image_size: int,
) -> tuple[Any, DistillationTrainingReport]:
    if len(student_images) != len(teacher_targets):
        raise ValueError("student_images and teacher_targets must have the same row count.")
    if len(positive_pairs) == 0:
        raise ValueError("No positive training pairs were provided.")
    if (
        contrastive_weight < 0.0
        or alignment_weight < 0.0
        or structure_weight < 0.0
        or view_consistency_weight < 0.0
        or positive_pair_weight < 0.0
        or hard_negative_weight < 0.0
    ):
        raise ValueError("Loss weights must be non-negative.")
    if positive_exact_weight <= 0.0 or positive_weak_weight <= 0.0:
        raise ValueError("Positive sampling weights must be strictly positive.")
    if not -1.0 <= hard_negative_max_similarity <= 1.0:
        raise ValueError("hard_negative_max_similarity must be within [-1, 1].")
    if hard_negative_gap_scale < 0.0:
        raise ValueError("hard_negative_gap_scale must be non-negative.")
    if not -1.0 <= hard_negative_min_similarity <= 1.0:
        raise ValueError("hard_negative_min_similarity must be within [-1, 1].")
    if hard_negative_min_similarity > hard_negative_max_similarity:
        raise ValueError(
            "hard_negative_min_similarity must not exceed hard_negative_max_similarity."
        )
    if not 0.0 < adversarial_negative_top_fraction <= 1.0:
        raise ValueError("adversarial_negative_top_fraction must be within (0, 1].")
    if adversarial_negative_max_pairs <= 0:
        raise ValueError("adversarial_negative_max_pairs must be positive.")
    if not 0.0 < augmentation_min_crop_scale <= 1.0:
        raise ValueError("augmentation_min_crop_scale must be within (0, 1].")
    if augmentation_noise_std < 0.0 or augmentation_gamma_jitter < 0.0:
        raise ValueError("Augmentation magnitudes must be non-negative.")
    if not 0.0 <= augmentation_blur_probability <= 1.0:
        raise ValueError("augmentation_blur_probability must be within [0, 1].")

    torch = require_torch()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device(device_name)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.cuda.reset_peak_memory_stats(device)

    teacher_targets = np.asarray(teacher_targets, dtype=np.float32)
    teacher_targets = teacher_targets / np.clip(
        np.linalg.norm(teacher_targets, axis=1, keepdims=True),
        1e-12,
        None,
    )
    embedding_dim = int(teacher_targets.shape[1])
    model = create_student_model(
        torch,
        arch=student_arch,
        input_channels=int(student_images.shape[1]),
        base_channels=base_channels,
        embedding_dim=embedding_dim,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and device.type == "cuda")
    rng = np.random.default_rng(seed)
    amp_context = torch.autocast if amp_enabled and device.type == "cuda" else None

    batch_latencies: list[float] = []
    losses: list[float] = []
    contrastive_losses: list[float] = []
    alignment_losses: list[float] = []
    structure_losses: list[float] = []
    view_consistency_losses: list[float] = []
    positive_pair_losses: list[float] = []
    hard_negative_losses: list[float] = []

    total_images_seen = 0
    start = perf_counter()
    for _epoch in range(epochs):
        model.train()
        for _step in range(steps_per_epoch):
            positive_batch_pairs, positive_batch_pair_indices = sample_weighted_pairs(
                positive_pairs,
                pair_weights=positive_pair_weights,
                pairs_per_batch=pairs_per_batch,
                rng=rng,
            )
            batch_indices = positive_batch_pairs.reshape(-1).astype(np.int64, copy=False)
            total_images_seen += int(len(batch_indices))
            batch_start = perf_counter()
            student_batch = torch.from_numpy(student_images[batch_indices]).to(
                device=device,
                dtype=torch.float32,
            )
            teacher_batch = torch.from_numpy(teacher_targets[batch_indices]).to(
                device=device,
                dtype=torch.float32,
            )
            optimizer.zero_grad(set_to_none=True)
            autocast_context = (
                amp_context(device_type="cuda", dtype=torch.float16)
                if amp_context is not None
                else nullcontext()
            )
            with autocast_context:
                student_view_a = augment_pan_batch(
                    torch,
                    student_batch,
                    rng=rng,
                    min_crop_scale=augmentation_min_crop_scale,
                    noise_std=augmentation_noise_std,
                    gamma_jitter=augmentation_gamma_jitter,
                    blur_probability=augmentation_blur_probability,
                )
                student_view_b = augment_pan_batch(
                    torch,
                    student_batch,
                    rng=rng,
                    min_crop_scale=augmentation_min_crop_scale,
                    noise_std=augmentation_noise_std,
                    gamma_jitter=augmentation_gamma_jitter,
                    blur_probability=augmentation_blur_probability,
                )
                student_embeddings_a = model(student_view_a)
                student_embeddings_b = model(student_view_b)
                student_embeddings = torch.nn.functional.normalize(
                    student_embeddings_a + student_embeddings_b,
                    dim=1,
                )
                contrastive_loss = nt_xent_loss(
                    torch,
                    student_embeddings,
                    temperature=temperature,
                )
                alignment_loss = 1.0 - torch.sum(student_embeddings * teacher_batch, dim=1).mean()
                structure_loss = teacher_structure_loss(
                    torch,
                    student_embeddings,
                    teacher_batch,
                )
                student_pair_embeddings = student_embeddings.reshape(
                    -1,
                    2,
                    student_embeddings.shape[1],
                )
                teacher_pair_embeddings = teacher_batch.reshape(
                    -1,
                    2,
                    teacher_batch.shape[1],
                )
                student_pair_cos = torch.sum(
                    student_pair_embeddings[:, 0, :] * student_pair_embeddings[:, 1, :],
                    dim=1,
                )
                teacher_pair_cos = torch.sum(
                    teacher_pair_embeddings[:, 0, :] * teacher_pair_embeddings[:, 1, :],
                    dim=1,
                )
                view_consistency_loss = 1.0 - torch.sum(
                    student_embeddings_a * student_embeddings_b,
                    dim=1,
                ).mean()
                selected_positive_weights = torch.from_numpy(
                    positive_pair_weights[positive_batch_pair_indices]
                ).to(device=device, dtype=torch.float32)
                positive_pair_loss = (
                    selected_positive_weights
                    * torch.square(student_pair_cos - teacher_pair_cos)
                ).sum() / torch.clamp(selected_positive_weights.sum(), min=1e-6)
                hard_negative_loss = torch.zeros((), device=device, dtype=student_embeddings.dtype)
                if hard_negative_weight > 0.0 and len(negative_pairs) > 0:
                    negative_batch_pairs, _negative_pair_indices = sample_weighted_pairs(
                        negative_pairs,
                        pair_weights=None,
                        pairs_per_batch=max(1, pairs_per_batch // 2),
                        rng=rng,
                    )
                    negative_indices = negative_batch_pairs.reshape(-1).astype(np.int64, copy=False)
                    total_images_seen += int(len(negative_indices))
                    negative_batch = torch.from_numpy(student_images[negative_indices]).to(
                        device=device,
                        dtype=torch.float32,
                    )
                    negative_view_a = augment_pan_batch(
                        torch,
                        negative_batch,
                        rng=rng,
                        min_crop_scale=augmentation_min_crop_scale,
                        noise_std=augmentation_noise_std,
                        gamma_jitter=augmentation_gamma_jitter,
                        blur_probability=augmentation_blur_probability,
                    )
                    negative_view_b = augment_pan_batch(
                        torch,
                        negative_batch,
                        rng=rng,
                        min_crop_scale=augmentation_min_crop_scale,
                        noise_std=augmentation_noise_std,
                        gamma_jitter=augmentation_gamma_jitter,
                        blur_probability=augmentation_blur_probability,
                    )
                    negative_embeddings = torch.nn.functional.normalize(
                        model(negative_view_a) + model(negative_view_b),
                        dim=1,
                    )
                    negative_teacher = torch.from_numpy(teacher_targets[negative_indices]).to(
                        device=device,
                        dtype=torch.float32,
                    )
                    negative_student_pairs = negative_embeddings.reshape(
                        -1,
                        2,
                        negative_embeddings.shape[1],
                    )
                    negative_teacher_pairs = negative_teacher.reshape(
                        -1,
                        2,
                        negative_teacher.shape[1],
                    )
                    negative_student_cos = torch.sum(
                        negative_student_pairs[:, 0, :] * negative_student_pairs[:, 1, :],
                        dim=1,
                    )
                    negative_teacher_cos = torch.sum(
                        negative_teacher_pairs[:, 0, :] * negative_teacher_pairs[:, 1, :],
                        dim=1,
                    )
                    positive_teacher_mean = teacher_pair_cos.mean()
                    dynamic_gap = torch.clamp(
                        positive_teacher_mean - negative_teacher_cos,
                        min=0.0,
                    )
                    negative_target = torch.minimum(
                        negative_teacher_cos - hard_negative_gap_scale * dynamic_gap,
                        torch.full_like(negative_teacher_cos, hard_negative_max_similarity),
                    )
                    negative_target = torch.clamp(
                        negative_target,
                        min=hard_negative_min_similarity,
                    )
                    hard_negative_loss = torch.relu(negative_student_cos - negative_target).mean()
                loss = (
                    contrastive_weight * contrastive_loss
                    + alignment_weight * alignment_loss
                    + structure_weight * structure_loss
                    + view_consistency_weight * view_consistency_loss
                    + positive_pair_weight * positive_pair_loss
                    + hard_negative_weight * hard_negative_loss
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            batch_latencies.append((perf_counter() - batch_start) * 1000.0)
            losses.append(float(loss.detach().cpu().item()))
            contrastive_losses.append(float(contrastive_loss.detach().cpu().item()))
            alignment_losses.append(float(alignment_loss.detach().cpu().item()))
            structure_losses.append(float(structure_loss.detach().cpu().item()))
            view_consistency_losses.append(float(view_consistency_loss.detach().cpu().item()))
            positive_pair_losses.append(float(positive_pair_loss.detach().cpu().item()))
            hard_negative_losses.append(float(hard_negative_loss.detach().cpu().item()))
    elapsed_seconds = perf_counter() - start

    max_memory_mib = 0.0
    if device.type == "cuda":
        max_memory_mib = float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))
    total_steps = epochs * steps_per_epoch
    report = DistillationTrainingReport(
        teacher_model=teacher_model,
        teacher_weights=teacher_weights,
        student_arch=student_arch,
        train_samples=int(len(student_images)),
        train_positive_pairs=int(len(positive_pairs)),
        train_positive_exact_pairs=positive_exact_pairs,
        train_positive_weak_pairs=positive_weak_pairs,
        train_hard_negative_pairs=int(len(negative_pairs)),
        hard_negative_teacher_similarity_mean=hard_negative_teacher_similarity_mean,
        hard_negative_teacher_similarity_p95=hard_negative_teacher_similarity_p95,
        student_image_size=student_image_size,
        student_base_channels=base_channels,
        student_embedding_dim=embedding_dim,
        contrastive_weight=contrastive_weight,
        alignment_weight=alignment_weight,
        structure_weight=structure_weight,
        view_consistency_weight=view_consistency_weight,
        positive_pair_weight=positive_pair_weight,
        hard_negative_weight=hard_negative_weight,
        positive_exact_weight=positive_exact_weight,
        positive_weak_weight=positive_weak_weight,
        hard_negative_max_similarity=hard_negative_max_similarity,
        hard_negative_gap_scale=hard_negative_gap_scale,
        hard_negative_min_similarity=hard_negative_min_similarity,
        adversarial_negative_top_fraction=adversarial_negative_top_fraction,
        adversarial_negative_max_pairs=adversarial_negative_max_pairs,
        adversarial_negative_min_teacher_similarity=adversarial_negative_min_teacher_similarity,
        augmentation_min_crop_scale=augmentation_min_crop_scale,
        augmentation_noise_std=augmentation_noise_std,
        augmentation_gamma_jitter=augmentation_gamma_jitter,
        augmentation_blur_probability=augmentation_blur_probability,
        device=device.type,
        device_name="CPU" if device.type == "cpu" else str(torch.cuda.get_device_name(device)),
        amp_enabled=bool(amp_enabled and device.type == "cuda"),
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pairs_per_batch=pairs_per_batch,
        batch_size=pairs_per_batch * 2,
        elapsed_seconds=elapsed_seconds,
        steps_per_second=safe_rate(total_steps, elapsed_seconds),
        images_per_second=safe_rate(total_images_seen, elapsed_seconds),
        batch_latency_ms_mean=mean(batch_latencies),
        batch_latency_ms_p95=percentile(batch_latencies, 95.0),
        loss_initial=float(losses[0]),
        loss_final=float(losses[-1]),
        loss_best=float(min(losses)),
        contrastive_loss_mean=mean(contrastive_losses),
        alignment_loss_mean=mean(alignment_losses),
        structure_loss_mean=mean(structure_losses),
        view_consistency_loss_mean=mean(view_consistency_losses),
        positive_pair_loss_mean=mean(positive_pair_losses),
        hard_negative_loss_mean=mean(hard_negative_losses),
        max_memory_mib=max_memory_mib,
    )
    return model, report


def write_outputs(
    *,
    run_root: Path,
    teacher_model_name: str,
    teacher_benchmark_train: PretrainedBenchmarkReport,
    teacher_benchmark_eval: PretrainedBenchmarkReport,
    teacher_train_retrieval: PairRetrievalReport | None,
    teacher_eval_retrieval: PairRetrievalReport,
    student_train_preprocess: PreprocessReport,
    student_eval_preprocess: PreprocessReport,
    student_training: DistillationTrainingReport,
    student_eval_embedding: TorchEmbeddingReport,
    student_eval_retrieval: PairRetrievalReport,
    teacher_train_embeddings: np.ndarray,
    teacher_eval_embeddings: np.ndarray,
    student_eval_embeddings: np.ndarray,
    teacher_train_metadata: pd.DataFrame,
    teacher_eval_metadata: pd.DataFrame,
    student_train_records: Sequence[dict[str, object]],
    student_eval_records: Sequence[dict[str, object]],
    model: Any,
) -> dict[str, Path]:
    torch = require_torch()
    output_root = run_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    model_path = output_root / "student_model.pt"
    teacher_train_embeddings_path = output_root / "teacher_train_embeddings.npy"
    teacher_eval_embeddings_path = output_root / "teacher_eval_embeddings.npy"
    student_eval_embeddings_path = output_root / "student_eval_embeddings.npy"
    teacher_train_index_path = output_root / "teacher_train_index.parquet"
    teacher_eval_index_path = output_root / "teacher_eval_index.parquet"
    student_train_index_path = output_root / "student_train_index.parquet"
    student_eval_index_path = output_root / "student_eval_index.parquet"
    teacher_train_benchmark_path = output_root / "teacher_train_benchmark.json"
    teacher_eval_benchmark_path = output_root / "teacher_eval_benchmark.json"
    teacher_train_retrieval_path = output_root / "teacher_train_retrieval.json"
    teacher_eval_retrieval_path = output_root / "teacher_eval_retrieval.json"
    student_train_preprocess_path = output_root / "student_train_preprocess.json"
    student_eval_preprocess_path = output_root / "student_eval_preprocess.json"
    student_training_path = output_root / "student_training.json"
    student_eval_embedding_path = output_root / "student_eval_embedding.json"
    student_eval_retrieval_path = output_root / "student_eval_retrieval.json"
    summary_path = output_root / "summary.json"

    torch.save({"state_dict": model.state_dict()}, model_path)
    np.save(teacher_train_embeddings_path, teacher_train_embeddings)
    np.save(teacher_eval_embeddings_path, teacher_eval_embeddings)
    np.save(student_eval_embeddings_path, student_eval_embeddings)
    teacher_train_metadata.to_parquet(teacher_train_index_path, index=False, compression="zstd")
    teacher_eval_metadata.to_parquet(teacher_eval_index_path, index=False, compression="zstd")
    pd.DataFrame(list(student_train_records)).to_parquet(
        student_train_index_path,
        index=False,
        compression="zstd",
    )
    pd.DataFrame(list(student_eval_records)).to_parquet(
        student_eval_index_path,
        index=False,
        compression="zstd",
    )
    teacher_train_benchmark_path.write_text(
        json.dumps(asdict(teacher_benchmark_train), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    teacher_eval_benchmark_path.write_text(
        json.dumps(asdict(teacher_benchmark_eval), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    teacher_train_retrieval_path.write_text(
        json.dumps(
            asdict(teacher_train_retrieval) if teacher_train_retrieval is not None else None,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    teacher_eval_retrieval_path.write_text(
        json.dumps(asdict(teacher_eval_retrieval), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    student_train_preprocess_path.write_text(
        json.dumps(asdict(student_train_preprocess), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    student_eval_preprocess_path.write_text(
        json.dumps(asdict(student_eval_preprocess), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    student_training_path.write_text(
        json.dumps(asdict(student_training), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    student_eval_embedding_path.write_text(
        json.dumps(asdict(student_eval_embedding), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    student_eval_retrieval_path.write_text(
        json.dumps(asdict(student_eval_retrieval), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(
            {
                "teacher_model": teacher_model_name,
                "teacher_eval_retrieval": asdict(teacher_eval_retrieval),
                "student_eval_retrieval": asdict(student_eval_retrieval),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return {
        "model": model_path,
        "summary": summary_path,
        "teacher_eval_retrieval": teacher_eval_retrieval_path,
        "student_eval_retrieval": student_eval_retrieval_path,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    torch = require_torch()

    chips_path = args.chips_path.resolve()
    pairs_path = args.pairs_path.resolve()
    run_root = args.run_root.resolve()
    gdal_prefix = args.gdal_prefix.resolve() if args.gdal_prefix is not None else None
    load_gdal(gdal_prefix)
    _torchvision = require_torchvision()
    train_splits = normalize_multi_arg(args.train_split, default=("train",))
    query_splits = normalize_multi_arg(args.query_split, default=("val", "test"))
    gallery_splits = normalize_multi_arg(args.gallery_split, default=("val", "test"))
    modalities = normalize_multi_arg(args.modality, default=("PAN",))
    device_name, resolved_device_name = resolve_device(torch, args.device)
    pair_frame = pd.read_parquet(pairs_path)

    train_dataset, train_pairs = build_pan_adapt_training_dataset(
        chips_path,
        splits=train_splits,
        modalities=modalities,
        pair_frame=pair_frame,
        limit=args.train_limit,
        gdal_prefix=gdal_prefix,
        output_dtype=args.output_dtype,
        clip_min=args.clip_min,
        clip_max=args.clip_max,
        scale_max=args.scale_max,
    )
    eval_dataset, eval_pairs = build_eval_dataset(
        chips_path,
        pair_frame=pair_frame,
        query_splits=query_splits,
        gallery_splits=gallery_splits,
        modalities=modalities,
        limit=args.eval_limit,
        gdal_prefix=gdal_prefix,
        output_dtype=args.output_dtype,
        clip_min=args.clip_min,
        clip_max=args.clip_max,
        scale_max=args.scale_max,
    )

    teacher_model, teacher_spec, _parameter_count = load_pretrained_model(
        args.teacher_model,
        device_name=device_name,
    )
    (
        teacher_train_embeddings,
        teacher_train_metadata,
        teacher_train_benchmark,
    ) = embed_teacher_dataset(
        train_dataset,
        model=teacher_model,
        spec=teacher_spec,
        batch_size=args.teacher_batch_size,
        device_name=device_name,
        amp_enabled=args.amp,
    )
    (
        teacher_eval_embeddings,
        teacher_eval_metadata,
        teacher_eval_benchmark,
    ) = embed_teacher_dataset(
        eval_dataset,
        model=teacher_model,
        spec=teacher_spec,
        batch_size=args.teacher_batch_size,
        device_name=device_name,
        amp_enabled=args.amp,
    )

    train_images, _train_indices, train_preprocess_report, train_records = extract_images(
        train_dataset,
        image_size=args.student_image_size,
    )
    eval_images, _eval_indices, eval_preprocess_report, eval_records = extract_images(
        eval_dataset,
        image_size=args.student_image_size,
    )
    assert_record_alignment(
        expected_records=train_records,
        actual_metadata=teacher_train_metadata,
        field_name="teacher_train",
    )
    assert_record_alignment(
        expected_records=eval_records,
        actual_metadata=teacher_eval_metadata,
        field_name="teacher_eval",
    )

    filtered_train_pairs = filter_pairs_for_records(
        train_pairs,
        pd.DataFrame(train_records),
        pair_labels={"positive_exact", "positive_weak", "negative_hard"},
    )
    positive_pairs, positive_pair_weights, exact_pairs, weak_pairs = build_weighted_positive_pairs(
        train_records,
        filtered_train_pairs,
        exact_weight=args.positive_exact_weight,
        weak_weight=args.positive_weak_weight,
    )
    negative_pairs, negative_similarity_mean, negative_similarity_p95 = (
        mine_adversarial_negative_pairs(
            train_records,
            filtered_train_pairs,
            teacher_train_embeddings,
            top_fraction=args.adversarial_negative_top_fraction,
            max_pairs=args.adversarial_negative_max_pairs,
            min_teacher_similarity=args.adversarial_negative_min_teacher_similarity,
        )
    )
    if len(negative_pairs) == 0:
        negative_pairs = build_negative_pairs(train_records, filtered_train_pairs)
        negative_similarity_mean, negative_similarity_p95 = summarize_pair_teacher_similarities(
            negative_pairs,
            teacher_train_embeddings,
        )
    if len(positive_pairs) == 0:
        raise SystemExit("No positive training pairs remain after preprocessing.")

    teacher_train_retrieval: PairRetrievalReport | None = None
    train_and_eval_overlap = set(train_splits).intersection(set(query_splits).union(gallery_splits))
    if query_splits and gallery_splits and train_and_eval_overlap:
        train_query_gallery = tuple(dict.fromkeys(train_splits))
        teacher_train_retrieval = evaluate_pair_retrieval(
            teacher_train_embeddings,
            teacher_train_metadata,
            train_pairs,
            query_splits=train_query_gallery,
            gallery_splits=train_query_gallery,
        )
    teacher_eval_retrieval = evaluate_pair_retrieval(
        teacher_eval_embeddings,
        teacher_eval_metadata,
        eval_pairs,
        query_splits=query_splits,
        gallery_splits=gallery_splits,
    )

    student_model, student_training = train_student_with_teacher(
        train_images,
        teacher_train_embeddings,
        positive_pairs,
        positive_pair_weights,
        negative_pairs,
        positive_exact_pairs=exact_pairs,
        positive_weak_pairs=weak_pairs,
        hard_negative_teacher_similarity_mean=negative_similarity_mean,
        hard_negative_teacher_similarity_p95=negative_similarity_p95,
        student_arch=args.student_arch,
        base_channels=args.student_base_channels,
        dropout=args.student_dropout,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        pairs_per_batch=args.pairs_per_batch,
        learning_rate=args.learning_rate,
        temperature=args.temperature,
        weight_decay=args.weight_decay,
        contrastive_weight=args.contrastive_weight,
        alignment_weight=args.alignment_weight,
        structure_weight=args.structure_weight,
        view_consistency_weight=args.view_consistency_weight,
        positive_pair_weight=args.positive_pair_weight,
        hard_negative_weight=args.hard_negative_weight,
        positive_exact_weight=args.positive_exact_weight,
        positive_weak_weight=args.positive_weak_weight,
        hard_negative_max_similarity=args.hard_negative_max_similarity,
        hard_negative_gap_scale=args.hard_negative_gap_scale,
        hard_negative_min_similarity=args.hard_negative_min_similarity,
        adversarial_negative_top_fraction=args.adversarial_negative_top_fraction,
        adversarial_negative_max_pairs=args.adversarial_negative_max_pairs,
        adversarial_negative_min_teacher_similarity=args.adversarial_negative_min_teacher_similarity,
        augmentation_min_crop_scale=args.augmentation_min_crop_scale,
        augmentation_noise_std=args.augmentation_noise_std,
        augmentation_gamma_jitter=args.augmentation_gamma_jitter,
        augmentation_blur_probability=args.augmentation_blur_probability,
        device_name=device_name,
        amp_enabled=args.amp,
        seed=args.seed,
        teacher_model=teacher_spec.name,
        teacher_weights=teacher_spec.weights_label,
        student_image_size=args.student_image_size,
    )
    student_eval_embeddings, student_eval_embedding = embed_images_torch(
        student_model,
        eval_images,
        eval_batch_size=args.eval_batch_size,
        device_name=device_name,
    )
    student_eval_retrieval = evaluate_pair_retrieval(
        student_eval_embeddings,
        pd.DataFrame(eval_records),
        eval_pairs,
        query_splits=query_splits,
        gallery_splits=gallery_splits,
    )
    paths = write_outputs(
        run_root=run_root,
        teacher_model_name=teacher_spec.name,
        teacher_benchmark_train=teacher_train_benchmark,
        teacher_benchmark_eval=teacher_eval_benchmark,
        teacher_train_retrieval=teacher_train_retrieval,
        teacher_eval_retrieval=teacher_eval_retrieval,
        student_train_preprocess=train_preprocess_report,
        student_eval_preprocess=eval_preprocess_report,
        student_training=student_training,
        student_eval_embedding=student_eval_embedding,
        student_eval_retrieval=student_eval_retrieval,
        teacher_train_embeddings=teacher_train_embeddings,
        teacher_eval_embeddings=teacher_eval_embeddings,
        student_eval_embeddings=student_eval_embeddings,
        teacher_train_metadata=teacher_train_metadata,
        teacher_eval_metadata=teacher_eval_metadata,
        student_train_records=train_records,
        student_eval_records=eval_records,
        model=student_model,
    )

    print(f"teacher: {teacher_spec.name} weights={teacher_spec.weights_label}")
    print(
        f"device: {device_name} ({resolved_device_name}) "
        f"amp={bool(args.amp and device_name == 'cuda')}"
    )
    print(
        f"teacher_eval: exact_R@10={teacher_eval_retrieval.exact_recall_at_10:.3f} "
        f"any_R@10={teacher_eval_retrieval.any_recall_at_10:.3f} "
        f"any_MRR={teacher_eval_retrieval.any_mean_reciprocal_rank:.3f}"
    )
    print(
        f"student_eval: exact_R@10={student_eval_retrieval.exact_recall_at_10:.3f} "
        f"any_R@10={student_eval_retrieval.any_recall_at_10:.3f} "
        f"any_MRR={student_eval_retrieval.any_mean_reciprocal_rank:.3f}"
    )
    print(
        "student_training: "
        f"loss={student_training.loss_initial:.4f}->{student_training.loss_final:.4f} "
        f"contrastive_mean={student_training.contrastive_loss_mean:.4f} "
        f"alignment_mean={student_training.alignment_loss_mean:.4f} "
        f"struct_mean={student_training.structure_loss_mean:.4f} "
        f"view_mean={student_training.view_consistency_loss_mean:.4f} "
        f"pospair_mean={student_training.positive_pair_loss_mean:.4f} "
        f"hardneg_mean={student_training.hard_negative_loss_mean:.4f} "
        f"images/s={student_training.images_per_second:.2f}"
    )
    print(
        "hard_negative_mining: "
        f"pairs={student_training.train_hard_negative_pairs} "
        f"teacher_sim_mean={student_training.hard_negative_teacher_similarity_mean:.4f} "
        f"teacher_sim_p95={student_training.hard_negative_teacher_similarity_p95:.4f}"
    )
    print(f"summary: {paths['summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
