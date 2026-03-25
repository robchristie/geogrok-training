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

from geogrok.retrieval.baseline import (
    RetrievalReport,
    build_dataset,
    evaluate_retrieval,
    mean,
    percentile,
    safe_rate,
)
from geogrok.retrieval.cnn import PreprocessReport, extract_images
from geogrok.retrieval.learned import sample_pair_batch

DEFAULT_RUN_ROOT = Path("artifacts/runs/torch-embedding-baseline")


@dataclass(frozen=True)
class TorchTrainingReport:
    device: str
    device_name: str
    amp_enabled: bool
    train_samples: int
    train_scenes: int
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
    max_memory_mib: float


@dataclass(frozen=True)
class TorchEmbeddingReport:
    device: str
    device_name: str
    samples: int
    batch_size: int
    embedding_dim: int
    elapsed_seconds: float
    samples_per_second: float
    batch_latency_ms_mean: float
    batch_latency_ms_p95: float
    max_memory_mib: float


def require_torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "PyTorch is not installed in this repo environment. "
            "Run `uv sync --extra dev --extra train`."
        ) from exc
    return torch


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a small PyTorch PAN retrieval encoder on on-demand chips."
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
    parser.add_argument("--train-limit", type=int, default=256)
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
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--base-channels", type=int, default=48)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--steps-per-epoch", type=int, default=48)
    parser.add_argument("--pairs-per-batch", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Training and inference device.",
    )
    parser.add_argument("--amp", action="store_true", help="Use mixed precision on CUDA.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def build_positive_groups(
    metadata_records: Sequence[dict[str, object]],
    *,
    positive_key: str,
) -> list[np.ndarray]:
    metadata = pd.DataFrame(metadata_records)
    metadata["positive_label"] = metadata[positive_key].fillna("null").astype(str)
    return [
        group.index.to_numpy(dtype=np.int64)
        for _, group in metadata.groupby("positive_label", sort=False)
        if len(group) >= 2
    ]


def resolve_device(torch: Any, requested: str) -> tuple[str, str]:
    if requested == "cpu":
        return "cpu", "CPU"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA was requested but is not available.")
        return "cuda", str(torch.cuda.get_device_name(0))
    if torch.cuda.is_available():
        return "cuda", str(torch.cuda.get_device_name(0))
    return "cpu", "CPU"


def create_model(
    torch: Any,
    *,
    input_channels: int,
    base_channels: int,
    embedding_dim: int,
    dropout: float,
) -> Any:
    nn = torch.nn

    class PanEmbeddingNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(input_channels, base_channels, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(base_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(base_channels * 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(base_channels * 4),
                nn.ReLU(inplace=True),
                nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(base_channels * 4),
                nn.ReLU(inplace=True),
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
            x = self.features(x)
            x = self.pool(x)
            x = self.projection(x)
            return torch.nn.functional.normalize(x, dim=1)

    return PanEmbeddingNet()


def augment_batch(torch: Any, batch: Any) -> Any:
    flip_h = torch.rand(batch.shape[0], device=batch.device) < 0.5
    flip_v = torch.rand(batch.shape[0], device=batch.device) < 0.5
    gain = torch.empty((batch.shape[0], 1, 1, 1), device=batch.device).uniform_(0.9, 1.1)
    noise = torch.randn_like(batch) * 0.01
    augmented = batch.clone()
    if bool(flip_h.any()):
        augmented[flip_h] = torch.flip(augmented[flip_h], dims=(3,))
    if bool(flip_v.any()):
        augmented[flip_v] = torch.flip(augmented[flip_v], dims=(2,))
    augmented = torch.clamp(augmented * gain + noise, 0.0, 1.0)
    return augmented


def nt_xent_loss(torch: Any, embeddings: Any, *, temperature: float) -> Any:
    logits = embeddings @ embeddings.T / temperature
    indices = torch.arange(logits.shape[0], device=embeddings.device)
    logits[indices, indices] = float("-inf")
    targets = indices ^ 1
    return torch.nn.functional.cross_entropy(logits, targets)


def train_torch_encoder(
    images: np.ndarray,
    metadata_records: Sequence[dict[str, object]],
    *,
    positive_key: str,
    base_channels: int,
    embedding_dim: int,
    dropout: float,
    epochs: int,
    steps_per_epoch: int,
    pairs_per_batch: int,
    learning_rate: float,
    temperature: float,
    weight_decay: float,
    device_name: str,
    amp_enabled: bool,
    seed: int,
) -> tuple[Any, TorchTrainingReport]:
    if epochs <= 0 or steps_per_epoch <= 0 or pairs_per_batch <= 0:
        raise ValueError("epochs, steps_per_epoch, and pairs_per_batch must be positive.")

    torch = require_torch()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    positive_groups = build_positive_groups(metadata_records, positive_key=positive_key)
    if not positive_groups:
        raise ValueError("No positive groups with at least two chips were found for training.")

    device = torch.device(device_name)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.cuda.reset_peak_memory_stats(device)

    model = create_model(
        torch,
        input_channels=int(images.shape[1]),
        base_channels=base_channels,
        embedding_dim=embedding_dim,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    amp_context = torch.autocast if amp_enabled and device.type == "cuda" else None
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and device.type == "cuda")
    rng = np.random.default_rng(seed)

    batch_latencies: list[float] = []
    losses: list[float] = []
    start = perf_counter()
    for _epoch in range(epochs):
        model.train()
        for _step in range(steps_per_epoch):
            batch_indices = sample_pair_batch(
                positive_groups,
                pairs_per_batch=pairs_per_batch,
                rng=rng,
            )
            batch_start = perf_counter()
            batch = torch.from_numpy(images[batch_indices]).to(device=device, dtype=torch.float32)
            batch = augment_batch(torch, batch)
            optimizer.zero_grad(set_to_none=True)
            autocast_context = (
                amp_context(device_type="cuda", dtype=torch.float16)
                if amp_context is not None
                else nullcontext()
            )
            with autocast_context:
                embeddings = model(batch)
                loss = nt_xent_loss(torch, embeddings, temperature=temperature)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            batch_latencies.append((perf_counter() - batch_start) * 1000.0)
            losses.append(float(loss.detach().cpu().item()))
    elapsed_seconds = perf_counter() - start

    max_memory_mib = 0.0
    if device.type == "cuda":
        max_memory_mib = float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))
    total_steps = epochs * steps_per_epoch
    total_images = total_steps * pairs_per_batch * 2
    report = TorchTrainingReport(
        device=device.type,
        device_name="CPU" if device.type == "cpu" else str(torch.cuda.get_device_name(device)),
        amp_enabled=bool(amp_enabled and device.type == "cuda"),
        train_samples=int(len(images)),
        train_scenes=len(positive_groups),
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pairs_per_batch=pairs_per_batch,
        batch_size=pairs_per_batch * 2,
        elapsed_seconds=elapsed_seconds,
        steps_per_second=safe_rate(total_steps, elapsed_seconds),
        images_per_second=safe_rate(total_images, elapsed_seconds),
        batch_latency_ms_mean=mean(batch_latencies),
        batch_latency_ms_p95=percentile(batch_latencies, 95.0),
        loss_initial=float(losses[0]),
        loss_final=float(losses[-1]),
        loss_best=float(min(losses)),
        max_memory_mib=max_memory_mib,
    )
    return model, report


def embed_images_torch(
    model: Any,
    images: np.ndarray,
    *,
    eval_batch_size: int,
    device_name: str,
) -> tuple[np.ndarray, TorchEmbeddingReport]:
    torch = require_torch()
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    outputs: list[np.ndarray] = []
    batch_latencies: list[float] = []
    start = perf_counter()
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(images), eval_batch_size):
            batch_start = perf_counter()
            batch_np = images[offset : offset + eval_batch_size]
            batch = torch.from_numpy(batch_np).to(device=device, dtype=torch.float32)
            embeddings = model(batch)
            outputs.append(embeddings.detach().cpu().numpy().astype(np.float32, copy=False))
            batch_latencies.append((perf_counter() - batch_start) * 1000.0)
    elapsed_seconds = perf_counter() - start

    max_memory_mib = 0.0
    if device.type == "cuda":
        max_memory_mib = float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))
    matrix = np.concatenate(outputs, axis=0) if outputs else np.empty((0, 0), dtype=np.float32)
    report = TorchEmbeddingReport(
        device=device.type,
        device_name="CPU" if device.type == "cpu" else str(torch.cuda.get_device_name(device)),
        samples=int(len(matrix)),
        batch_size=eval_batch_size,
        embedding_dim=int(matrix.shape[1]) if matrix.size else 0,
        elapsed_seconds=elapsed_seconds,
        samples_per_second=safe_rate(len(matrix), elapsed_seconds),
        batch_latency_ms_mean=mean(batch_latencies),
        batch_latency_ms_p95=percentile(batch_latencies, 95.0),
        max_memory_mib=max_memory_mib,
    )
    return matrix.astype(np.float32, copy=False), report


def write_outputs(
    *,
    model: Any,
    train_images: np.ndarray,
    eval_embeddings: np.ndarray,
    train_metadata_records: Sequence[dict[str, object]],
    eval_metadata_records: Sequence[dict[str, object]],
    train_preprocess_report: PreprocessReport,
    eval_preprocess_report: PreprocessReport,
    training_report: TorchTrainingReport,
    embedding_report: TorchEmbeddingReport,
    retrieval_report: RetrievalReport,
    run_root: Path,
) -> dict[str, Path]:
    torch = require_torch()
    output_root = run_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    model_path = output_root / "model.pt"
    train_images_path = output_root / "train_images.npy"
    eval_embeddings_path = output_root / "eval_embeddings.npy"
    train_index_path = output_root / "train_index.parquet"
    eval_index_path = output_root / "eval_index.parquet"
    train_preprocess_path = output_root / "train_preprocess_benchmark.json"
    eval_preprocess_path = output_root / "eval_preprocess_benchmark.json"
    training_path = output_root / "training.json"
    embedding_path = output_root / "embedding.json"
    retrieval_path = output_root / "retrieval.json"

    torch.save({"state_dict": model.state_dict()}, model_path)
    np.save(train_images_path, train_images)
    np.save(eval_embeddings_path, eval_embeddings)
    pd.DataFrame(list(train_metadata_records)).to_parquet(
        train_index_path,
        index=False,
        compression="zstd",
    )
    pd.DataFrame(list(eval_metadata_records)).to_parquet(
        eval_index_path,
        index=False,
        compression="zstd",
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
    torch = require_torch()

    chips_path = args.chips_path.resolve()
    run_root = args.run_root.resolve()
    gdal_prefix = args.gdal_prefix.resolve() if args.gdal_prefix is not None else None
    train_splits = tuple(dict.fromkeys(args.train_split))
    query_splits = tuple(dict.fromkeys(args.query_split))
    gallery_splits = tuple(dict.fromkeys(args.gallery_split))
    eval_splits = tuple(sorted(set(query_splits).union(gallery_splits)))
    device_name, resolved_device_name = resolve_device(torch, args.device)

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

    model, training_report = train_torch_encoder(
        train_images,
        train_records,
        positive_key=args.positive_key,
        base_channels=args.base_channels,
        embedding_dim=args.embedding_dim,
        dropout=args.dropout,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        pairs_per_batch=args.pairs_per_batch,
        learning_rate=args.learning_rate,
        temperature=args.temperature,
        weight_decay=args.weight_decay,
        device_name=device_name,
        amp_enabled=args.amp,
        seed=args.seed,
    )
    eval_embeddings, embedding_report = embed_images_torch(
        model,
        eval_images,
        eval_batch_size=args.eval_batch_size,
        device_name=device_name,
    )
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
    print(f"device: {device_name} ({resolved_device_name}) amp={training_report.amp_enabled}")
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
        f"training: emb_dim={args.embedding_dim} "
        f"loss={training_report.loss_initial:.4f}->{training_report.loss_final:.4f} "
        f"images/s={training_report.images_per_second:.2f} "
        f"max_mem_mib={training_report.max_memory_mib:.1f}"
    )
    print(
        f"embed_eval: samples/s={embedding_report.samples_per_second:.2f} "
        f"batch_p95={embedding_report.batch_latency_ms_p95:.2f} "
        f"max_mem_mib={embedding_report.max_memory_mib:.1f}"
    )
    print(
        f"retrieval: R@1={retrieval_report.recall_at_1:.3f} "
        f"R@5={retrieval_report.recall_at_5:.3f} "
        f"R@10={retrieval_report.recall_at_10:.3f} "
        f"MRR={retrieval_report.mean_reciprocal_rank:.3f}"
    )
    print(f"model: {paths['model']}")
    print(f"training_report: {paths['training']}")
    print(f"embedding_report: {paths['embedding']}")
    print(f"retrieval: {paths['retrieval']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
