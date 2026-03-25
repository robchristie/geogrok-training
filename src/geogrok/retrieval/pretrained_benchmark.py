from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
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
from geogrok.retrieval.pair_eval import (
    PairRetrievalReport,
    chip_ids_from_pairs,
    evaluate_pair_retrieval,
)
from geogrok.retrieval.torch_encoder import filter_pairs_for_records

DEFAULT_RUN_ROOT = Path("artifacts/runs/pretrained-benchmark")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


@dataclass(frozen=True)
class PretrainedModelSpec:
    name: str
    weights_label: str
    input_size: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    loader: Callable[[Any, Any], Any]
    encoder_kind: str = "torchvision"


@dataclass(frozen=True)
class PretrainedBenchmarkReport:
    model_name: str
    weights_label: str
    device: str
    device_name: str
    amp_enabled: bool
    samples: int
    batch_size: int
    embedding_dim: int
    parameter_count: int
    elapsed_seconds: float
    samples_per_second: float
    read_latency_ms_mean: float
    read_latency_ms_p95: float
    transform_latency_ms_mean: float
    transform_latency_ms_p95: float
    model_prep_latency_ms_mean: float
    model_prep_latency_ms_p95: float
    forward_latency_ms_mean: float
    forward_latency_ms_p95: float
    total_latency_ms_mean: float
    total_latency_ms_p95: float
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


def require_torchvision() -> Any:
    try:
        import torchvision
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "torchvision is not installed in this repo environment. "
            "Run `uv sync --extra dev --extra train`."
        ) from exc
    return torchvision


def require_huggingface_hub() -> Any:
    try:
        import huggingface_hub
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "huggingface_hub is not installed in this repo environment. "
            "Run `uv sync --extra dev --extra train`."
        ) from exc
    return huggingface_hub


def require_open_clip() -> Any:
    try:
        import open_clip
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "open_clip_torch is not installed in this repo environment. "
            "Run `uv sync --extra dev --extra train`."
        ) from exc
    return open_clip


def require_timm() -> Any:
    try:
        import timm
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "timm is not installed in this repo environment. "
            "Run `uv sync --extra dev --extra train`."
        ) from exc
    return timm


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark frozen pretrained image encoders on held-out PAN retrieval."
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
        help="Explicit pairs.parquet used for held-out retrieval evaluation.",
    )
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument(
        "--gdal-prefix",
        type=Path,
        help="Override the local GDAL/Kakadu install prefix.",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=None,
        help="Pretrained model name. Repeat to benchmark multiple models.",
    )
    parser.add_argument(
        "--query-split",
        action="append",
        default=None,
        help="Query split for retrieval evaluation. Repeat to add more splits.",
    )
    parser.add_argument(
        "--gallery-split",
        action="append",
        default=None,
        help="Gallery split for retrieval evaluation. Repeat to add more splits.",
    )
    parser.add_argument(
        "--modality",
        action="append",
        default=None,
        help="Modality to include. Repeat to add more modalities.",
    )
    parser.add_argument(
        "--eval-limit",
        type=int,
        help="Optional cap on eval chips after pair filtering.",
    )
    parser.add_argument("--output-dtype", default="float32")
    parser.add_argument("--clip-min", type=float, default=0.0)
    parser.add_argument("--clip-max", type=float, default=2047.0)
    parser.add_argument("--scale-max", type=float, default=2047.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Inference device.",
    )
    parser.add_argument("--amp", action="store_true", help="Use mixed precision on CUDA.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def normalize_multi_arg(values: Sequence[str] | None, *, default: Sequence[str]) -> tuple[str, ...]:
    selected = values if values else default
    return tuple(dict.fromkeys(selected))


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


def available_model_specs() -> dict[str, PretrainedModelSpec]:
    torchvision = require_torchvision()
    models = torchvision.models
    torch = require_torch()

    def load_resnet18(_torch: Any, _torchvision: Any) -> Any:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = torch.nn.Identity()
        return model

    def load_resnet50(_torch: Any, _torchvision: Any) -> Any:
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = torch.nn.Identity()
        return model

    def load_resnet101(_torch: Any, _torchvision: Any) -> Any:
        model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        model.fc = torch.nn.Identity()
        return model

    def load_resnet152(_torch: Any, _torchvision: Any) -> Any:
        model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
        model.fc = torch.nn.Identity()
        return model

    def load_vit_b_16(_torch: Any, _torchvision: Any) -> Any:
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        model.heads = torch.nn.Identity()
        return model

    def load_remoteclip_rn50(_torch: Any, _torchvision: Any) -> Any:
        open_clip = require_open_clip()
        huggingface_hub = require_huggingface_hub()
        checkpoint_path = huggingface_hub.hf_hub_download(
            repo_id="chendelong/RemoteCLIP",
            filename="RemoteCLIP-RN50.pt",
        )
        model, _, _ = open_clip.create_model_and_transforms("RN50", pretrained="openai")
        checkpoint = _torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint, strict=False)
        return model

    def load_georsclip_vit_b32(_torch: Any, _torchvision: Any) -> Any:
        open_clip = require_open_clip()
        huggingface_hub = require_huggingface_hub()
        checkpoint_path = huggingface_hub.hf_hub_download(
            repo_id="Zilun/GeoRSCLIP",
            filename="ckpt/RS5M_ViT-B-32.pt",
        )
        model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
        checkpoint = _torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint, strict=False)
        return model

    def load_georsclip_vit_b32_ret2(_torch: Any, _torchvision: Any) -> Any:
        open_clip = require_open_clip()
        huggingface_hub = require_huggingface_hub()
        checkpoint_path = huggingface_hub.hf_hub_download(
            repo_id="Zilun/GeoRSCLIP",
            filename="ckpt/RS5M_ViT-B-32_RET-2.pt",
        )
        model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
        checkpoint = _torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint, strict=False)
        return model

    def load_dinov2_vitb14(_torch: Any, _torchvision: Any) -> Any:
        timm = require_timm()
        return timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True, num_classes=0)

    def load_dinov3_vitb16(_torch: Any, _torchvision: Any) -> Any:
        timm = require_timm()
        return timm.create_model("vit_base_patch16_dinov3.lvd1689m", pretrained=True, num_classes=0)

    return {
        "resnet18": PretrainedModelSpec(
            name="resnet18",
            weights_label="imagenet1k_v1",
            input_size=224,
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
            loader=load_resnet18,
        ),
        "resnet50": PretrainedModelSpec(
            name="resnet50",
            weights_label="imagenet1k_v2",
            input_size=224,
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
            loader=load_resnet50,
        ),
        "resnet101": PretrainedModelSpec(
            name="resnet101",
            weights_label="imagenet1k_v2",
            input_size=224,
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
            loader=load_resnet101,
        ),
        "resnet152": PretrainedModelSpec(
            name="resnet152",
            weights_label="imagenet1k_v2",
            input_size=224,
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
            loader=load_resnet152,
        ),
        "vit_b_16": PretrainedModelSpec(
            name="vit_b_16",
            weights_label="imagenet1k_swag_e2e_v1",
            input_size=224,
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
            loader=load_vit_b_16,
        ),
        "remoteclip_rn50": PretrainedModelSpec(
            name="remoteclip_rn50",
            weights_label="chendelong/RemoteCLIP:RemoteCLIP-RN50.pt",
            input_size=224,
            mean=CLIP_MEAN,
            std=CLIP_STD,
            loader=load_remoteclip_rn50,
            encoder_kind="open_clip",
        ),
        "georsclip_vit_b32": PretrainedModelSpec(
            name="georsclip_vit_b32",
            weights_label="Zilun/GeoRSCLIP:ckpt/RS5M_ViT-B-32.pt",
            input_size=224,
            mean=CLIP_MEAN,
            std=CLIP_STD,
            loader=load_georsclip_vit_b32,
            encoder_kind="open_clip",
        ),
        "georsclip_vit_b32_ret2": PretrainedModelSpec(
            name="georsclip_vit_b32_ret2",
            weights_label="Zilun/GeoRSCLIP:ckpt/RS5M_ViT-B-32_RET-2.pt",
            input_size=224,
            mean=CLIP_MEAN,
            std=CLIP_STD,
            loader=load_georsclip_vit_b32_ret2,
            encoder_kind="open_clip",
        ),
        "dinov2_vitb14": PretrainedModelSpec(
            name="dinov2_vitb14",
            weights_label="timm:vit_base_patch14_dinov2.lvd142m",
            input_size=518,
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
            loader=load_dinov2_vitb14,
            encoder_kind="timm",
        ),
        "dinov3_vitb16": PretrainedModelSpec(
            name="dinov3_vitb16",
            weights_label="timm:vit_base_patch16_dinov3.lvd1689m",
            input_size=256,
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
            loader=load_dinov3_vitb16,
            encoder_kind="timm",
        ),
    }


def load_pretrained_model(
    model_name: str,
    *,
    device_name: str,
) -> tuple[Any, PretrainedModelSpec, int]:
    torch = require_torch()
    torchvision = require_torchvision()
    registry = available_model_specs()
    if model_name not in registry:
        available = ", ".join(sorted(registry))
        raise SystemExit(f"Unknown model {model_name!r}. Available: {available}")
    spec = registry[model_name]
    model = spec.loader(torch, torchvision).to(device_name)
    model.eval()
    parameter_count = sum(int(parameter.numel()) for parameter in model.parameters())
    return model, spec, parameter_count


def build_eval_dataset(
    chips_path: Path,
    *,
    pair_frame: pd.DataFrame,
    query_splits: Sequence[str],
    gallery_splits: Sequence[str],
    modalities: Sequence[str],
    limit: int | None,
    gdal_prefix: Path | None,
    output_dtype: str,
    clip_min: float,
    clip_max: float,
    scale_max: float,
) -> tuple[TrainingChipDataset, pd.DataFrame]:
    eval_splits = tuple(sorted(set(query_splits).union(gallery_splits)))
    dataset = TrainingChipDataset.from_manifest(
        chips_path,
        splits=eval_splits,
        modalities=tuple(modalities),
        limit=None,
        gdal_prefix=gdal_prefix,
        output_dtype=output_dtype,
        clip_min=clip_min,
        clip_max=clip_max,
        scale_max=scale_max,
    )
    records_frame = dataset.records_frame().copy()
    filtered_pairs = filter_pairs_for_records(pair_frame, records_frame)
    if filtered_pairs.empty:
        raise SystemExit("No eval pairs overlap with the requested query/gallery splits.")

    chip_ids = chip_ids_from_pairs(filtered_pairs)
    records_frame = (
        records_frame[records_frame["chip_id"].astype(str).isin(chip_ids)]
        .sort_values(["split", "asset_id", "chip_id"])
        .reset_index(drop=True)
    )
    if limit is not None and len(records_frame) > limit:
        records_frame = records_frame.head(limit).reset_index(drop=True)
        filtered_pairs = filter_pairs_for_records(filtered_pairs, records_frame)
        if filtered_pairs.empty:
            raise SystemExit(
                "Eval chip limit removed all pair labels. Increase --eval-limit or disable it."
            )
        retained_chip_ids = chip_ids_from_pairs(filtered_pairs)
        records_frame = (
            records_frame[records_frame["chip_id"].astype(str).isin(retained_chip_ids)]
            .reset_index(drop=True)
        )

    dataset = TrainingChipDataset(
        OnDemandChipDataset(records_frame, gdal_prefix=gdal_prefix),
        output_dtype=output_dtype,
        clip_min=clip_min,
        clip_max=clip_max,
        scale_max=scale_max,
    )
    filtered_pairs = filter_pairs_for_records(filtered_pairs, dataset.records_frame())
    if filtered_pairs.empty:
        raise SystemExit("No pair labels remain after eval dataset filtering.")
    return dataset, filtered_pairs


def model_batch_inputs(
    torch: Any,
    batch: np.ndarray,
    *,
    input_size: int,
    mean: Sequence[float],
    std: Sequence[float],
    device_name: str,
) -> Any:
    tensor = torch.from_numpy(batch).to(device=device_name, dtype=torch.float32)
    if tensor.ndim != 4:
        raise ValueError(f"Expected batch in (N, C, H, W), got shape {tuple(tensor.shape)}")
    if tensor.shape[1] == 1:
        tensor = tensor.repeat(1, 3, 1, 1)
    elif tensor.shape[1] != 3:
        raise ValueError(f"Expected 1 or 3 channels, got {int(tensor.shape[1])}")
    if tensor.shape[-1] != input_size or tensor.shape[-2] != input_size:
        tensor = torch.nn.functional.interpolate(
            tensor,
            size=(input_size, input_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
    mean_tensor = torch.tensor(mean, dtype=torch.float32, device=device_name).view(1, 3, 1, 1)
    std_tensor = torch.tensor(std, dtype=torch.float32, device=device_name).view(1, 3, 1, 1)
    return (tensor - mean_tensor) / std_tensor


def embed_dataset(
    dataset: TrainingChipDataset,
    *,
    model: Any,
    spec: PretrainedModelSpec,
    batch_size: int,
    device_name: str,
    amp_enabled: bool,
) -> tuple[np.ndarray, pd.DataFrame, PretrainedBenchmarkReport]:
    torch = require_torch()
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    if device_name == "cuda":
        torch.cuda.reset_peak_memory_stats(torch.device(device_name))

    records: list[dict[str, object]] = []
    embeddings: list[np.ndarray] = []
    read_latencies: list[float] = []
    transform_latencies: list[float] = []
    model_prep_latencies: list[float] = []
    forward_latencies: list[float] = []
    total_latencies: list[float] = []

    start = perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(dataset), batch_size):
            batch_end = min(offset + batch_size, len(dataset))
            samples = [dataset.sample(index) for index in range(offset, batch_end)]
            batch = np.stack([sample.image for sample in samples], axis=0).astype(
                np.float32,
                copy=False,
            )

            prep_start = perf_counter()
            model_inputs = model_batch_inputs(
                torch,
                batch,
                input_size=spec.input_size,
                mean=spec.mean,
                std=spec.std,
                device_name=device_name,
            )
            model_prep_ms = (perf_counter() - prep_start) * 1000.0

            autocast_context = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if amp_enabled and device_name == "cuda"
                else None
            )
            forward_start = perf_counter()
            if autocast_context is None:
                outputs = forward_model(model, model_inputs, encoder_kind=spec.encoder_kind)
            else:
                with autocast_context:
                    outputs = forward_model(model, model_inputs, encoder_kind=spec.encoder_kind)
            outputs = torch.nn.functional.normalize(outputs, dim=1)
            forward_ms = (perf_counter() - forward_start) * 1000.0
            output_batch = outputs.detach().cpu().numpy().astype(np.float32, copy=False)
            per_sample_model_prep_ms = model_prep_ms / max(1, len(samples))
            per_sample_forward_ms = forward_ms / max(1, len(samples))

            for sample, embedding in zip(samples, output_batch, strict=True):
                records.append(
                    {
                        "chip_id": sample.record.chip_id,
                        "asset_id": sample.record.asset_id,
                        "capture_id": sample.record.capture_id,
                        "scene_id": sample.record.scene_id,
                        "split": sample.record.split,
                        "city": sample.record.city,
                        "modality": sample.record.modality,
                        "sensor": sample.record.sensor,
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
                model_prep_latencies.append(per_sample_model_prep_ms)
                forward_latencies.append(per_sample_forward_ms)
                total_latencies.append(
                    sample.timing.total_ms + per_sample_model_prep_ms + per_sample_forward_ms
                )
    elapsed_seconds = perf_counter() - start

    max_memory_mib = 0.0
    if device_name == "cuda":
        max_memory_mib = float(
            torch.cuda.max_memory_allocated(torch.device(device_name)) / (1024.0 * 1024.0)
        )
    matrix = np.stack(embeddings, axis=0) if embeddings else np.empty((0, 0), dtype=np.float32)
    report = PretrainedBenchmarkReport(
        model_name=spec.name,
        weights_label=spec.weights_label,
        device=device_name,
        device_name="CPU" if device_name == "cpu" else str(torch.cuda.get_device_name(0)),
        amp_enabled=bool(amp_enabled and device_name == "cuda"),
        samples=len(records),
        batch_size=batch_size,
        embedding_dim=int(matrix.shape[1]) if matrix.size else 0,
        parameter_count=sum(int(parameter.numel()) for parameter in model.parameters()),
        elapsed_seconds=elapsed_seconds,
        samples_per_second=safe_rate(len(records), elapsed_seconds),
        read_latency_ms_mean=mean(read_latencies),
        read_latency_ms_p95=percentile(read_latencies, 95.0),
        transform_latency_ms_mean=mean(transform_latencies),
        transform_latency_ms_p95=percentile(transform_latencies, 95.0),
        model_prep_latency_ms_mean=mean(model_prep_latencies),
        model_prep_latency_ms_p95=percentile(model_prep_latencies, 95.0),
        forward_latency_ms_mean=mean(forward_latencies),
        forward_latency_ms_p95=percentile(forward_latencies, 95.0),
        total_latency_ms_mean=mean(total_latencies),
        total_latency_ms_p95=percentile(total_latencies, 95.0),
        max_memory_mib=max_memory_mib,
    )
    return matrix, pd.DataFrame(records), report


def forward_model(model: Any, model_inputs: Any, *, encoder_kind: str) -> Any:
    if encoder_kind == "open_clip":
        return model.encode_image(model_inputs)
    return model(model_inputs)


def write_model_outputs(
    *,
    output_root: Path,
    spec: PretrainedModelSpec,
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    benchmark_report: PretrainedBenchmarkReport,
    retrieval_report: PairRetrievalReport,
) -> dict[str, Path]:
    model_root = output_root / spec.name
    model_root.mkdir(parents=True, exist_ok=True)

    embeddings_path = model_root / "embeddings.npy"
    index_path = model_root / "index.parquet"
    benchmark_path = model_root / "benchmark.json"
    retrieval_path = model_root / "retrieval.json"

    np.save(embeddings_path, embeddings)
    metadata.to_parquet(index_path, index=False, compression="zstd")
    benchmark_path.write_text(
        json.dumps(asdict(benchmark_report), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    retrieval_path.write_text(
        json.dumps(asdict(retrieval_report), indent=2, sort_keys=True),
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
    torch = require_torch()
    chips_path = args.chips_path.resolve()
    pairs_path = args.pairs_path.resolve()
    run_root = args.run_root.resolve()
    gdal_prefix = args.gdal_prefix.resolve() if args.gdal_prefix is not None else None
    # Import GDAL before torchvision/Pillow so the Kakadu runtime wins library resolution.
    load_gdal(gdal_prefix)
    _torchvision = require_torchvision()
    model_names = normalize_multi_arg(
        args.model,
        default=("resnet50", "remoteclip_rn50", "georsclip_vit_b32_ret2"),
    )
    query_splits = normalize_multi_arg(args.query_split, default=("val", "test"))
    gallery_splits = normalize_multi_arg(args.gallery_split, default=("val", "test"))
    modalities = normalize_multi_arg(args.modality, default=("PAN",))
    device_name, resolved_device_name = resolve_device(torch, args.device)
    pair_frame = pd.read_parquet(pairs_path)
    eval_dataset, filtered_pairs = build_eval_dataset(
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
    if len(eval_dataset) == 0:
        raise SystemExit("Evaluation dataset is empty after filtering.")

    run_root.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []
    for model_name in model_names:
        model, spec, parameter_count = load_pretrained_model(model_name, device_name=device_name)
        embeddings, metadata, benchmark_report = embed_dataset(
            eval_dataset,
            model=model,
            spec=spec,
            batch_size=args.batch_size,
            device_name=device_name,
            amp_enabled=args.amp,
        )
        retrieval_report = evaluate_pair_retrieval(
            embeddings,
            metadata,
            filtered_pairs,
            query_splits=query_splits,
            gallery_splits=gallery_splits,
        )
        paths = write_model_outputs(
            output_root=run_root,
            spec=spec,
            embeddings=embeddings,
            metadata=metadata,
            benchmark_report=benchmark_report,
            retrieval_report=retrieval_report,
        )
        summary_rows.append(
            {
                "model_name": spec.name,
                "weights_label": spec.weights_label,
                "parameter_count": parameter_count,
                "embedding_dim": benchmark_report.embedding_dim,
                "samples": benchmark_report.samples,
                "samples_per_second": benchmark_report.samples_per_second,
                "total_latency_ms_p95": benchmark_report.total_latency_ms_p95,
                "exact_recall_at_1": retrieval_report.exact_recall_at_1,
                "exact_recall_at_5": retrieval_report.exact_recall_at_5,
                "exact_recall_at_10": retrieval_report.exact_recall_at_10,
                "any_recall_at_1": retrieval_report.any_recall_at_1,
                "any_recall_at_5": retrieval_report.any_recall_at_5,
                "any_recall_at_10": retrieval_report.any_recall_at_10,
                "any_mean_reciprocal_rank": retrieval_report.any_mean_reciprocal_rank,
                "hard_negative_at_1_rate": retrieval_report.hard_negative_at_1_rate,
                "benchmark_path": str(paths["benchmark"]),
                "retrieval_path": str(paths["retrieval"]),
            }
        )
        print(
            f"model={spec.name} weights={spec.weights_label} "
            f"samples/s={benchmark_report.samples_per_second:.2f} "
            f"exact_R@10={retrieval_report.exact_recall_at_10:.3f} "
            f"any_R@10={retrieval_report.any_recall_at_10:.3f} "
            f"any_MRR={retrieval_report.any_mean_reciprocal_rank:.3f}"
        )
        del model
        if device_name == "cuda":
            torch.cuda.empty_cache()

    summary_frame = pd.DataFrame(summary_rows).sort_values(
        ["any_recall_at_10", "any_mean_reciprocal_rank"],
        ascending=False,
    )
    summary_json = run_root / "summary.json"
    summary_parquet = run_root / "summary.parquet"
    summary_json.write_text(
        json.dumps(
            {
                "chips_path": str(chips_path),
                "pairs_path": str(pairs_path),
                "device": device_name,
                "device_name": resolved_device_name,
                "query_splits": list(query_splits),
                "gallery_splits": list(gallery_splits),
                "models": summary_frame.to_dict(orient="records"),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    summary_frame.to_parquet(summary_parquet, index=False, compression="zstd")
    print(f"chips: {chips_path}")
    print(f"pairs: {pairs_path}")
    print(
        f"device: {device_name} ({resolved_device_name}) "
        f"amp={bool(args.amp and device_name == 'cuda')}"
    )
    print(f"eval_samples: {len(eval_dataset)} models={model_names}")
    print(f"summary_json: {summary_json}")
    print(f"summary_parquet: {summary_parquet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
