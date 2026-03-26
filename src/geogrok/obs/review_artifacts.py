from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from geogrok.io.raster import normalize_chip_array, read_chip_array

from .data import ObsDataPaths, chip_frame_with_strings, chip_record, default_data_paths, load_chips
from .quicklook import normalize_for_display, require_pillow

DEFAULT_ARTIFACT_ROOT = Path("artifacts/observability/review_artifacts")
DEFAULT_CODEC_PROFILE = "review_visually_lossless"


@dataclass(frozen=True)
class ReviewArtifactRecord:
    chip_id: str
    artifact_kind: str
    codec_profile: str
    media_type: str
    content_path: str
    width: int
    height: int
    channels: int
    bits_per_sample: int
    is_signed: bool
    generated_at: str
    file_size_bytes: int
    source_path: str
    source_window: dict[str, int]


def runtime_capabilities() -> dict[str, Any]:
    kakadu_js_dir = Path("web/static/kakadujs")
    return {
        "pykdu_available": load_pykdu() is not None,
        "kakadujs_assets_available": bool(
            (kakadu_js_dir / "kakadujs.js").exists() and (kakadu_js_dir / "kakadujs.wasm").exists()
        ),
        "artifact_root": str(DEFAULT_ARTIFACT_ROOT.resolve()),
        "codec_profile": DEFAULT_CODEC_PROFILE,
    }


def load_pykdu() -> Any | None:
    try:
        return importlib.import_module("pykdu")
    except ModuleNotFoundError:
        return None


def ensure_chip_review_artifact(
    chip: dict[str, Any],
    *,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    gdal_prefix: Path | None = None,
    codec_profile: str = DEFAULT_CODEC_PROFILE,
) -> ReviewArtifactRecord:
    artifact_dir = artifact_root / "chips"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    pykdu = load_pykdu()
    existing = _load_existing_record(chip["chip_id"], artifact_dir)
    if (
        existing is not None
        and Path(existing.content_path).exists()
        and (existing.artifact_kind == "j2c" or pykdu is None)
    ):
        return existing

    raster = read_chip_array(
        chip["local_path"],
        x0=int(chip["x0"]),
        y0=int(chip["y0"]),
        width=int(chip["width"]),
        height=int(chip["height"]),
        prefix=gdal_prefix,
    )
    array = normalize_chip_array(raster.array)

    if pykdu is not None:
        record = _write_j2c_artifact(
            chip=chip,
            array=array,
            artifact_dir=artifact_dir,
            codec_profile=codec_profile,
            pykdu=pykdu,
        )
    else:
        record = _write_png_artifact(
            chip=chip,
            array=array,
            artifact_dir=artifact_dir,
            codec_profile=codec_profile,
        )

    _write_record_json(record, artifact_dir)
    return record


def chip_review_artifact_payload(
    chips_frame: Any,
    *,
    chip_id: str,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    gdal_prefix: Path | None = None,
    codec_profile: str = DEFAULT_CODEC_PROFILE,
) -> dict[str, Any]:
    chip = chip_record(chips_frame, chip_id)
    record = ensure_chip_review_artifact(
        chip,
        artifact_root=artifact_root,
        gdal_prefix=gdal_prefix,
        codec_profile=codec_profile,
    )
    return {
        **asdict(record),
        "content_url": f"/api/chips/{chip_id}/review-artifact/content",
        "fallback_png_url": f"/api/chips/{chip_id}/image?size=320",
    }


def pair_review_artifact_payload(
    chips_frame: Any,
    *,
    pair_key: str,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    gdal_prefix: Path | None = None,
    codec_profile: str = DEFAULT_CODEC_PROFILE,
) -> dict[str, Any]:
    query_chip_id, candidate_chip_id = pair_key.split("__", maxsplit=1)
    return {
        "pair_key": pair_key,
        "query": chip_review_artifact_payload(
            chips_frame,
            chip_id=query_chip_id,
            artifact_root=artifact_root,
            gdal_prefix=gdal_prefix,
            codec_profile=codec_profile,
        ),
        "candidate": chip_review_artifact_payload(
            chips_frame,
            chip_id=candidate_chip_id,
            artifact_root=artifact_root,
            gdal_prefix=gdal_prefix,
            codec_profile=codec_profile,
        ),
        "fallback_png_url": f"/api/pairs/{pair_key}/image?size=224&gap=14",
    }


def load_chip_artifact_content(
    chips_frame: Any,
    *,
    chip_id: str,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    gdal_prefix: Path | None = None,
    codec_profile: str = DEFAULT_CODEC_PROFILE,
) -> tuple[bytes, ReviewArtifactRecord]:
    chip = chip_record(chips_frame, chip_id)
    record = ensure_chip_review_artifact(
        chip,
        artifact_root=artifact_root,
        gdal_prefix=gdal_prefix,
        codec_profile=codec_profile,
    )
    return Path(record.content_path).read_bytes(), record


def _write_j2c_artifact(
    *,
    chip: dict[str, Any],
    array: np.ndarray,
    artifact_dir: Path,
    codec_profile: str,
    pykdu: Any,
) -> ReviewArtifactRecord:
    plane = _artifact_plane(array)
    bit_depth = infer_bit_depth(plane)
    encoder = pykdu.Encoder(
        params="Cmodes=HT Qfactor=92",
        container="j2k",
        bit_depth=bit_depth,
    )
    payload = encoder.encode(plane)
    content_path = artifact_dir / f"{chip['chip_id']}.j2c"
    content_path.write_bytes(payload)
    return ReviewArtifactRecord(
        chip_id=str(chip["chip_id"]),
        artifact_kind="j2c",
        codec_profile=codec_profile,
        media_type="application/octet-stream",
        content_path=str(content_path.resolve()),
        width=int(plane.shape[1]),
        height=int(plane.shape[0]),
        channels=1,
        bits_per_sample=bit_depth,
        is_signed=bool(np.issubdtype(plane.dtype, np.signedinteger)),
        generated_at=_timestamp_now(),
        file_size_bytes=content_path.stat().st_size,
        source_path=str(chip["local_path"]),
        source_window={
            "x0": int(chip["x0"]),
            "y0": int(chip["y0"]),
            "width": int(chip["width"]),
            "height": int(chip["height"]),
        },
    )


def _write_png_artifact(
    *,
    chip: dict[str, Any],
    array: np.ndarray,
    artifact_dir: Path,
    codec_profile: str,
) -> ReviewArtifactRecord:
    image = require_pillow().fromarray(normalize_for_display(array, pmin=2.0, pmax=98.0), mode="L")
    content_path = artifact_dir / f"{chip['chip_id']}.png"
    image.save(content_path, format="PNG")
    return ReviewArtifactRecord(
        chip_id=str(chip["chip_id"]),
        artifact_kind="png",
        codec_profile=f"{codec_profile}_fallback",
        media_type="image/png",
        content_path=str(content_path.resolve()),
        width=int(image.width),
        height=int(image.height),
        channels=1,
        bits_per_sample=8,
        is_signed=False,
        generated_at=_timestamp_now(),
        file_size_bytes=content_path.stat().st_size,
        source_path=str(chip["local_path"]),
        source_window={
            "x0": int(chip["x0"]),
            "y0": int(chip["y0"]),
            "width": int(chip["width"]),
            "height": int(chip["height"]),
        },
    )


def _artifact_plane(array: np.ndarray) -> np.ndarray:
    plane = np.asarray(array[0])
    if plane.ndim != 2:
        raise ValueError(f"Expected a single-band 2D plane, got shape {plane.shape}")
    return np.ascontiguousarray(plane)


def infer_bit_depth(array: np.ndarray) -> int:
    if np.issubdtype(array.dtype, np.uint8):
        return 8
    if np.issubdtype(array.dtype, np.uint16):
        max_value = int(np.max(array))
        if max_value <= 0:
            return 1
        return max(1, min(16, int(max_value).bit_length()))
    if np.issubdtype(array.dtype, np.int16):
        max_value = int(np.max(np.abs(array)))
        if max_value <= 0:
            return 1
        return max(1, min(16, int(max_value).bit_length() + 1))
    return 16


def _record_json_path(chip_id: str, artifact_dir: Path) -> Path:
    return artifact_dir / f"{chip_id}.json"


def _write_record_json(record: ReviewArtifactRecord, artifact_dir: Path) -> None:
    path = _record_json_path(record.chip_id, artifact_dir)
    path.write_text(json.dumps(asdict(record), indent=2, sort_keys=True) + "\n")


def _load_existing_record(chip_id: str, artifact_dir: Path) -> ReviewArtifactRecord | None:
    path = _record_json_path(chip_id, artifact_dir)
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    return ReviewArtifactRecord(**payload)


def _timestamp_now() -> str:
    return datetime.now(UTC).isoformat()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build review artifacts for manifest-backed chips.",
    )
    parser.add_argument(
        "--chips-path",
        type=Path,
        default=None,
        help="Optional chips parquet path. Defaults to observability manifest discovery.",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT,
        help="Directory to store derived review artifacts.",
    )
    parser.add_argument(
        "--chip-id",
        type=str,
        default=None,
        help="Build a single chip artifact by chip_id.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=16,
        help="Maximum number of chip artifacts to build when --chip-id is not set.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.chips_path is not None:
        default_paths = default_data_paths()
        obs_paths = ObsDataPaths(
            chips_path=args.chips_path,
            pairs_path=default_paths.pairs_path,
        )
        chips = chip_frame_with_strings(
            load_chips(obs_paths)
        )
    else:
        chips = chip_frame_with_strings(load_chips(default_data_paths()))

    records = chips.to_dict(orient="records")
    if args.chip_id:
        records = [record for record in records if record["chip_id"] == args.chip_id]
    else:
        records = records[: max(0, args.limit)]

    built: list[dict[str, Any]] = []
    for record in records:
        artifact = ensure_chip_review_artifact(record, artifact_root=args.artifact_root)
        built.append(asdict(artifact))

    print(json.dumps({"count": len(built), "artifacts": built}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
